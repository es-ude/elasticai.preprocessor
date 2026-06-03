from dataclasses import dataclass
from logging import Logger, getLogger
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as scft
from elasticai.creator.arithmetic import FxpArithmetic, FxpConverter, FxpParams

import elasticai.creator_plugins.filter_data as hw_filters
from elasticai.preprocessor._common_func import CommonDigitalFunctions
from elasticai.preprocessor._plot_helper import (
    get_plot_color,
    get_textsize_paper,
    save_figure,
)


@dataclass
class FilterCoeffs:
    """Dataclass with filter coefficients
    Attributes:
        a: List with filter coefficients a
        b: List with filter coefficients b
    """

    a: list
    b: list


@dataclass
class SettingsFilter:
    """Configuration class for defining the filter processor
    Attributes:
        gain:       Integer with applied amplification factor [V/V]
        fs:         Sampling rate [Hz]
        n_order:    Integer with number of filter order or delay values in FIR-allpass
        f_filt:     List with filter frequencies [Hz] (low/high-pass: only one value - rest: two values)
        type:       String with selected filter algorithm ['iir', 'fir']
        f_type:     String with selected filter structure ['butter', 'cheby1', 'cheby2', 'ellip', 'bessel']
        b_type:     String with selected filter type ['lowpass', 'highpass', 'bandpass', 'bandstop', 'notch', 'allpass', 'simple_low']
    """

    gain: float
    fs: float
    n_order: int
    f_filt: list
    type: str
    f_type: str
    b_type: str


DefaultSettingsFilter = SettingsFilter(
    gain=1.0,
    fs=0.3e3,
    n_order=2,
    f_filt=[0.1, 100],
    type="iir",
    f_type="butter",
    b_type="bandpass",
)


class Filtering(CommonDigitalFunctions):
    __logger: Logger
    _type_supported: list = ["fir", "iir"]
    _btype_supported: list = [
        "lowpass",
        "highpass",
        "bandpass",
        "bandstop",
        "notch",
        "allpass",
        "simple_low",
    ]
    _ftype_supported: list = ["butter", "bessel", "cheby1", "cheby2", "ellip"]
    _coeff_a: np.ndarray
    _coeff_b: np.ndarray
    _settings: SettingsFilter

    def __init__(self, setting: SettingsFilter, use_filtfilt: bool = False):
        """Class for Emulating Digital Signal Processing on FPGA
        :param setting:         Class for handling the filter stage (using SettingsFilter)
        :param use_filtfilt:    Boolean for applying zero-phase filtering
        :return:                None
        """
        super().__init__()
        self.__logger = getLogger(__name__)
        self._settings = setting
        self.__use_filtfilt = use_filtfilt
        self.__process_filter()

    def get_coeffs(self) -> FilterCoeffs:
        """Getting the filter coefficients
        :return:            dataclass FilterCoeffs with filter coefficients
        """
        return FilterCoeffs(
            b=self._coeff_b.tolist(),
            a=self._coeff_a.tolist(),
        )

    def get_coeffs_quantized(self, bit_size: int) -> tuple[FilterCoeffs, FilterCoeffs]:
        """Quantize the coefficients with given bit fraction for adding into hardware designs
        :param bit_size:    Integer with total bitwidth
        :return:            dataclass FilterCoeffs with quantized filter coefficients
        """
        self.define_limits(
            total_bitwidth=bit_size,
            frac_bitwidth=bit_size - (1 if self._settings.type == "fir" else 2),
            bit_signed=True,
        )
        arith = FxpArithmetic(
            FxpParams(
                total_bits=bit_size,
                frac_bits=bit_size - (1 if self._settings.type == "fir" else 2),
                signed=True,
            )
        )
        if self._settings.type.lower() == "fir":
            quant_a = [1.0]
        else:
            quant_a = arith.cut_as_integer(self._coeff_a.tolist())
            quant_a = [arith._config.minimum_step_as_rational * val for val in quant_a]
        error_a = self._coeff_a - np.asarray(quant_a)

        quant_b = list()
        quant_b.extend(arith.cut_as_integer(self._coeff_b.tolist()))
        quant_b = [arith._config.minimum_step_as_rational * val for val in quant_b]
        error_b = self._coeff_b - np.asarray(quant_b)
        return FilterCoeffs(
            b=quant_b,
            a=quant_a,
        ), FilterCoeffs(b=error_b.tolist(), a=error_a.tolist())

    def get_coeffs_verilog_string(self, bitwidth: int, only_half_fir: bool = False) -> str:
        params: FilterCoeffs = self.get_coeffs_quantized(bit_size=bitwidth)[0]
        if self._settings.type.lower() == "fir":
            conv = FxpConverter(FxpParams(total_bits=bitwidth, frac_bits=bitwidth - 1, signed=True))
            used_params = params.b[: int(len(params.b) / 2 + 1)] if only_half_fir else params.b.copy()
            return conv.rational_to_hex_string_array_verilog(used_params)
        else:
            conv = FxpConverter(FxpParams(total_bits=bitwidth, frac_bits=bitwidth - 2, signed=True))
            used_params = params.b.copy()
            used_params.extend([-val for val in params.a[1:]])
            return conv.rational_to_hex_string_array_verilog(used_params)

    def __extract_filter_coeffs_iir(self) -> None:
        frange = np.array(self._settings.f_filt)
        match self._settings.b_type.lower():
            case "notch":
                filter = scft.iirnotch(
                    w0=float(self._settings.f_filt[0]),
                    Q=self._settings.n_order,
                    fs=self._settings.fs,
                )
                self._coeff_b = filter[0]
                self._coeff_a = filter[1]
            case "allpass":
                if self._settings.n_order == 1:
                    assert len(self._settings.f_filt) == 1, (
                        "f_filt should have length of 1 with [f_b] value"
                    )
                    val = np.tan(np.pi * frange[0] / self._settings.fs)
                    iir_c0 = (val - 1) / (val + 1)
                    self._coeff_b = np.array([iir_c0, 1.0])
                    self._coeff_a = np.array([1.0, iir_c0])
                elif self._settings.n_order == 2:
                    assert len(self._settings.f_filt) == 2, (
                        "f_filt should have length of 2 with [f_b, bandwidth] value"
                    )
                    val = np.tan(np.pi * frange[1] / self._settings.fs)
                    iir_c0 = (val - 1) / (val + 1)
                    iir_c1 = -np.cos(2 * np.pi * frange[0] / self._settings.fs)
                    self._coeff_b = np.array([-iir_c0, iir_c1 * (1 - iir_c0), 1.0])
                    self._coeff_a = np.array([1.0, iir_c1 * (1 - iir_c0), -iir_c0])
                else:
                    raise NotImplementedError
            case _:
                filter = scft.iirfilter(
                    N=self._settings.n_order,
                    Wn=frange[0] if len(frange) == 1 else frange,
                    fs=self._settings.fs,
                    ftype=self._settings.f_type.lower(),
                    btype=self._settings.b_type.lower(),
                    analog=False,
                    output="ba",
                )
                self._coeff_b = filter[0]
                self._coeff_a = filter[1]

    def __extract_filter_coeffs_fir(self) -> None:
        frange = np.array(self._settings.f_filt)
        self._coeff_a = np.array(1.0)
        match self._settings.b_type.lower():
            case "notch":
                assert len(self._settings.f_filt) == 2, (
                    "Size of f_filt should be 2 with [f_notch, bandwidth]"
                )
                freq = [
                    0,
                    frange[0] - frange[1],
                    frange[0],
                    frange[0] + frange[1],
                    self._settings.fs / 2,
                ]
                gain = [1, 1, 0, 1, 1]

                self._coeff_b = scft.firwin2(
                    numtaps=self._settings.n_order,
                    freq=freq,
                    gain=gain,
                    fs=self._settings.fs,
                )
            case "simple_low":
                self._coeff_a = np.array(1.0)
                self._coeff_b = np.array([0.5, 0.5])
            case "allpass":
                self._coeff_b = self._coeff_a
            case _:
                self._coeff_a = np.array(1.0)
                self._coeff_b = scft.firwin(
                    numtaps=self._settings.n_order,
                    cutoff=frange,
                    fs=self._settings.fs,
                    pass_zero=self._settings.b_type.lower(),
                )

    def __process_filter(self) -> None:
        assert self._settings.type.lower() in self._type_supported, (
            f"Type {self._settings.type} is not supported from {self._type_supported}"
        )
        assert self._settings.f_type.lower() in self._ftype_supported, (
            f"Filter type {self._settings.f_type} is not supported from {self._ftype_supported}"
        )
        assert self._settings.b_type.lower() in self._btype_supported, (
            f"Structure type {self._settings.b_type} is not supported from {self._btype_supported}"
        )
        self.__logger.debug(
            f"Build {self._settings.type.upper()} filter: {self._settings.b_type}, {self._settings.f_type}"
        )

        if self._settings.type.lower() == "iir":
            self.__extract_filter_coeffs_iir()
        elif self._settings.type.lower() == "fir":
            self.__extract_filter_coeffs_fir()

    def filt(self, xin: np.ndarray) -> np.ndarray:
        """Apply filter structure on transient input data
        :param xin:     Numpy array with transient input data
        :return:        Numpy array with filtered data
        """
        if self._settings.type.lower() == "fir" and self._settings.b_type.lower() == "allpass":
            mat = np.zeros(shape=(self._settings.n_order,), dtype=float)
            xout = np.concatenate((mat, xin[0 : xin.size - self._settings.n_order]), axis=None)

        elif not self.__use_filtfilt:
            xout = self._settings.gain * scft.lfilter(b=self._coeff_b, a=self._coeff_a, x=xin)
        else:
            xout = self._settings.gain * scft.filtfilt(b=self._coeff_b, a=self._coeff_a, x=xin)
        return xout

    def filt_quantized(
        self,
        xin: np.ndarray,
        total_bitwidth: int,
        fraction_width: int,
        is_signed: bool = True,
    ) -> np.ndarray:
        """Apply filter structure on transient input data
        :param xin:                 Numpy array with transient data
        :param total_bitwidth:      Integer with total bitwidth
        :param fraction_width:      Integer with fraction width
        :param is_signed:           Boolean with whether to sign the coefficients
        :return:                    Numpy array with filtered and quantized data
        """
        self.define_limits(
            bit_signed=is_signed,
            total_bitwidth=total_bitwidth,
            frac_bitwidth=fraction_width,
        )
        if self._settings.type.lower() == "fir" and self._settings.b_type.lower() == "allpass":
            xin_fxp = self._quantize_fxp(xin)
            xout = self.filt(xin_fxp)
        else:
            params = self.get_coeffs_quantized(bit_size=total_bitwidth)[0]
            self._coeff_b = np.asarray(params.b)
            self._coeff_a = np.asarray(params.a)

            xin_fxp = self._quantize_fxp(xin)
            filt = self.filt(xin_fxp)
            xout = (
                self._quantize_fxp(filt)
                - (3 if self._settings.type.lower() == "iir" else (filt < 0)) * 2**-fraction_width
            )
        return xout

    def __get_frequency_behaviour(self, num_points: int = 1001) -> tuple[np.ndarray, np.ndarray]:
        if self._settings.type == "iir":
            frange = np.array(self._settings.f_filt)
            filter = scft.iirfilter(
                N=self._settings.n_order,
                Wn=frange[0] if len(frange) == 1 else frange,
                ftype=self._settings.f_type.lower(),
                btype=self._settings.b_type.lower(),
                analog=True,
                output="ba",
            )
            return scft.freqs(b=filter[0], a=filter[1], worN=num_points)
        else:
            return scft.freqs(b=self._coeff_b, a=1, worN=num_points)

    def plot_freq_response(
        self, num_points: int = 1001, show_plot: bool = True, path2save: str = ""
    ) -> None:
        """Function for plotting the frequency response of desired filter type
        :param num_points:  Number of points to plot
        :param show_plot:   Boolean for showing plot
        :param path2save:   Path to save figure
        """
        w, h = self.__get_frequency_behaviour(num_points=num_points)
        f = w / (2 * np.pi) if self._settings.f_filt == "fir" else w

        fig1, ax11 = plt.subplots()
        plt.title("Frequency response")
        amplit_log = 20 * np.log10(np.abs(h))
        plt.semilogx(f, amplit_log, color=get_plot_color(0), label="Gain")
        plt.ylabel(
            r"Amplitude |$H(\omega)$| (dB)",
            size=get_textsize_paper(),
            color=get_plot_color(0),
        )
        plt.xlabel(r"Frequency $f_\mathrm{sig}$ (Hz)", size=get_textsize_paper())
        plt.xlim([f[0], f[-1]])

        ax11.grid(True, which="both", ls="--")
        ax11.twinx()

        phase = np.angle(h, deg=True)
        plt.semilogx(f, phase, color=get_plot_color(1), label="Phase", alpha=0.6)
        plt.ylabel(r"Phase $\alpha$ (°)", size=get_textsize_paper(), color=get_plot_color(1))
        plt.tight_layout()
        if path2save:
            save_figure(plt, path2save, "freq_response")
        plt.show(block=show_plot)

    def plot_grp_delay(self, num_points: int = 1001, show_plot: bool = False) -> None:
        """Plotting the Group Delay of filter
        :param num_points:  Number of points to plot
        :param show_plot:   Boolean for showing plot
        :return:            None
        """
        w, h = self.__get_frequency_behaviour(num_points=num_points)
        f = w / (2 * np.pi)
        phase = np.unwrap(np.angle(h)) / np.pi * 180
        grp_dly = -np.diff(phase) / np.diff(w)

        plt.figure()
        plt.semilogx(f[2:], grp_dly[1:], "k", linewidth=1)
        plt.ylabel(r"Group Delay $\tau_\mathrm{grp}$ (s)")
        plt.xlabel(r"Frequency $f_\mathrm{sig}$ (Hz)")
        plt.grid()
        plt.tight_layout()
        plt.show(block=show_plot)

    def create_design(self, target: str, bitwidth: int, id: str, path2save: Path) -> None:
        """Creating the hardware design for executing on specific target
        :param target:      String with target name ["mcu", "pc", "fpga"]
        :param bitwidth:    Integer with total bitwidth
        :param id:          String with unique identifier of device (appended to the name)
        :param path2save:   Path to save the hardware files
        :return:            None
        """
        supported_targets = ["mcu", "pc", "fpga"]
        if target.lower() not in supported_targets:
            raise ValueError(f"Target {target} is not supported: only {supported_targets}")
        if target.lower() in ["mcu", "pc"]:
            self._create_design_c()
        else:
            self._create_design_verilog(id=id, bitwidth=bitwidth, path2save=path2save, num_mult=1)

    def _create_iir_biquad_verilog(
        self, id: str, bitwidth: int, use_dsp_mult: bool, num_mult: int = 1
    ) -> dict:
        filt_params = self.get_coeffs_verilog_string(bitwidth=bitwidth, only_half_fir=True)
        return {
            "type": "biquad_df1",
            "id": id,
            "params": {
                "BITWIDTH": bitwidth,
                "LENGTH": self._settings.n_order,
                "NUM_MULT": num_mult,
                "FILT_COEFFS": filt_params,
            },
            "add_ringbuffer": False,
            "add_mac": True,
            "use_dsp_mult": use_dsp_mult,
        }

    def _create_fir_delay_verilog(self, id: str, bitwidth: int) -> dict:
        return {
            "type": "fir_delay",
            "id": id,
            "params": {
                "BITWIDTH": bitwidth,
                "LENGTH": self._settings.n_order,
            },
            "add_ringbuffer": True,
            "add_mac": False,
            "use_dsp_mult": False,
        }

    def _create_fir_full_verilog(
        self, id: str, bitwidth: int, use_dsp_mult: bool, num_mult: int = 1
    ) -> dict:
        filt_params = self.get_coeffs_verilog_string(bitwidth=bitwidth, only_half_fir=False)
        return {
            "type": "fir_full",
            "id": id,
            "params": {
                "BITWIDTH": bitwidth,
                "LENGTH": self._settings.n_order,
                "NUM_MULT": num_mult,
                "FILT_COEFFS": filt_params,
            },
            "add_ringbuffer": True,
            "add_mac": True,
            "use_dsp_mult": use_dsp_mult,
        }

    def _create_fir_half_verilog(
        self, id: str, bitwidth: int, use_dsp_mult: bool, num_mult: int = 1
    ) -> dict:
        filt_params = self.get_coeffs_verilog_string(bitwidth=bitwidth, only_half_fir=True)
        return {
            "type": "fir_half",
            "id": id,
            "params": {
                "BITWIDTH": bitwidth,
                "LENGTH": int(self._settings.n_order / 2) + 1,
                "NUM_MULT": num_mult,
                "FILT_COEFFS": filt_params,
            },
            "add_ringbuffer": True,
            "add_mac": True,
            "use_dsp_mult": use_dsp_mult,
        }

    def _create_fir_simple_lowpass_verilog(self, id: str, bitwidth: int) -> dict:
        return {
            "type": "fir_low",
            "id": id,
            "params": {"BITWIDTH": bitwidth},
            "add_ringbuffer": False,
            "add_mac": False,
            "use_dsp_mult": False,
        }

    def _create_design_verilog(self, id: str, bitwidth: int, path2save: Path, num_mult: int = 1) -> None:
        if self._settings.type.lower() == "iir":
            if self._settings.n_order not in [2]:
                raise ValueError(
                    f"IIR filter order {self._settings.n_order} is not supported for biquad filter"
                )
            if self._settings.b_type.lower() in ["simple_lowpass"]:
                raise ValueError(
                    f"IIR filter type {self._settings.b_type} is not supported for biquad filter"
                )
            params = self._create_iir_biquad_verilog(
                id=id, bitwidth=bitwidth, use_dsp_mult=True, num_mult=num_mult
            )
        elif self._settings.type.lower() == "fir":
            if self._settings.b_type.lower() not in ["allpass", "simple_low"]:
                if self._settings.n_order % 2 == 1:
                    params = self._create_fir_half_verilog(id, bitwidth, True, num_mult)
                else:
                    params = self._create_fir_full_verilog(id, bitwidth, True, num_mult)
            else:
                if self._settings.b_type.lower() == "allpass":
                    params = self._create_fir_delay_verilog(id, bitwidth)
                elif self._settings.b_type.lower() == "simple_low":
                    params = self._create_fir_simple_lowpass_verilog(id, bitwidth)
                else:
                    raise ValueError(f"FIR filter type {self._settings.b_type} is not supported")
        else:
            raise ValueError(f"Filter type {self._settings.type} is not supported")

        hw_filters.load_and_plugin(packages=["filter_data"], path2save=path2save, **params)

    def _create_design_c(self) -> None:
        raise NotImplementedError
