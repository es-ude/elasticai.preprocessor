from dataclasses import dataclass, replace
from enum import Enum
from fractions import Fraction
from logging import Logger, getLogger
from pathlib import Path

import numpy as np
from elasticai.creator.arithmetic import FxpArithmetic, FxpParams
from elasticai.creator_plugins.bram.utils import translate_path_to_int, write_mem_file
from scipy.signal import resample_poly

from elasticai.preprocessor._basics import DataTransient, PreprocessingModule
from elasticai.preprocessor._common_func import CommonDigitalFunctions


class ConversionResamplerTargets(Enum):
    FromInteger = "integer"
    FromFxp = "fxp"
    FromVoltage = "voltage"


@dataclass
class ConversionResamplerSettings:
    """Settings for defining the properties of the Analog-Digital Converter (ADC).
    Attributes:
        method:         Defining the input data type as ResamplerTargets
        total_bits:     Integer with total number of bits
        frac_bits:      Integer with fractional number of bits (0= only integer)
        is_signed:      Boolean if conversion will be signed or not
        is_int_output:  Boolean for deciding if the output representation is integer (True) or fixed_point (False)
        srate_orig:     Float with sampling rate of the input data stream [Hz]
        srate_new:      Float with sampling rate of the output data stream [Hz]
        vneg:           Float with minimum negative input voltage value
        vpos:           Float with maximum positive input voltage value
    """

    method: ConversionResamplerTargets
    total_bits: int
    frac_bits: int
    is_signed: bool
    is_int_output: bool
    srate_orig: float
    srate_new: float
    vneg: float = -1.0
    vpos: float = 1.0

    @property
    def vcm(self) -> float:
        """Returning the common mode voltage (mid voltage of the voltage ranges)"""
        return (self.vpos + self.vneg) / 2

    @property
    def lsb(self) -> float:
        """Returning the voltage value to represent the possible Least Significant Bit (LSB)"""
        return (self.vpos - self.vneg) / (2**self.total_bits)


ConversionResamplerSettingsDefault = ConversionResamplerSettings(
    method=ConversionResamplerTargets.FromVoltage,
    total_bits=8,
    frac_bits=6,
    is_signed=False,
    is_int_output=False,
    srate_orig=0.0,
    srate_new=0.0,
    vneg=0.0,
    vpos=1.8,
)


class ConversionResampler(PreprocessingModule[DataTransient, DataTransient]):
    _logger: Logger
    _funcs: CommonDigitalFunctions
    _arith: FxpArithmetic
    _settings: ConversionResamplerSettings
    _data: np.ndarray
    _trgg: list[int]

    def __init__(self, settings: ConversionResamplerSettings) -> None:
        """Class for resampling pre-recorded transient data to get a new data stream output with adapted characteristics
        :param settings:    Settings for defining the properties of the Analog-Digital Converter
        :return:            None
        """
        super().__init__()
        self._logger = getLogger(__name__)
        self._funcs = CommonDigitalFunctions()
        self._funcs.define_limits(
            bit_signed=settings.is_signed,
            total_bitwidth=settings.total_bits,
            frac_bitwidth=settings.frac_bits,
        )
        self._arith = FxpArithmetic(
            FxpParams(
                total_bits=settings.total_bits, frac_bits=settings.frac_bits, signed=settings.is_signed
            )
        )
        self._settings = replace(settings)
        self._trgg = []
        self._settings = settings
        if isinstance(settings.method, str):
            self._settings.method = ConversionResamplerTargets(settings.method.lower())

    def _clamp_analog(self, data: np.ndarray) -> np.ndarray:
        return np.clip(a=data, a_min=self._settings.vneg, a_max=self._settings.vpos)

    def _clamp_digital(self, data: np.ndarray, use_integer: bool) -> np.ndarray:
        if use_integer:
            return np.clip(
                a=data, a_min=self._arith.minimum_as_integer, a_max=self._arith.maximum_as_integer
            )
        else:
            return np.clip(
                a=data, a_min=self._arith.minimum_as_rational, a_max=self._arith.maximum_as_rational
            )

    def _quantize(self, data: np.ndarray, is_int_output: bool) -> np.ndarray:
        def _get_dtype(total_bits: int, is_signed: bool) -> np.dtype:
            for bits in (8, 16, 32, 64):
                if total_bits <= bits:
                    return np.dtype(f"{'int' if is_signed else 'uint'}{bits}")
            raise AttributeError(f"Unknown datatype for total_bits = {total_bits}")

        if is_int_output:
            xout = list(map(self._arith.round_to_integer, data))
            shape = _get_dtype(total_bits=self._settings.total_bits, is_signed=self._settings.is_signed)
        else:
            xout = [self._arith.round_to_rational(float(val)) for val in data]
            shape = np.float32
        return np.asarray(xout, dtype=shape)

    def _quantize_digital(self, data: np.ndarray, is_int_input: bool, is_int_output: bool) -> np.ndarray:
        xin = self._clamp_digital(data=data, use_integer=is_int_input)
        if is_int_input:
            xin = (self._arith._config.minimum_step_as_rational * xin).tolist()
        else:
            xin = xin.tolist()
        return self._quantize(data=xin, is_int_output=is_int_output)

    def _quantize_voltage(self, data: np.ndarray, is_int_output: bool) -> np.ndarray:
        xlsb_ana = self._settings.lsb
        xlsb_fxp = self._arith._config.minimum_step_as_rational
        xmin = self._settings.vneg

        xin = np.round((self._clamp_analog(data) - xmin) / xlsb_ana) + self._arith.minimum_as_integer
        xin = self._clamp_digital(xin, use_integer=True) * xlsb_fxp
        return self._quantize(data=xin, is_int_output=is_int_output)

    def _do_resample(self, data: np.ndarray) -> np.ndarray:
        if (
            self._settings.srate_new == 0.0
            or data.size == 1
            or self._settings.srate_new == self._settings.srate_orig
        ):
            self._logger.debug("No resampling necessary")
            return data
        else:
            frac = Fraction(int(self._settings.srate_new) / int(self._settings.srate_orig))
            p_ratio = frac.numerator
            q_ratio = frac.denominator

            xoff = data[0]
            xin = np.subtract(data, xoff)
            xrm = resample_poly(
                x=xin, up=p_ratio, down=q_ratio, axis=-1, padtype="mean", window="hamming"
            )
            self._logger.debug(
                f"Apply upsampling with sampling rate {self._settings.srate_new} (p={p_ratio}, q={q_ratio})"
            )
            return np.add(xrm, xoff)

    def do_cut_transient(
        self, data: np.ndarray, t_range_sec: list[float], use_srate_orig: bool = True
    ) -> np.ndarray:
        """Cutting the transient data array to defined time range
        :param data:            Numpy array with transient data [shape=(num_samples, )]
        :param t_range_sec:     List with time value [start, stop] or empty
        :param use_srate_orig:  Boolean for taking the original sampling rate or new one
        :return:                Numpy array with cutted transient data
        """
        if self._settings.srate_orig == 0.0:  # pragma: no branch
            raise ValueError("Sampling rate (orig) is zero")
        if self._settings.srate_new == 0.0:  # pragma: no branch
            raise ValueError("Sampling rate (new) is zero")

        if len(t_range_sec) == 0:
            return data
        elif len(t_range_sec) == 2:
            if t_range_sec[0] > t_range_sec[1]:  # pragma: no branch
                raise ValueError("Wrong time order in t_range_sec")

            srate_used = self._settings.srate_orig if use_srate_orig else self._settings.srate_new
            idx0 = int(t_range_sec[0] * srate_used)
            idx1 = int(t_range_sec[1] * srate_used)
            if idx0 > data.size and idx1 > data.size:  # pragma: no branch
                raise ValueError(
                    f"t_range_sec ({t_range_sec}) is out-of-range [0., {data.size / srate_used}]"
                )
            return data[idx0:idx1]
        else:
            raise ValueError(f"t_range should be empty or have a length of 2 (not {len(t_range_sec)})")

    def do_cut_labels(
        self,
        label_id: np.ndarray,
        label_pos: np.ndarray,
        t_range_sec: list[float],
        use_srate_orig: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Cutting the labels to defined time range
        :param label_id:        Numpy array with label ID of the detected event [shape=(num_events, )]
        :param label_pos:       Numpy array with the position of each label ID [shape=(num_events, )]
        :param t_range_sec:     List with time value [start, stop] or empty
        :param use_srate_orig:  Boolean for taking the original sampling rate or new one
        :return:                Tuple with reduced numpy arrays: (0) id, (1) pos
        """
        if label_id.size != label_pos.size:  # pragma: no branch
            raise ValueError("label_id and label_pos must have the same size")
        if self._settings.srate_orig == 0.0:  # pragma: no branch
            raise ValueError("Sampling rate (orig) is zero")
        if self._settings.srate_new == 0.0:  # pragma: no branch
            raise ValueError("Sampling rate (new) is zero")

        if len(t_range_sec) == 0:
            return label_id, label_pos
        elif len(t_range_sec) == 2:
            if t_range_sec[0] > t_range_sec[1]:  # pragma: no branch
                raise ValueError("Wrong time order in t_range_sec")

            srate_used = self._settings.srate_orig if use_srate_orig else self._settings.srate_new
            time0 = label_pos / srate_used
            idx0 = np.argwhere(time0 >= t_range_sec[0]).flatten()[0]
            idx1 = np.argwhere(time0 >= t_range_sec[1]).flatten()[0]
            if idx0 > label_pos[-1] and idx1 > label_pos[-1]:  # pragma: no branch
                raise ValueError(
                    f"t_range_sec ({t_range_sec}) is out-of-range [0., {label_pos[-1] / srate_used}]"
                )
            return label_id[idx0:idx1], label_pos[idx0:idx1]
        else:
            raise ValueError(f"t_range should be empty or have a length of 2 (not {len(t_range_sec)})")

    def _redefine_from_voltage(self, data: np.ndarray) -> np.ndarray:
        return self._quantize_voltage(data, is_int_output=self._settings.is_int_output)

    def _redefine_from_fxp(self, data: np.ndarray) -> np.ndarray:
        return self._quantize_digital(
            data, is_int_input=False, is_int_output=self._settings.is_int_output
        )

    def _redefine_from_int(self, data: np.ndarray) -> np.ndarray:
        return self._quantize_digital(data, is_int_input=True, is_int_output=self._settings.is_int_output)

    def forward(self, x: np.ndarray) -> np.ndarray:
        xin = self._do_resample(x)
        match self._settings.method:
            case ConversionResamplerTargets.FromFxp:
                self._data = self._redefine_from_fxp(xin)
            case ConversionResamplerTargets.FromInteger:
                self._data = self._redefine_from_int(xin)
            case ConversionResamplerTargets.FromVoltage:
                self._data = self._redefine_from_voltage(xin)
            case _:
                raise AttributeError
        return self._data

    def __call__(self, data: DataTransient) -> DataTransient:
        self._settings.srate_orig = data.sampling_rate
        srate_out = self._settings.srate_new or self._settings.srate_orig
        return DataTransient(data=self.forward(x=data.data), sampling_rate=srate_out)

    def add_trigger_list(self, trgg: list) -> None:
        """Add trigger list for the hardware design
        :param trgg:    List with trigger positions
        """
        self._trgg = trgg

    def create_design(
        self,
        id: str,
        target: str,
        path2save: Path,
        total_bitwidth: int = 0,
        frac_bitwidth: int = 0,
        is_signed: bool = False,
    ):
        if self._data is None:  # pragma: no branch
            raise ValueError("No data is available - Please apply data to the forward func")
        if self._settings.is_int_output:
            data = self._data
        else:
            data = self._quantize_digital(self._data, is_int_input=False, is_int_output=True)
        if "int" not in data.dtype.name:  # pragma: no branch
            raise ValueError("Type of input data is not 'int'")
        if data.shape not in ((data.size,), (1, data.size)):  # pragma: no branch
            raise ValueError("shape")

        supported_targets = ["mcu", "pc", "fpga"]
        if target.lower() not in supported_targets:
            raise ValueError(f"Target {target} is not supported: only {supported_targets}")
        if target.lower() in ["mcu", "pc"]:
            self._create_design_c(id=id, path2save=path2save, data=data.tolist(), trgg=self._trgg)
        else:
            self._create_design_verilog(id=id, path2save=path2save, data=data.tolist(), trgg=self._trgg)

    def _create_design_c(
        self, id: str, path2save: Path, data: list, trgg: list, define_path: str = "src"
    ):
        from elasticai.creator_plugins.adc.src import c_compile

        use_trgg = len(self._trgg) > 0
        if use_trgg:
            c_compile.build_replayer_with_trigger(
                replayer_id=id,
                define_path=define_path,
                data=data,
                trigger=trgg,
                bitwidth=self._settings.total_bits,
                signed=self._settings.is_signed,
                path2save=path2save,
            )
        else:
            c_compile.build_replayer(
                replayer_id=id,
                define_path=define_path,
                data=data,
                bitwidth=self._settings.total_bits,
                signed=self._settings.is_signed,
                path2save=path2save,
            )

    def _create_design_verilog(self, id: str, path2save: Path, data: list, trgg: list) -> None:
        from elasticai.creator_plugins.adc.utils import load_and_plugin

        use_trgg = len(self._trgg) > 0

        path2data = path2save / f"replayer_{id}_data.mem"
        write_mem_file(path=path2data, data=data, bitwidth=self._settings.total_bits)
        path2trgg = path2save / f"replayer_{id}_trgg.mem"
        if use_trgg:
            write_mem_file(path=path2trgg, data=trgg, bitwidth=1)

        load_and_plugin(
            type="replayer",
            id=id,
            params={
                "BITWIDTH": self._settings.total_bits,
                "NUM_VALUES": len(data),
                "PATH2DATA": translate_path_to_int(path2data),
                "PATH2TRGG": translate_path_to_int(path2trgg),
                "ADD_TRIGGER": use_trgg,
            },
            packages=["adc"],
            path2save=path2save,
        )
