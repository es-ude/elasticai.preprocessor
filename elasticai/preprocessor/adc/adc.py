from dataclasses import dataclass
from fractions import Fraction
from logging import Logger, getLogger
from pathlib import Path

import numpy as np
from elasticai.creator.arithmetic import FxpArithmetic, FxpParams
from elasticai.creator_plugins.bram.utils import translate_path_to_int, write_mem_file
from scipy.signal import resample_poly

from elasticai.creator_plugins.player.utils import load_and_plugin
from elasticai.preprocessor._common_func import CommonDigitalFunctions


@dataclass
class SettingsResampler:
    """Settings for defining the properties of the Analog-Digital Converter (ADC).
    Attributes:
        total_bits:     Integer with total number of bits
        frac_bits:      Integer with fractional number of bits (0= only integer)
        is_signed:      Boolean if conversion will be signed or not
        srate_orig:     Float with sampling rate of the input data stream [Hz]
        srate_new:      Float with sampling rate of the output data stream [Hz]
        vneg:           Float with minimum negative input voltage value
        vpos:           Float with maximum positive input voltage value
    """

    total_bits: int
    frac_bits: int
    is_signed: bool
    srate_orig: float
    srate_new: float
    vneg: float
    vpos: float

    @property
    def vcm(self) -> float:
        """Returning the common mode voltage (mid voltage of the voltage ranges)"""
        return (self.vpos + self.vneg) / 2

    @property
    def lsb(self) -> float:
        """Returning the voltage value to represent the possible Least Significant Bit (LSB)"""
        return (self.vpos - self.vneg) / (2**self.total_bits)


class TransientResampler:
    _logger: Logger
    _funcs = CommonDigitalFunctions
    _arith = FxpArithmetic
    _settings: SettingsResampler

    def __init__(self, settings: SettingsResampler) -> None:
        """Class for resampling pre-recorded transient data to get a new data stream output with adapted characteristics
        :param settings:    Settings for defining the properties of the Analog-Digital Converter
        :return:            None
        """
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
        self._settings = settings

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
            xout = [self._arith.round_to_integer(val) for val in data]
            shape = _get_dtype(total_bits=self._settings.total_bits, is_signed=self._settings.is_signed)
        else:
            xout = [self._arith.round_to_rational(val) for val in data]
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
        xlsb = self._settings.lsb
        xmin = self._settings.vneg

        xin = np.round((self._clamp_analog(data) - xmin) / xlsb)
        xin = (xin + self._arith.minimum_as_integer) * self._arith._config.minimum_step_as_rational
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
            frac = Fraction(self._settings.srate_new / self._settings.srate_orig)
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

    def redefine_from_voltage(self, data: np.ndarray, is_int_output: bool = True) -> np.ndarray:
        """Function for translating the voltage transient data into digital data stream
        :param data:            Numpy array with voltage data [shape=(num_samples, )]
        :param is_int_output:   Boolean for getting the data output in integers else fxp
        :return:                Numpy array with digital data stream
        """
        xin = self._do_resample(data)
        return self._quantize_voltage(xin, is_int_output=is_int_output)

    def redefine_from_fxp(self, data: np.ndarray, is_int_output: bool = True) -> np.ndarray:
        """Function for translating the fixed-point transient data into new values
        :param data:            Numpy array with fxp data [shape=(num_samples, )]
        :param is_int_output:   Boolean for getting the data output in integers else fxp
        :return:                Numpy array with redefined digital data stream
        """
        xin = self._do_resample(data)
        return self._quantize_digital(xin, is_int_input=False, is_int_output=is_int_output)

    def redefine_from_int(self, data: np.ndarray, is_int_output: bool = True) -> np.ndarray:
        """Function for translating the integer transient data into new values
        :param data:            Numpy array with integer data [shape=(num_samples, )]
        :param is_int_output:   Boolean for getting the data output in integers else fxp
        :return:                Numpy array with redefined digital data stream
        """
        xin = self._do_resample(data)
        return self._quantize_digital(xin, is_int_input=True, is_int_output=is_int_output)

    def create_verilog_design(self, id: str, path2save: Path, data: np.ndarray, trgg: list = []) -> None:
        """Function for creating the Verilog designs to use pre-recorded in simulations
        :param id:          ID of Verilog designs
        :param path2save:   Path to the saved Verilog designs
        :param data:        Numpy array with transient data / frame used in Simulation [shape=(num_samples, ), type=int]
        :param trgg:        List with trigger output (event detection, ...) used in Simulation [shape=(num_samples, ), type=int]
        :return:            None
        """
        use_trgg = len(trgg) > 0
        if "int" not in data.dtype.name:  # pragma: no branch
            raise ValueError("Type of input data is not 'int'")
        if data.shape not in ((data.size,), (1, data.size)):  # pragma: no branch
            raise ValueError("shape")

        path2data = path2save / f"replayer_{id}_data.mem"
        write_mem_file(path=path2data, data=data.tolist(), bitwidth=self._settings.total_bits)
        path2trgg = path2save / f"replayer_{id}_trgg.mem"
        if use_trgg:
            write_mem_file(path=path2trgg, data=trgg, bitwidth=1)

        load_and_plugin(
            type="replayer",
            id=id,
            params={
                "BITWIDTH": self._settings.total_bits,
                "NUM_VALUES": data.size,
                "PATH2DATA": translate_path_to_int(path2data),
                "PATH2TRGG": translate_path_to_int(path2trgg),
                "ADD_TRIGGER": use_trgg,
            },
            packages=["player"],
            path2save=path2save,
        )
