from dataclasses import dataclass
from enum import Enum
from logging import Logger, getLogger
from pathlib import Path

import numpy as np

import elasticai.creator_plugins.thresholding.utils as hw_utils


class TargetsThreshold(Enum):
    Const = "const"
    AbsoluteMean = "abs_mean"
    MedianAbsoluteDeviation = "mad"
    MovingAverage = "mavg"
    RmsNorm = "rms_norm"
    RmsMove = "rms_move"
    RmsBlackrock = "rms_black"
    Welford = "welford"


@dataclass
class SettingsThreshold:
    """Dataclass for defining the funcs for determining properties to calculate thresholding
    Attributes:
        method:         Applied method for thresholding ['const': constant given value,
                        'abs_mean': absolute mean value, 'mad': median absolute derivation, 'mavg', moving average,
                        'mavg_abs': moving absolute average, 'rms_norm': Root-Mean-Squared,
                        'rms_move': Moving RMS, 'rms_black': RMS method used in Blackrock Neurotechnology Systems,
                        'welford': Welford Online Algorithm for STD Calculation]
        sampling_rate:  Sampling rate of the transient signal [Hz]
        window_sec:     Window length in sec [Only applied in moving methods]
        do_quant:       Boolean for performing quantized operations
    """

    method: str
    sampling_rate: float
    window_sec: float
    do_quant: bool

    @property
    def window_steps(self) -> int:
        """Getting the stepsize of the window"""
        return int(self.window_sec * self.sampling_rate)


DefaultSettingsThreshold = SettingsThreshold(
    method="const", sampling_rate=1e3, window_sec=10e-3, do_quant=False
)


class Thresholding:
    def __init__(self, settings: SettingsThreshold) -> None:
        """Class for calculating the thresholding values based on the transient input signal
        :param settings:    Class SettingsThreshold for configuring the properties
        :return:            None
        """
        self._logger: Logger = getLogger(__name__)
        self._settings: SettingsThreshold = settings
        self._methods = {
            "const": "_constant",
            "abs_mean": "_absolute_median",
            "mad": "_median_absolute_derivation",
            "mavg": "_moving_average",
            "mavg_abs": "_moving_absolute_average",
            "rms_norm": "_root_mean_squared_normal",
            "rms_black": "_root_mean_squared_blackrock",
            "welford": "_welford_online",
        }
        self._hwmap = {
            "const": "const",
            "abs_mean": "const",
            "mad": "const",
            "mavg": "mov_avg_norm",
            "mavg_abs": "mov_avg_abs_norm",
            "rms_norm": "const",
            "rms_black": "const",
        }

    def _map_method_to_hardware(self) -> str:
        if self._settings.method.lower() not in list(self._hwmap.keys()):
            raise ValueError(f"Method '{self._settings.method}' in hardware generation is not supported.")
        return self._hwmap[self._settings.method]

    @staticmethod
    def _is_power_of_two(M: int) -> bool:
        return M > 0 and (M & (M - 1)) == 0

    def create_design(
        self, data: np.ndarray, id: str, target: str, bitwidth: int, path2save: Path, **kwargs
    ) -> int:
        """Create hardware design for thresholding.
        :param data:        Numpy array with transient signal
        :param id:          String with additional ID
        :param target:      String with target hardware type ["mcu", "pc", "fpga"]
        :param bitwidth:    Integer with bitwidth for target hardware type
        :param path2save:   Path to save the design
        :return:            Integer value of the threshold value (for testing purposes)
        """
        supported_targets = ["mcu", "pc", "fpga"]
        if target.lower() not in supported_targets:
            raise ValueError(f"Target {target} is not supported: only {supported_targets}")

        thr_val0 = int(self.get_threshold(xin=data, gain=1.0, **kwargs)[0])

        if target.lower() in ["mcu", "pc"]:
            self._create_design_c(
                thr_val=thr_val0,
                id=id,
                bitwidth=bitwidth,
                path2save=path2save,
            )
        else:
            self._create_design_verilog(
                thr_val=thr_val0,
                id=id,
                bitwidth=bitwidth,
                path2save=path2save,
            )
        return thr_val0

    def _create_design_c(
        self,
        thr_val: int,
        id: str,
        bitwidth: int,
        path2save: Path,
    ) -> None:

        raise NotImplementedError("Target 'mcu/pc' is currently not supported.")

    def _create_design_verilog(
        self,
        thr_val: int,
        id: str,
        bitwidth: int,
        path2save: Path,
    ) -> None:

        module_type = self._map_method_to_hardware()
        params = {
            "type": module_type,
            "id": id,
            "params": {
                "BITWIDTH": bitwidth,
            },
        }
        match self._settings.method.lower():
            case "const":
                params["params"].update({"CONST_THR": thr_val})
            case "mavg":
                if self._is_power_of_two(self._settings.window_steps):
                    params["type"] = "mov_avg_pow2"
                else:
                    params["type"] = "mov_avg_norm"
                params["params"].update({"LENGTH": self._settings.window_steps})
            case "mavg_abs":
                if self._is_power_of_two(self._settings.window_steps):
                    params["type"] = "mov_avg_abs_pow2"
                else:
                    params["type"] = "mov_avg_abs_norm"
                params["params"].update({"LENGTH": self._settings.window_steps})
            case _:
                raise NotImplementedError(
                    f"Threshold method '{self._settings.method}' does not have a Verilog implementation."
                )

        hw_utils.load_and_plugin(
            packages=["thresholding"],
            path2save=path2save,
            **params,
        )

    def _get_overview(self) -> list:
        return [key.lower() for key in self._methods.keys()]

    def get_threshold(self, xin: np.ndarray, gain: float = 1.0, **kwargs) -> np.ndarray:
        """Function for getting the thresholding value from input
        :param xin:     Numpy array with transient raw signal
        :param gain:    Float with gain applied on threshold output
        :return:        Numpy array with thresholding value from applied method
        """
        used_method = self._settings.method.lower()
        if used_method == "const" and "thr_val" not in list(kwargs.keys()):
            raise TypeError(
                "Constant threshold method needs the definition of 'thr_val' (threshold value) "
                "as float, like thr_val=0.5 in kwargs"
            )

        if used_method not in self._get_overview():
            raise ValueError(
                f"Thresholding method {used_method} not available - Please change to {self._get_overview()}"
            )
        val = gain * getattr(self, self._methods[used_method])(xin, **kwargs)
        return np.floor(val).astype(int) if self._settings.do_quant else val

    def get_threshold_position(
        self, xin: np.ndarray, pre_time: float = 0.0, gain: float = 1.0, **kwargs
    ) -> np.ndarray:
        """Function for getting the crosspoints of thresholding value and transient input
        :param xin:         Numpy array with transient raw signal
        :param gain:        Float with additional gain on threshold
        :param pre_time:    Floating value with pre-time in the window before event is detected [s]
        :return:            Numpy array with thresholding value from applied method
        """
        xthr = self.get_threshold(xin=xin, gain=gain, **kwargs)
        if xthr.min() < 0:
            pos = np.argwhere(xin < xthr).flatten()
        else:
            pos = np.argwhere(xin >= xthr).flatten()
        pos_pre = int(self._settings.sampling_rate * pre_time)
        return np.array(self._get_values_non_incremented_change(pos)) - pos_pre

    @staticmethod
    def _get_values_non_incremented_change(data: np.ndarray) -> list:
        """Returns values that are not incremented by one from the previous value.
        Always includes the first element.
        """
        data0 = data.tolist()
        if not data0:
            return []
        else:
            return [data0[0]] + [data0[i] for i in range(1, len(data0)) if data0[i] != data0[i - 1] + 1]

    def _constant(self, xin: np.ndarray, thr_val: float) -> np.ndarray:
        return np.zeros_like(xin) + thr_val

    def _absolute_median(self, xin: np.ndarray) -> np.ndarray:
        return np.zeros_like(xin) + np.median(np.abs(xin), axis=0)

    def _median_absolute_derivation(self, xin: np.ndarray) -> np.ndarray:
        median = np.median(xin, axis=0, keepdims=True)
        mad = np.median(np.abs(xin - median), axis=0, keepdims=True)
        std_estimate = mad / 0.6745
        threshold = std_estimate
        return np.zeros_like(xin) + threshold

    def _moving_average(self, xin: np.ndarray) -> np.ndarray:
        M = self._settings.window_steps
        xin_padded = np.pad(xin, (M - 1, 0), mode="constant")
        conv = np.convolve(xin_padded, np.ones(M) / M, mode="valid")
        return conv

    def _moving_absolute_average(self, xin: np.ndarray) -> np.ndarray:
        return self._moving_average(np.abs(xin))

    def _root_mean_squared_normal(self, xin: np.ndarray) -> np.ndarray:
        return np.zeros_like(xin) + np.sqrt(np.sum(xin**2) / xin.size)

    def _root_mean_squared_blackrock(self, xin: np.ndarray) -> np.ndarray:
        return 4.5 * self._root_mean_squared_normal(xin)

    def _welford_online(self, xin: np.ndarray) -> np.ndarray:
        n = 0
        mean = 0.0
        sigma = 0.0
        std_out = np.zeros_like(xin)

        for idx, x in enumerate(xin):
            n += 1
            mean_old = mean
            mean += (x - mean) / n
            sigma += ((x - mean) * (x - mean_old) - sigma) / n
            std_out[idx] = sigma

        std_out[0:1] = std_out[2]
        return np.sqrt(std_out)
