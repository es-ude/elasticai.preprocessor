from dataclasses import dataclass
from enum import Enum
from logging import Logger, getLogger
from pathlib import Path

import numpy as np

import elasticai.creator_plugins.thresholding.utils as hw_utils
from elasticai.creator_plugins.thresholding.src import c_compile


class TargetsThreshold(Enum):
    Constant = "constant"
    AbsoluteMean = "median_abs"
    MedianAbsoluteDeviation = "mad"
    MovingAverage = "mavg"
    MovingAverageAbsolute = "mavg_abs"
    RmsNorm = "rms_norm"
    RmsBlackrock = "rms_black"
    Welford = "welford"


@dataclass
class SettingsThreshold:
    """Dataclass for defining the funcs for determining properties to calculate thresholding
    Attributes:
        method:         Applied method for thresholding from TargetsThreshold
        sampling_rate:  Sampling rate of the transient signal [Hz]
        window_sec:     Window length in sec [Only applied in moving methods]
        thr_val:        Value for the constant threshold value (only for TargetsThreshold.Constant)
        do_quant:       Boolean for performing quantized operations
    """

    method: TargetsThreshold
    sampling_rate: float
    window_sec: float
    thr_val: float | int
    do_quant: bool

    @property
    def window_steps(self) -> int:
        """Getting the stepsize of the window"""
        return int(self.window_sec * self.sampling_rate)


DefaultSettingsThreshold = SettingsThreshold(
    method=TargetsThreshold.Constant, sampling_rate=1000.0, window_sec=10e-3, thr_val=0.1, do_quant=False
)


class Thresholding:
    def __init__(self, settings: SettingsThreshold) -> None:
        """Class for calculating the thresholding values based on the transient input signal
        :param settings:    Class SettingsThreshold for configuring the properties
        :return:            None
        """
        self._logger: Logger = getLogger(__name__)
        self._settings: SettingsThreshold = settings
        if isinstance(settings.method, str):
            self._settings.method = TargetsThreshold(settings.method)
        self._map_hardware = {
            TargetsThreshold.Constant: "const",
            TargetsThreshold.AbsoluteMean: "const",
            TargetsThreshold.MedianAbsoluteDeviation: "const",
            TargetsThreshold.MovingAverage: "mov_avg_norm",
            TargetsThreshold.MovingAverageAbsolute: "mov_avg_abs_norm",
            TargetsThreshold.RmsNorm: "const",
            TargetsThreshold.RmsBlackrock: "const",
            TargetsThreshold.Welford: "welford",
        }

    def _map_method_to_hardware(self) -> str:
        if self._settings.method not in list(self._map_hardware.keys()):
            raise ValueError(f"Method '{self._settings.method}' in hardware generation is not supported.")
        return self._map_hardware[self._settings.method]

    @staticmethod
    def _is_power_of_two(M: int) -> bool:
        return M > 0 and (M & (M - 1)) == 0

    def create_design(
        self,
        id: str,
        target: str,
        bitwidth: int,
        path2save: Path,
        signed: bool,
        data: np.ndarray | None = None,
    ) -> int:
        """Create hardware design for thresholding.
        :param data:            Numpy array with transient signal
        :param id:              String with additional ID
        :param target:          String with target hardware type ["mcu", "pc", "fpga"]
        :param bitwidth:        Integer with bitwidth for target hardware type
        :param signed:          bool if input type signed ["signed", "unsigned"]
        :param path2save:       Path to save the design
        :return:                Integer value of the threshold value (for testing purpose)
        """
        supported_targets = ["mcu", "pc", "fpga"]
        if target.lower() not in supported_targets:
            raise ValueError(f"Target {target} is not supported: only {supported_targets}")

        if data is None:
            thr_val0 = 0
        else:
            thr_val0 = int(self.get_threshold(xin=data)[0])
        if target.lower() in ["mcu", "pc"]:
            self._create_design_c(
                thr_val=thr_val0,
                id=id,
                bitwidth=bitwidth,
                signed=signed,
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
        signed: bool,
        path2save: Path,
    ) -> None:
        method_type = self._map_method_to_hardware()
        match method_type:
            case TargetsThreshold.Welford:
                c_compile.build_thresholding_welford(
                    bitwidth=bitwidth,
                    signed=signed,
                    path2save=path2save,
                    thresholding_id=id,
                    define_path=".",
                )
            case TargetsThreshold.MovingAverage:
                if self._is_power_of_two(self._settings.window_steps):
                    c_compile.build_thresholding_mavg_pow2(
                        log_size=int(np.log2(self._settings.window_steps)),
                        window_size=self._settings.window_steps,
                        bitwidth=bitwidth,
                        signed=signed,
                        path2save=path2save,
                        thresholding_id=id,
                        define_path=".",
                    )
                else:
                    c_compile.build_thresholding_mavg(
                        window_size=self._settings.window_steps,
                        bitwidth=bitwidth,
                        signed=signed,
                        path2save=path2save,
                        thresholding_id=id,
                        define_path=".",
                    )
            case TargetsThreshold.MovingAverageAbsolute:
                if self._is_power_of_two(self._settings.window_steps):
                    c_compile.build_thresholding_mavg_pow2_abs(
                        log_size=int(np.log2(self._settings.window_steps)),
                        window_size=self._settings.window_steps,
                        bitwidth=bitwidth,
                        signed=signed,
                        path2save=path2save,
                        thresholding_id=id,
                        define_path=".",
                    )
                else:
                    c_compile.build_thresholding_mavg_abs(
                        window_size=self._settings.window_steps,
                        bitwidth=bitwidth,
                        signed=signed,
                        path2save=path2save,
                        thresholding_id=id,
                        define_path=".",
                    )
            case _:
                c_compile.build_thresholding_const(
                    threshold=thr_val,
                    bitwidth=bitwidth,
                    signed=signed,
                    path2save=path2save,
                    thresholding_id=id,
                    define_path=".",
                )

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
        match self._settings.method:
            case TargetsThreshold.Constant:
                params["params"].update({"CONST_THR": thr_val})
            case TargetsThreshold.MovingAverage:
                if self._is_power_of_two(self._settings.window_steps):
                    params["type"] = "mov_avg_pow2"
                else:
                    params["type"] = "mov_avg_norm"
                params["params"].update({"LENGTH": self._settings.window_steps})
            case TargetsThreshold.MovingAverageAbsolute:
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

    def _get_methods(self) -> list:
        split_key = "_thr_"
        return [method.split(split_key)[-1] for method in dir(self) if split_key in method]

    def get_threshold(self, xin: np.ndarray) -> np.ndarray:
        """Function for getting the thresholding value from input
        :param xin:     Numpy array with transient raw signal
        :return:        Numpy array with thresholding value from applied method
        """
        used_method = self._settings.method.value
        if used_method not in self._get_methods():
            raise ValueError(
                f"Thresholding method {used_method} not available - Please change to {self._get_methods()}"
            )
        val = getattr(self, f"_thr_{used_method}")(xin)
        return np.floor(val).astype(int) if self._settings.do_quant else val

    def get_threshold_position(self, xin: np.ndarray, pre_time: float = 0.0) -> np.ndarray:
        """Function for getting the crosspoints of thresholding value and transient input
        :param xin:         Numpy array with transient raw signal
        :param pre_time:    Floating value with pre-time in the window before event is detected [s]
        :return:            Numpy array with thresholding value from applied method
        """
        xthr = self.get_threshold(xin=xin)
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

    def _thr_constant(self, xin: np.ndarray) -> np.ndarray:
        return np.zeros_like(xin) + self._settings.thr_val

    def _thr_median_abs(self, xin: np.ndarray) -> np.ndarray:
        return np.zeros_like(xin) + np.median(np.abs(xin), axis=0)

    def _thr_mad(self, xin: np.ndarray) -> np.ndarray:
        median = np.median(xin, axis=0, keepdims=True)
        mad = np.median(np.abs(xin - median), axis=0, keepdims=True)
        std_estimate = mad / 0.6745
        threshold = std_estimate
        return np.zeros_like(xin) + threshold

    def _thr_mavg(self, xin: np.ndarray) -> np.ndarray:
        M = self._settings.window_steps
        xin_padded = np.pad(xin, (M - 1, 0), mode="constant")
        if np.issubdtype(xin.dtype, np.integer):
            window_sums = np.convolve(
                xin_padded.astype(np.int64), np.ones(M, dtype=np.int64), mode="valid"
            )
            return window_sums / M
        else:
            return np.convolve(xin_padded, np.ones(M) / M, mode="valid")

    def _thr_mavg_abs(self, xin: np.ndarray) -> np.ndarray:
        return self._thr_mavg(np.abs(xin))

    def _thr_rms_norm(self, xin: np.ndarray) -> np.ndarray:
        return np.zeros_like(xin) + np.sqrt(np.sum(xin**2) / xin.size)

    def _thr_rms_black(self, xin: np.ndarray) -> np.ndarray:
        return 4.5 * self._thr_rms_norm(xin)

    def _thr_welford(self, xin: np.ndarray) -> np.ndarray:
        n = 0
        mean = 0.0
        sigma = 0.0
        std_out = np.zeros_like(xin, dtype=float)

        for idx, x in enumerate(xin):
            n += 1
            mean_old = mean
            mean += (x - mean) / n
            sigma += ((x - mean) * (x - mean_old) - sigma) / n
            std_out[idx] = sigma

        std_out[0:1] = std_out[2]
        return np.sqrt(std_out)
