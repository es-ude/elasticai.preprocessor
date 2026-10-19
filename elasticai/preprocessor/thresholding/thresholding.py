from dataclasses import dataclass
from enum import Enum
from logging import Logger, getLogger
from pathlib import Path

import numpy as np

import elasticai.creator_plugins.thresholding.utils as hw_utils
from elasticai.creator_plugins.thresholding.src import c_compile
from elasticai.preprocessor._basics import DataTransient, PreprocessingModule


class ThresholdTargets(Enum):
    Constant = "constant"
    AbsoluteMean = "median_abs"
    MedianAbsoluteDeviation = "mad"
    MovingAverage = "mavg"
    MovingAverageAbsolute = "mavg_abs"
    RmsNorm = "rms_norm"
    RmsBlackrock = "rms_black"
    Welford = "welford"


@dataclass
class ThresholdSettings:
    """Dataclass for defining the funcs for determining properties to calculate thresholding
    Attributes:
        method:         Applied method for thresholding from TargetsThreshold
        sampling_rate:  Sampling rate of the transient signal [Hz]
        window_sec:     Window length in sec [Only applied in moving methods]
        thr_val:        Value for the constant threshold value (only for TargetsThreshold.Constant)
        do_quant:       Boolean for performing quantized operations
    """

    method: ThresholdTargets
    sampling_rate: float
    window_sec: float
    thr_val: float | int
    do_quant: bool

    @property
    def window_length(self) -> int:
        """Returning an integer with total number of samples for building the window sequence"""
        assert self.window_sec > 0, "Window length must be greater than zero"
        return int(abs(self.window_sec * self.sampling_rate))


DefaultThresholdSettings = ThresholdSettings(
    method=ThresholdTargets.Constant,
    sampling_rate=1000.0,
    window_sec=10e-3,
    thr_val=0.1,
    do_quant=False,
)


class Thresholding(PreprocessingModule[DataTransient, DataTransient]):
    _val_thr: int = 0

    def __init__(self, settings: ThresholdSettings) -> None:
        """Class for calculating the thresholding values based on the transient input signal
        :param settings:    Class SettingsThreshold for configuring the properties
        :return:            None
        """
        super().__init__()
        self._logger: Logger = getLogger(__name__)
        self._settings: ThresholdSettings = settings
        if isinstance(settings.method, str):
            self._settings.method = ThresholdTargets(settings.method)

        self._map_hardware = {
            ThresholdTargets.Constant: "const",
            ThresholdTargets.AbsoluteMean: "const",
            ThresholdTargets.MedianAbsoluteDeviation: "const",
            ThresholdTargets.MovingAverage: "mov_avg_norm",
            ThresholdTargets.MovingAverageAbsolute: "mov_avg_abs_norm",
            ThresholdTargets.RmsNorm: "const",
            ThresholdTargets.RmsBlackrock: "const",
            ThresholdTargets.Welford: "welford",
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
        path2save: Path,
        total_bitwidth: int,
        frac_bitwidth: int = 0,
        is_signed: bool = True,
    ) -> None:
        supported_targets = ["mcu", "pc", "fpga"]
        if target.lower() not in supported_targets:
            raise ValueError(f"Target {target} is not supported: only {supported_targets}")

        if target.lower() in ["mcu", "pc"]:
            self._create_design_c(
                id=id,
                bitwidth=total_bitwidth,
                signed=is_signed,
                path2save=path2save,
            )
        else:
            self._create_design_verilog(
                id=id,
                bitwidth=total_bitwidth,
                path2save=path2save,
            )

    def _create_design_c(
        self,
        id: str,
        bitwidth: int,
        signed: bool,
        path2save: Path,
    ) -> None:
        method_type = self._map_method_to_hardware()
        match method_type:
            case "welford":
                c_compile.build_thresholding_welford(
                    bitwidth=bitwidth,
                    signed=signed,
                    path2save=path2save,
                    thresholding_id=id,
                    define_path=".",
                )
            case "mov_avg_norm":
                if self._is_power_of_two(self._settings.window_length):
                    c_compile.build_thresholding_mavg_pow2(
                        log_size=int(np.log2(self._settings.window_length)),
                        window_size=self._settings.window_length,
                        bitwidth=bitwidth,
                        signed=signed,
                        path2save=path2save,
                        thresholding_id=id,
                        define_path=".",
                    )
                else:
                    c_compile.build_thresholding_mavg(
                        window_size=self._settings.window_length,
                        bitwidth=bitwidth,
                        signed=signed,
                        path2save=path2save,
                        thresholding_id=id,
                        define_path=".",
                    )
            case "mov_avg_abs_norm":
                if self._is_power_of_two(self._settings.window_length):
                    c_compile.build_thresholding_mavg_pow2_abs(
                        log_size=int(np.log2(self._settings.window_length)),
                        window_size=self._settings.window_length,
                        bitwidth=bitwidth,
                        signed=signed,
                        path2save=path2save,
                        thresholding_id=id,
                        define_path=".",
                    )
                else:
                    c_compile.build_thresholding_mavg_abs(
                        window_size=self._settings.window_length,
                        bitwidth=bitwidth,
                        signed=signed,
                        path2save=path2save,
                        thresholding_id=id,
                        define_path=".",
                    )
            case _:
                c_compile.build_thresholding_const(
                    threshold=self._val_thr,
                    bitwidth=bitwidth,
                    signed=signed,
                    path2save=path2save,
                    thresholding_id=id,
                    define_path=".",
                )

    def _create_design_verilog(
        self,
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
            case ThresholdTargets.Constant:
                params["params"].update({"CONST_THR": self._val_thr})
            case ThresholdTargets.MovingAverage:
                if self._is_power_of_two(self._settings.window_length):
                    params["type"] = "mov_avg_pow2"
                else:
                    params["type"] = "mov_avg_norm"
                params["params"].update({"LENGTH": self._settings.window_length})
            case ThresholdTargets.MovingAverageAbsolute:
                if self._is_power_of_two(self._settings.window_length):
                    params["type"] = "mov_avg_abs_pow2"
                else:
                    params["type"] = "mov_avg_abs_norm"
                params["params"].update({"LENGTH": self._settings.window_length})
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

    def forward(self, x: np.ndarray) -> np.ndarray:
        used_method = self._settings.method.value
        if used_method not in self._get_methods():
            raise AttributeError(
                f"Thresholding method {used_method} not available - Please change to {self._get_methods()}"
            )
        val = getattr(self, f"_thr_{used_method}")(x)
        data = np.floor(val).astype(int) if self._settings.do_quant else val
        self._val_thr = data[0]
        return data

    def __call__(self, x: DataTransient) -> DataTransient:
        return DataTransient(
            data=self.forward(x=x.data),
            sampling_rate=x.sampling_rate,
        )

    def get_threshold_position(self, xin: np.ndarray, pre_time: float = 0.0) -> np.ndarray:
        """Function for getting the crosspoints of thresholding value and transient input
        :param xin:         Numpy array with transient raw signal
        :param pre_time:    Floating value with pre-time in the window before event is detected [s]
        :return:            Numpy array with thresholding value from applied method
        """
        xthr = self.forward(x=xin)
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
        M = self._settings.window_length
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
