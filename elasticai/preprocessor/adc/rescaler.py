from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
from elasticai.creator.arithmetic import FxpArithmetic, FxpParams

from elasticai.preprocessor._basics import DataTransient, PreprocessingModule
from elasticai.preprocessor.sequential import BuildPlatformTargets


class RescalerTargets(Enum):
    FxP = "fxp"
    IntSym = "int_sym"
    IntAsym = "int_asym"


@dataclass
class RescalerSettings:
    """Dataclass for defining the processing module to redefine data samples of the input data stream
    Attributes:
        method:     String with output rescaling method: int -> ["fxp", "int_sym", "int_asym"]
        total_bits: Integer with total bitwidth at output
        frac_bits:  Integer with fraction bitwidth at output
        is_signed:  Boolean for signed data type at output
    """

    method: RescalerTargets
    total_bits: int
    frac_bits: int
    is_signed: bool


RescalerSettingsDefault = RescalerSettings(
    method=RescalerTargets.FxP, total_bits=8, frac_bits=2, is_signed=False
)


class Rescaler(PreprocessingModule[DataTransient, DataTransient]):
    _settings: RescalerSettings
    _arith: FxpArithmetic
    _allowed_input_types = ["int"]
    _scaling_factor: float = 1.0
    _zero_point: float = 0.0

    def __init__(self, settings: RescalerSettings, update_params: bool = True) -> None:
        """Class for resampling transient input data to model desired data stream output (digital only)
        :param settings:        Settings for defining the properties of the data rescaling method
        :param update_params:   Boolean with updating the scaling and bias params (true) or not
        :return:                None
        """
        super().__init__()
        self._settings = settings
        if isinstance(settings.method, str):
            self._settings.method = RescalerTargets(settings.method.lower())

        self._update = update_params
        self._arith = FxpArithmetic(
            FxpParams(
                total_bits=self._settings.total_bits,
                frac_bits=self._settings.frac_bits if self._settings.method == RescalerTargets.FxP else 0,
                signed=self._settings.is_signed,
            )
        )

    def apply_params(self, scaling: float, bias: float) -> None:
        """Applying custom params for rescaling"""
        self._scaling_factor = scaling
        self._zero_point = bias
        self._update = False

    def __call__(self, x: DataTransient) -> DataTransient:
        return DataTransient(
            data=self.forward(x.data),
            sampling_rate=x.sampling_rate,
        )

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

    def forward(self, x: np.ndarray) -> np.ndarray:
        if "int" not in str(x.dtype):
            raise AttributeError("Input data must be an integer type (simulating an ADC output directly)")

        match self._settings.method:
            case RescalerTargets.FxP:
                return self._rescale_fxp(x)
            case RescalerTargets.IntSym:
                return self._rescale_int_sym(x)
            case RescalerTargets.IntAsym:
                return self._rescale_int_asym(x)
            case _:
                raise AttributeError

    def _get_data_statistics(self, x: np.ndarray) -> tuple[float, float, float]:
        return float(np.max(x)), float(np.min(x)), float(np.mean(x))

    def _get_int_range_and_dtype(self) -> tuple[int, int, np.dtype]:
        total_bits = self._settings.total_bits
        is_signed = self._settings.is_signed
        qmin, qmax = self._arith.minimum_as_integer, self._arith.maximum_as_integer

        for bits in (8, 16, 32, 64):
            if total_bits <= bits:
                return qmin, qmax, np.dtype(f"{'int' if is_signed else 'uint'}{bits}")
        raise AttributeError(f"Unknown datatype for total_bits = {total_bits}")

    def _rescale_fxp(self, x: np.ndarray) -> np.ndarray:
        qmin, qmax, dtype = self._get_int_range_and_dtype()
        xmax, xmin, xmean = self._get_data_statistics(x)
        scale = (xmax - xmin) / (qmax - qmin)
        offset = xmean / scale

        if self._update:
            self._scaling_factor = scale
            self._zero_point = offset

        x0 = (x - xmean) / scale + +(0 if self._settings.is_signed else qmax / 2)
        return self._quantize_digital(data=x0, is_int_input=True, is_int_output=True)

    def _rescale_int_sym(self, x: np.ndarray) -> np.ndarray:
        qmin, qmax, dtype = self._get_int_range_and_dtype()
        xmax, xmin, xmean = self._get_data_statistics(x)
        scale = (xmax - xmin) / (qmax - qmin)
        offset = xmean / scale

        if self._update:
            self._scaling_factor = scale
            self._zero_point = offset

        x0 = (x - xmean) / scale + (0 if self._settings.is_signed else qmax / 2)
        xq = np.clip(np.round(x0), a_min=qmin, a_max=qmax)
        return xq.astype(dtype)

    def _rescale_int_asym(self, x: np.ndarray) -> np.ndarray:
        qmin, qmax, dtype = self._get_int_range_and_dtype()
        xmax, xmin, xmean = self._get_data_statistics(x)
        scale = (xmax - xmin) / (qmax - qmin)
        offset = xmean / scale

        if self._update:
            self._scaling_factor = scale
            self._zero_point = offset

        xq = np.clip(
            np.round((x - xmean) / self._scaling_factor) + self._zero_point, a_min=qmin, a_max=qmax
        )
        return xq.astype(dtype)

    def create_design(
        self,
        id: str,
        target: str,
        path2save: Path,
        total_bitwidth: int,
        frac_bitwidth: int,
        is_signed: bool,
    ) -> None:
        if isinstance(target, str):
            target0 = BuildPlatformTargets(target.lower())
        else:
            target0 = target
        if target0 in [BuildPlatformTargets.Workstation, BuildPlatformTargets.MCU]:
            self._create_design_c(
                id=id, path2save=path2save, define_path="src", bitwidth_input=total_bitwidth
            )
        else:
            self._create_design_verilog(id=id, path2save=path2save, bitwidth_input=total_bitwidth)

    def _create_design_c(
        self, id: str, path2save: Path, bitwidth_input: int, define_path: str = "src"
    ) -> None:
        from elasticai.creator_plugins.adc.src import c_compile

        c_compile.build_rescaler(
            id=id,
            define_path=define_path,
            settings=self._settings,
            path2save=path2save,
            bitwidth_input=bitwidth_input,
            scaling_factor=self._scaling_factor,
            zero_point=self._zero_point,
        )

    def _create_design_verilog(self, id: str, path2save: Path, bitwidth_input: int) -> None:
        from elasticai.creator_plugins.adc.utils import load_and_plugin

        load_and_plugin(
            type="rescaler",
            id=id,
            params={"BITWIDTH_IN": bitwidth_input, "BITWIDTH_OUT": bitwidth_input, "NUM_SAMPLES": 1},
            path2save=path2save,
        )
