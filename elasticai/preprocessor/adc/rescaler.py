import math
from dataclasses import dataclass, replace
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
        method:      Output rescaling method: RescalerTargets or string -> ["fxp", "int_sym", "int_asym"]
        scale_safe:  Safety margin on the value range (>= 1.0). Values > 1.0 leave headroom, so unseen data
                     clips less often when fixed parameters are used (costs resolution).
        total_bits:  Total bitwidth at output
        frac_bits:   Fraction bitwidth at output (only relevant for FxP, defines the value of one LSB)
        is_signed:   Signed data type at output
        scale_pow2:  Scaling factor is rounded up to a power of two, so hardware only needs a shift
                     (rounding is then half-up: (x + (1 << (s-1))) >> s). Costs up to 50% of the output range.
    """

    method: RescalerTargets | str
    scale_safe: float
    total_bits: int
    frac_bits: int
    is_signed: bool
    scale_pow2: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.method, str):
            self.method = RescalerTargets(self.method.lower())

    @property
    def lsb(self) -> float:
        return 2**-self.frac_bits if self.method == RescalerTargets.FxP else 1


RescalerSettingsDefault = RescalerSettings(
    method=RescalerTargets.FxP, scale_safe=1.0, total_bits=8, frac_bits=2, is_signed=False
)


class Rescaler(PreprocessingModule[DataTransient, DataTransient]):
    """Maps integer input data (e.g. ADC output) onto the value range of the model quantization.

    All methods share one affine mapping with stored parameters:
        q = clip(round(x / scale) + zero_point, q_lo, q_hi)

    The methods differ only in the calibration (how scale and zero_point are derived from the data):
        fxp:      xmin -> qmin, xmax -> qmax, centered (uses the full range of the Q-format).
        int_asym: same calibration as fxp, always returns the integer code.
        int_sym:  zero_point fixed (0 for signed, mid-range for unsigned), scale from max(|xmin|, |xmax|).

    With scale_pow2 the scale is a power of two (scale = 2**shift), the mapping becomes a bit shift.
    """

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
        self._settings = replace(settings)
        self._update = update_params
        self._arith = FxpArithmetic(
            FxpParams(
                total_bits=self._settings.total_bits,
                frac_bits=self._settings.frac_bits if self._settings.method == RescalerTargets.FxP else 0,
                signed=self._settings.is_signed,
            )
        )

    def apply_params(self, scaling: float, bias: float) -> None:
        """Applying custom params for rescaling (disables further calibration)"""
        if self._settings.scale_pow2 and not self._is_pow2(scaling):
            raise ValueError(f"scaling must be a power of two if scale_pow2 is set, got {scaling}")
        self._scaling_factor = scaling
        self._zero_point = bias
        self._update = False

    @property
    def scaling_factor(self) -> float:
        return self._scaling_factor

    @property
    def zero_point(self) -> float:
        return self._zero_point

    @property
    def shift(self) -> int:
        """Exponent of the scaling factor (scale = 2**shift): > 0 shifts right, < 0 shifts left"""
        if not self._is_pow2(self._scaling_factor):
            raise ValueError("shift is only defined for a scaling factor that is a power of two")
        return int(round(math.log2(self._scaling_factor)))

    @staticmethod
    def _is_pow2(value: float) -> bool:
        return value > 0 and math.frexp(value)[0] == 0.5

    def _get_int_range_and_dtype(self) -> tuple[int, int, np.dtype]:
        total_bits = self._settings.total_bits
        qmin = int(self._arith.minimum_as_integer)
        qmax = int(self._arith.maximum_as_integer)
        for bits in (8, 16, 32, 64):
            if total_bits <= bits:
                prefix = "int" if self._settings.is_signed else "uint"
                return qmin, qmax, np.dtype(f"{prefix}{bits}")
        raise AttributeError(f"Unknown datatype for total_bits = {total_bits}")

    @staticmethod
    def _get_min_max(x: np.ndarray) -> tuple[float, float]:
        return float(np.min(x)), float(np.max(x))

    def _safe_scale(self, value_range: float, code_range: float) -> float:
        if value_range <= 1e-12:
            return 1.0
        scale = value_range / code_range * self._settings.scale_safe
        if self._settings.scale_pow2:
            scale = 2.0 ** math.ceil(math.log2(scale))
        return scale

    def _calibrate_affine(self, x: np.ndarray) -> tuple[float, float]:
        qmin, qmax, _ = self._get_int_range_and_dtype()
        xmin, xmax = self._get_min_max(x)
        scale = self._safe_scale(xmax - xmin, qmax - qmin)
        zero_point = round((qmin + qmax) / 2 - (xmin + xmax) / 2 / scale)
        return scale, float(zero_point)

    def _calibrate_sym(self, x: np.ndarray) -> tuple[float, float]:
        _, qmax, _ = self._get_int_range_and_dtype()
        xmin, xmax = self._get_min_max(x)
        max_abs = max(abs(xmin), abs(xmax))
        mean = np.mean(x)

        if self._settings.is_signed:
            zero_point = 0
            half_range = qmax
        else:
            zero_point = (qmax + 1) // 2
            half_range = qmax - zero_point
        return self._safe_scale(max_abs, half_range), float(zero_point)

    def _scale_and_round(self, x: np.ndarray) -> np.ndarray:
        scaled = x.astype(np.float64) / self._scaling_factor
        rounded = np.floor(scaled + 0.5) if self._settings.scale_pow2 else np.round(scaled)
        return rounded + self._zero_point

    def _forward_fxp(self, x: np.ndarray) -> np.ndarray:
        if self._update:
            self._scaling_factor, self._zero_point = self._calibrate_affine(x)
        qmin, qmax, _ = self._get_int_range_and_dtype()
        q = np.clip(self._scale_and_round(x), qmin, qmax)
        return (q * self._settings.lsb).astype(np.float32)

    def _forward_int_sym(self, x: np.ndarray) -> np.ndarray:
        if self._update:
            self._scaling_factor, self._zero_point = self._calibrate_sym(x)
        qmin, qmax, dtype = self._get_int_range_and_dtype()
        if self._settings.is_signed:
            qmin = -qmax
        return np.clip(self._scale_and_round(x), qmin, qmax).astype(dtype)

    def _forward_int_asym(self, x: np.ndarray) -> np.ndarray:
        if self._update:
            self._scaling_factor, self._zero_point = self._calibrate_affine(x)
        qmin, qmax, dtype = self._get_int_range_and_dtype()
        return np.clip(self._scale_and_round(x), qmin, qmax).astype(dtype)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.dtype.kind not in "iu":
            raise AttributeError("Input data must be an integer type (simulating an ADC output directly)")

        match self._settings.method:
            case RescalerTargets.FxP:
                return self._forward_fxp(x)
            case RescalerTargets.IntSym:
                return self._forward_int_sym(x)
            case RescalerTargets.IntAsym:
                return self._forward_int_asym(x)
            case _:
                raise AttributeError(f"Unknown rescaling method: {self._settings.method}")

    def __call__(self, x: DataTransient) -> DataTransient:
        return DataTransient(
            data=self.forward(x.data),
            sampling_rate=x.sampling_rate,
        )

    def create_design(
        self,
        id: str,
        target: str,
        path2save: Path,
        total_bitwidth: int,
        frac_bitwidth: int,
        is_signed: bool,
    ) -> None:
        """total_bitwidth is the bitwidth of the INPUT data, the output bitwidth comes from the settings"""
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

        vtype = "rescaler"
        params = {
            "BITWIDTH_IN": bitwidth_input,
            "IS_SIGNED_IN": int(self._settings.is_signed),
            "BITWIDTH_OUT": self._settings.total_bits,
            "IS_SIGNED_OUT": int(self._settings.is_signed),
            "NUM_SAMPLES": 1,
            "ZERO_POINT": int(self._zero_point),
        }

        if self._settings.scale_pow2:
            vtype += "_pow2"
            params.update(
                {
                    "SHIFT": 1,
                }
            )
        else:
            vtype += "_mult"
            params.update(
                {
                    "SCALE_MULT": 1,
                    "SCALE_BIAS": 0,
                }
            )
        load_and_plugin(type=vtype, id=id, params=params, path2save=path2save)
