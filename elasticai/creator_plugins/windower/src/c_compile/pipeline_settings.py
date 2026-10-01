from dataclasses import dataclass, field
from enum import Enum

from elasticai.preprocessor.windower.window import SettingsWindow


class TargetsFilterC(Enum):
    IIR = "iir"
    FIR = "fir"
    MovingAverage = "mavg"
    FirDelay = "delay"
    Bypass = "bypass"


class TargetsDownsamplingC(Enum):
    PolyOne = "poly_one"
    PolyTwo = "poly_two"
    CIC = "cic"
    Simple = "simple"
    Bypass = "bypass"


@dataclass
class SettingsPipelineFilter:
    """Filter-stage configuration for SettingsPipeline.
    param method:   select filter type
                    IIR:
                        iir_a   — denominator coefficients [a0, a1, ..., aN]  (a0 usually 1.0)
                        iir_b   — numerator  coefficients [b0, b1, ..., bN]  (same length as iir_a)
                    FIR:
                        fir_order        — size of the tap-delay buffer (>= len(fir_coefficients))
                        fir_coefficients — filter weights h[0], h[1], ..., h[M]
                    MovingAverage:
                        mavg_order — number of taps; coefficient 1/mavg_order is applied automatically
                    FirDelay:
                        delay_order — delay length in samples
                    Bypass:
                        no additional fields required
    """

    method: TargetsFilterC
    # IIR
    iir_a: list[float] = field(default_factory=list)
    iir_b: list[float] = field(default_factory=list)
    # FIR
    fir_order: int = 0
    fir_coefficients: list[float] = field(default_factory=list)
    # MovingAverage
    mavg_order: int = 0
    # FirDelay
    delay_order: int = 0


@dataclass
class SettingsPipelineDownsampling:
    """Decimation-stage configuration for SettingsPipeline.
    param method:   select downsampling type
                    PolyOne / PolyTwo / Simple:
                        ratio — decimation factor (must be a power of two)
                    CIC:
                        ratio      — decimation factor
                        cic_stages — number of integrator–comb stages (default 1)
                    Bypass:
                        no additional fields required (ratio is ignored)
    """

    method: TargetsDownsamplingC
    ratio: int = 1
    cic_stages: int = 1


@dataclass
class SettingsPipeline:
    """Single container for all parameters of the streaming C pipeline.
    All stages are optional via `TargetsFilterC.Bypass` or
    `TargetsDownsamplingC.Bypass`.
    Fields:
        filter        — filter-stage configuration (SettingsPipelineFilter)
        downsampling  — decimation-stage configuration (SettingsPipelineDownsampling)
        window        — windower configuration (SettingsWindow);
                        method_window, method_thr, method_input, pre_time and
                        threshold are used Python-side only — the C pipeline uses
                        only window_length and overlap_length → num_shift
        bitwidth      — bit width of each sample for the C data type (2..32)
        signed        — whether the C data type is signed
    """

    filter: SettingsPipelineFilter
    downsampling: SettingsPipelineDownsampling
    window: SettingsWindow
    bitwidth: int
    signed: bool
