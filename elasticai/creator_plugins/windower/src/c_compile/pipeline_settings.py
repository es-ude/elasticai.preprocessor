from dataclasses import dataclass, field
from enum import Enum

from elasticai.preprocessor.windower.window import SettingsWindow


class TargetsFilterC(Enum):
    """Filter stage selection for the C streaming pipeline.

    IIR         — Direct-Form-II IIR filter.  Supply iir_a / iir_b.
    FIR         — General FIR filter.         Supply fir_order + fir_coefficients.
    MovingAverage — Unweighted moving average. Supply mavg_order.
    FirDelay    — Pure sample delay (allpass). Supply delay_order.
    Bypass      — No filtering; input passes through unchanged.
    """
    IIR           = "iir"
    FIR           = "fir"
    MovingAverage = "mavg"
    FirDelay      = "delay"
    Bypass        = "bypass"


class TargetsDownsamplingC(Enum):
    """Decimation stage selection for the C streaming pipeline.

    PolyOne — Non-recursive polyphase, 1st order.  ratio must be 2^n.
    PolyTwo — Non-recursive polyphase, 2nd order.  ratio must be 2^n.
    CIC     — Cascaded integrator-comb.             ratio >= 1; set cic_stages.
    Simple  — Averaging decimator (box filter).     ratio must be 2^n.
    Bypass  — No decimation; every sample is passed through.
    """
    PolyOne = "poly_one"
    PolyTwo = "poly_two"
    CIC     = "cic"
    Simple  = "simple"
    Bypass  = "bypass"


@dataclass
class SettingsPipelineFilter:
    """Filter-stage configuration for SettingsPipeline.

    Set `method` to select the filter type, then fill in the fields that
    apply to that type.  Fields for other types are ignored.

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

    Set `method` to select the decimator, then fill in the fields that apply.

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

    The pipeline processes one sample at a time and outputs a window array
    once enough samples have been collected:

        Filter → Decimation → Windower → bool (window ready)

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

    Example (2nd-order Butterworth bandpass, polyphase decimation, sliding window):

        from elasticai.preprocessor.filter import Filtering, SettingsFilter
        coeffs = Filtering(SettingsFilter(
            gain=1.0, fs=2000.0, n_order=2,
            f_filt=[300.0, 600.0], type="iir", f_type="butter", b_type="bandpass",
        )).get_coeffs()

        pipeline = SettingsPipeline(
            filter=SettingsPipelineFilter(
                method=TargetsFilterC.IIR,
                iir_a=coeffs.a,
                iir_b=coeffs.b,
            ),
            downsampling=SettingsPipelineDownsampling(
                method=TargetsDownsamplingC.PolyOne,
                ratio=4,
            ),
            window=SettingsWindow(
                method_window=TargetsWindower.Sliding,
                method_thr=TargetsThreshold.Constant,
                method_input=TargetsEventPreprocessors.Normal,
                sampling_rate=2000.0, window_sec=0.05, overlap_sec=0.0,
                pre_time=0.005, threshold=10.0,
            ),
            bitwidth=32,
            signed=True,
        )
    """
    filter:       SettingsPipelineFilter
    downsampling: SettingsPipelineDownsampling
    window:       SettingsWindow
    bitwidth:     int
    signed:       bool
