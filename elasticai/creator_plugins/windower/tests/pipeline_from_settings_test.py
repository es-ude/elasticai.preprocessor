"""Tests for build_pipeline: all supported filter and
decimation combinations compile and produce the expected window count.
"""

import subprocess
from pathlib import Path
from shutil import which

import pytest

from elasticai.creator_plugins.windower.src.c_compile import (
    SettingsPipeline,
    SettingsPipelineDownsampling,
    SettingsPipelineFilter,
    TargetsDownsamplingC,
    TargetsFilterC,
    build_pipeline,
)
from elasticai.preprocessor.eventdetection import TargetsEventPreprocessors
from elasticai.preprocessor.thresholding import TargetsThreshold
from elasticai.preprocessor.windower.window import SettingsWindow, TargetsWindower

pytestmark = pytest.mark.skipif(which("cc") is None, reason="requires a C compiler")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _window_settings(sampling_rate: float = 1000.0, window_sec: float = 0.008) -> SettingsWindow:
    return SettingsWindow(
        method_window=TargetsWindower.Sliding,
        method_thr=TargetsThreshold.Constant,
        method_input=TargetsEventPreprocessors.Normal,
        sampling_rate=sampling_rate,
        window_sec=window_sec,  # 8 samples at 1 kHz
        overlap_sec=0.0,
        pre_time=0.001,
        threshold=10.0,
    )


def _compile_and_run(
    tmp_path: Path, settings: SettingsPipeline, num_input_samples: int
) -> subprocess.CompletedProcess:
    """Build pipeline, compile a test driver, run it, return the result."""
    build_pipeline(
        settings=settings,
        path2save=tmp_path,
        pipeline_id="0",
        define_path=".",
    )

    wl = settings.window.window_length
    main_c = tmp_path / "main.c"
    main_c.write_text(
        f"""\
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include "pipeline_0.h"
#include "pipeline_0.c"

int main(void) {{
    int32_t window[{wl}];
    int windows_received = 0;
    for (int i = 0; i < {num_input_samples}; i++) {{
        int32_t sample = (int32_t)(i % 64 - 32);
        if (calc_pipeline_0(sample, window)) {{
            windows_received++;
        }}
    }}
    printf("windows=%d\\n", windows_received);
    return 0;
}}
"""
    )
    cc = which("gcc") or which("cc")
    compile_result = subprocess.run(
        [cc, "-Wall", str(main_c), "-I", str(tmp_path), "-o", str(tmp_path / "run")],
        capture_output=True,
        text=True,
    )
    assert compile_result.returncode == 0, f"Compilation failed:\n{compile_result.stderr}"
    return subprocess.run([str(tmp_path / "run")], capture_output=True, text=True)


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


def test_validate_iir_empty_coefficients(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="iir_a and iir_b must be non-empty"):
        build_pipeline(
            SettingsPipeline(
                filter=SettingsPipelineFilter(method=TargetsFilterC.IIR),
                downsampling=SettingsPipelineDownsampling(method=TargetsDownsamplingC.Bypass),
                window=_window_settings(),
                bitwidth=32,
                signed=True,
            ),
            path2save=tmp_path,
        )


def test_validate_poly_non_power_of_two(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="dsr must be 2"):
        build_pipeline(
            SettingsPipeline(
                filter=SettingsPipelineFilter(method=TargetsFilterC.Bypass),
                downsampling=SettingsPipelineDownsampling(method=TargetsDownsamplingC.PolyOne, ratio=3),
                window=_window_settings(),
                bitwidth=32,
                signed=True,
            ),
            path2save=tmp_path,
        )


# ---------------------------------------------------------------------------
# Compile + run tests for all filter × decimation combinations
# ---------------------------------------------------------------------------

# Each entry: (filter_settings, decimation_settings, expected_windows, num_samples_factor)
# num_samples = expected_windows * window_length * (ratio if not bypass else 1)

_PIPELINE_CONFIGS = [
    pytest.param(
        SettingsPipelineFilter(method=TargetsFilterC.IIR, iir_a=[1.0, 0.0], iir_b=[0.5, 0.5]),
        SettingsPipelineDownsampling(method=TargetsDownsamplingC.PolyOne, ratio=4),
        4,
        4,
        id="iir_poly1",
    ),
    pytest.param(
        SettingsPipelineFilter(method=TargetsFilterC.IIR, iir_a=[1.0, 0.0], iir_b=[0.5, 0.5]),
        SettingsPipelineDownsampling(method=TargetsDownsamplingC.PolyTwo, ratio=4),
        4,
        4,
        id="iir_poly2",
    ),
    pytest.param(
        SettingsPipelineFilter(
            method=TargetsFilterC.FIR, fir_order=3, fir_coefficients=[0.25, 0.5, 0.25]
        ),
        SettingsPipelineDownsampling(method=TargetsDownsamplingC.PolyOne, ratio=4),
        4,
        4,
        id="fir_poly1",
    ),
    pytest.param(
        SettingsPipelineFilter(method=TargetsFilterC.MovingAverage, mavg_order=4),
        SettingsPipelineDownsampling(method=TargetsDownsamplingC.Simple, ratio=4),
        4,
        4,
        id="mavg_simple",
    ),
    pytest.param(
        SettingsPipelineFilter(method=TargetsFilterC.FirDelay, delay_order=4),
        SettingsPipelineDownsampling(method=TargetsDownsamplingC.CIC, ratio=4, cic_stages=2),
        4,
        4,
        id="delay_cic",
    ),
    pytest.param(
        SettingsPipelineFilter(method=TargetsFilterC.Bypass),
        SettingsPipelineDownsampling(method=TargetsDownsamplingC.PolyOne, ratio=4),
        4,
        4,
        id="bypass_poly1",
    ),
    pytest.param(
        SettingsPipelineFilter(method=TargetsFilterC.IIR, iir_a=[1.0, 0.0], iir_b=[0.5, 0.5]),
        SettingsPipelineDownsampling(method=TargetsDownsamplingC.Bypass),
        4,
        1,
        id="iir_bypass",
    ),
    pytest.param(
        SettingsPipelineFilter(method=TargetsFilterC.Bypass),
        SettingsPipelineDownsampling(method=TargetsDownsamplingC.Bypass),
        4,
        1,
        id="bypass_bypass",
    ),
]


@pytest.mark.parametrize("filt,deci,expected_windows,dsr", _PIPELINE_CONFIGS)
def test_pipeline_compiles_and_produces_windows(
    tmp_path: Path,
    filt: SettingsPipelineFilter,
    deci: SettingsPipelineDownsampling,
    expected_windows: int,
    dsr: int,
) -> None:
    ws = _window_settings()
    wl = ws.window_length  # 8
    num_input = expected_windows * wl * dsr

    settings = SettingsPipeline(
        filter=filt,
        downsampling=deci,
        window=ws,
        bitwidth=32,
        signed=True,
    )
    result = _compile_and_run(tmp_path, settings, num_input)
    assert result.returncode == 0, result.stderr
    assert f"windows={expected_windows}" in result.stdout
