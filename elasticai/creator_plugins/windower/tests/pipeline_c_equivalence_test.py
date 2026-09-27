"""Tests for build_pipeline: generates C files from Python settings and verifies
that the generated pipeline compiles and produces the expected number of windows.
"""
import subprocess
from pathlib import Path
from shutil import which

import pytest

from elasticai.creator_plugins.windower.src.c_compile import build_pipeline
from elasticai.preprocessor.eventdetection import TargetsEventPreprocessors
from elasticai.preprocessor.thresholding import TargetsThreshold
from elasticai.preprocessor.windower.window import SettingsWindow, TargetsWindower

pytestmark = pytest.mark.skipif(which("cc") is None, reason="requires a C compiler")

_SETTINGS = SettingsWindow(
    method_window=TargetsWindower.Sliding,
    method_thr=TargetsThreshold.Constant,
    method_input=TargetsEventPreprocessors.Normal,
    sampling_rate=1000.0,
    window_sec=0.008,   # 8 samples at 1 kHz
    overlap_sec=0.0,
    pre_time=0.001,
    threshold=10.0,
)


def test_build_pipeline_rejects_non_power_of_two_dsr(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="dsr must be 2"):
        build_pipeline(
            iir_coefficients_a=[1.0, 0.0],
            iir_coefficients_b=[0.5, 0.5],
            downsampling_ratio=3,
            settings=_SETTINGS,
            bitwidth=32,
            signed=True,
            path2save=tmp_path,
        )


def test_build_pipeline_rejects_mismatched_coefficients(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="same length"):
        build_pipeline(
            iir_coefficients_a=[1.0, 0.0, 0.0],
            iir_coefficients_b=[0.5, 0.5],
            downsampling_ratio=4,
            settings=_SETTINGS,
            bitwidth=32,
            signed=True,
            path2save=tmp_path,
        )


def test_build_pipeline_generates_c_files(tmp_path: Path) -> None:
    build_pipeline(
        iir_coefficients_a=[1.0, 0.0],
        iir_coefficients_b=[0.5, 0.5],
        downsampling_ratio=4,
        settings=_SETTINGS,
        bitwidth=32,
        signed=True,
        path2save=tmp_path,
        pipeline_id="0",
    )
    assert (tmp_path / "pipeline_0.h").exists()
    assert (tmp_path / "pipeline_0.c").exists()
    assert (tmp_path / "pipeline_template.h").exists()
    assert (tmp_path / "filter_iir_template.h").exists()
    assert (tmp_path / "downsampling_poly_template.h").exists()
    assert (tmp_path / "windower_template.h").exists()


def test_build_pipeline_compiles_and_runs(tmp_path: Path) -> None:
    """Full integration: generate → compile → run → check window count."""
    dsr = 4
    window_length = _SETTINGS.window_length  # 8
    num_input_samples = 4 * window_length * dsr  # 4 complete windows

    build_pipeline(
        iir_coefficients_a=[1.0, 0.0],
        iir_coefficients_b=[0.5, 0.5],
        downsampling_ratio=dsr,
        settings=_SETTINGS,
        bitwidth=32,
        signed=True,
        path2save=tmp_path,
        pipeline_id="0",
        define_path=".",
    )

    main_c = tmp_path / "main.c"
    main_c.write_text(
        f"""\
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include "pipeline_0.h"
#include "pipeline_0.c"

int main(void) {{
    int32_t window[{window_length}];
    int windows_received = 0;
    for (int i = 0; i < {num_input_samples}; i++) {{
        int32_t sample = (int32_t)(i % 64 - 32);
        if (calc_pipeline_0(sample, window)) {{
            windows_received++;
        }}
    }}
    assert(windows_received == 4);
    printf("OK: %d windows\\n", windows_received);
    return 0;
}}
"""
    )

    cc = which("gcc") or which("cc")
    compile_result = subprocess.run(
        [cc, "-Wall", str(main_c), "-I", str(tmp_path), "-o", str(tmp_path / "pipeline_run")],
        capture_output=True,
        text=True,
    )
    assert compile_result.returncode == 0, (
        f"Compilation failed:\n{compile_result.stderr}"
    )

    run_result = subprocess.run(
        [str(tmp_path / "pipeline_run")], capture_output=True, text=True
    )
    assert run_result.returncode == 0, (
        f"Runtime assertion failed:\n{run_result.stdout}\n{run_result.stderr}"
    )
    assert "OK: 4 windows" in run_result.stdout
