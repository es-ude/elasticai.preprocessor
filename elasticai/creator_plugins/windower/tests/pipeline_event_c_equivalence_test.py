from pathlib import Path
from shutil import which
from uuid import uuid4

import numpy as np
import pytest
from elasticai.equichecker import CompileLoader, compare_values

from elasticai.creator_plugins.windower.src.c_compile import build_windower_event_pipeline
from elasticai.preprocessor.eventdetection.preprocessors import EventPreprocessorsTargets
from elasticai.preprocessor.thresholding import ThresholdTargets
from elasticai.preprocessor.windower.window import (
    WindowerTargets,
    WindowSequencer,
    WindowSettings,
)

pytestmark = pytest.mark.skipif(which("cc") is None, reason="requires a C compiler")

# (settings, bitwidth, signed, c_type, np_dtype)
PIPELINE_CONFIGS = [
    pytest.param(
        WindowSettings(
            sampling_rate=100.0, window_sec=0.10, overlap_sec=0.0,
            method_window=WindowerTargets.Event,
            method_thr=ThresholdTargets.Constant,
            method_input=EventPreprocessorsTargets.Normal,
            pre_time=0.02, threshold=50.0,
        ),
        16, True, "signed short", np.int16,
        id="Normal_Constant_int16",
    ),
    pytest.param(
        WindowSettings(
            sampling_rate=100.0, window_sec=0.08, overlap_sec=0.0,
            method_window=WindowerTargets.Event,
            method_thr=ThresholdTargets.Constant,
            method_input=EventPreprocessorsTargets.Normal,
            pre_time=0.01, threshold=50.0,
        ),
        8, True, "signed char", np.int8,
        id="Normal_Constant_int8",
    ),
    pytest.param(
        WindowSettings(
            sampling_rate=100.0, window_sec=0.10, overlap_sec=0.0,
            method_window=WindowerTargets.Event,
            method_thr=ThresholdTargets.Constant,
            method_input=EventPreprocessorsTargets.Absolute,
            pre_time=0.02, threshold=50.0,
        ),
        16, True, "signed short", np.int16,
        id="Absolute_Constant_int16",
    ),
    pytest.param(
        WindowSettings(
            sampling_rate=100.0, window_sec=0.10, overlap_sec=0.0,
            method_window=WindowerTargets.Event,
            method_thr=ThresholdTargets.Constant,
            method_input=EventPreprocessorsTargets.NEO,
            pre_time=0.02, threshold=5000.0,
        ),
        16, True, "signed short", np.int16,
        id="NEO_Constant_int16",
    ),
]


def _make_signal(settings: WindowSettings) -> np.ndarray:
    """Signal with two clear spikes well past the ring-buffer warm-up period."""
    n = 200
    signal = np.zeros(n, dtype=np.int16)
    if settings.method_input == EventPreprocessorsTargets.Absolute:
        # Negative spikes: only detected after abs() is applied
        signal[50] = -100
        signal[130] = -80
    else:
        signal[50] = 100
        signal[130] = 80
    return signal


def test_build_pipeline_generates_c_files(tmp_path: Path) -> None:
    settings = WindowSettings(
        sampling_rate=100.0, window_sec=0.10, overlap_sec=0.0,
        method_window=WindowerTargets.Event,
        method_thr=ThresholdTargets.Constant,
        method_input=EventPreprocessorsTargets.Normal,
        pre_time=0.02, threshold=50.0,
    )
    build_windower_event_pipeline(
        settings=settings, bitwidth=16, signed=True,
        path2save=tmp_path, windower_id="0",
    )
    assert (tmp_path / "windower_event_pipeline_0.h").exists()
    assert (tmp_path / "windower_event_pipeline_0.c").exists()
    assert (tmp_path / "preprocessing_normal_template.h").exists()
    assert (tmp_path / "thresholding_constant_template.h").exists()


@pytest.mark.simulation
@pytest.mark.parametrize("settings,bitwidth,signed,c_type,np_dtype", PIPELINE_CONFIGS)
def test_pipeline_c_matches_python(
    tmp_path: Path,
    settings: WindowSettings,
    bitwidth: int,
    signed: bool,
    c_type: str,
    np_dtype: type,
) -> None:
    output_dir = tmp_path / "src"
    build_windower_event_pipeline(
        settings=settings,
        bitwidth=bitwidth,
        signed=signed,
        path2save=output_dir,
        windower_id="0",
        define_path=".",
    )

    wl = settings.window_length
    signal = _make_signal(settings).astype(np_dtype)

    # Prototype for CFFI (uses _Bool for the C99 bool return type)
    adapter = tmp_path / "adapter.h"
    adapter.write_text(f"_Bool calc_windower_event_pipeline_0({c_type} data, {c_type} *out);\n")
    loader = CompileLoader(
        headers=str(adapter),
        sources=[str(output_dir / "windower_event_pipeline_0.c")],
        build_dir=str(tmp_path / "cffi-build"),
        module_name=f"pipeline_event_equivalence_{uuid4().hex}",
    )
    loader.load()

    py_windows = WindowSequencer(settings).get_windows(signal)
    assert py_windows.shape[1] > 1, "no events detected in Python — check test signal or threshold"

    ffi = loader.ffi()
    calc_pipeline = loader.get("calc_windower_event_pipeline_0")
    out = ffi.new(f"{c_type}[{wl}]")
    c_windows: list[list[int]] = []
    for sample in signal.tolist():
        if calc_pipeline(sample, out):
            c_windows.append([int(out[j]) for j in range(wl)])

    assert len(c_windows) == len(py_windows), (
        f"Window count mismatch: python={len(py_windows)}, c={len(c_windows)}"
    )
    for k, (py_win, c_win) in enumerate(zip(py_windows.tolist(), c_windows)):
        for j, (pv, cv) in enumerate(zip(py_win, c_win)):
            passed, reason = compare_values(int(pv), cv)
            assert passed, f"window {k}, sample {j}: python={int(pv)}, c={cv}: {reason}"
