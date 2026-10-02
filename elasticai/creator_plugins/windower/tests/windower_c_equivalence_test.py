from pathlib import Path
from shutil import which
from uuid import uuid4

import numpy as np
import pytest
from elasticai.equichecker import CompileLoader, compare_values

from elasticai.creator_plugins.windower.src.c_compile import build_windower_event, build_windower_sliding
from elasticai.preprocessor import get_path_to_project
from elasticai.preprocessor.eventdetection import TargetsEventPreprocessors
from elasticai.preprocessor.thresholding import TargetsThreshold
from elasticai.preprocessor.translation.cocotb_tmp import temporary_directory
from elasticai.preprocessor.windower.window import SettingsWindow, TargetsWindower, WindowSequencer

pytestmark = pytest.mark.skipif(which("cc") is None, reason="requires a C compiler")


def _ws(
    method: TargetsWindower, 
    sampling_rate: float,
    window_sec: float,
    overlap_sec: float
) -> SettingsWindow:
    return SettingsWindow(
        method_window=method,
        method_thr=TargetsThreshold.Constant,
        method_input=TargetsEventPreprocessors.Normal,
        sampling_rate=100.0,
        window_sec=window_sec,
        overlap_sec=overlap_sec,
        pre_time=0.001,
        threshold=10.0,
    )


WINDOWER_CONFIGS = [
    pytest.param(
        _ws(TargetsWindower.Sliding, 100.0, 0.10, 0.05),
        16,
        True,
        "signed short",
        np.int16,
        id="sliding_int16_window10_overlap5",
    ),
    pytest.param(
        _ws(TargetsWindower.Sliding, 100.0, 0.08, 0.04),
        8,
        True,
        "signed char",
        np.int8,
        id="sliding_int8_window8_overlap4",
    ),
    pytest.param(
        _ws(TargetsWindower.Sliding, 100.0, 0.06, 0.0),
        8,
        False,
        "unsigned char",
        np.uint8,
        marks=pytest.mark.filterwarnings(
            "ignore::RuntimeWarning"
            # window_size=6 triggers gaussian(6, int(0.16*6)=0) in
            # transformation_window_method, which is called eagerly for all
            # window types even though method="" selects np.ones().
        ),
        id="sliding_uint8_window6_nooverlap",
    ),
]

MY_CONFIGS = [
    pytest.param( 8,  True,   "int8_t", np.int8,   id="int8"),
    pytest.param( 8, False,  "uint8_t", np.uint8,  id="uint8"),
    pytest.param(16,  True,  "int16_t", np.int16,  id="int16"), 
    pytest.param(16, False, "uint16_t", np.uint16, id="uint16"), 
    pytest.param(32,  True,  "int32_t", np.int32,  id="int32"), 
    pytest.param(32, False, "uint32_t", np.uint32, id="uint32"),
]


def _event_windower_py(
    signal: np.ndarray,
    window_length: int,
    pre_samples: int,
    threshold: int,
) -> list[list[int]]:
    """Python mirror of the C windower_event logic."""
    buf: list[int] = [0] * window_length
    is_event = False
    count = 0
    windows: list[list[int]] = []
    for sample in signal.tolist():
        s = int(sample)
        for i in range(1, window_length):
            buf[i - 1] = buf[i]
        buf[window_length - 1] = s
        if count < window_length:
            count += 1
            if count < window_length:
                continue
        if buf[pre_samples] >= threshold:
            if not is_event:
                is_event = True
                windows.append(list(buf))
        else:
            if is_event:
                is_event = False
    return windows


def test_build_windower_sequence_generates_c_files(tmp_path: Path) -> None:
    settings = _ws(TargetsWindower.Sequence, 100.0, 0.10, 0.0)
    build_windower_sliding(
        settings=settings, 
        bitwidth=16, 
        signed=True, 
        path2save=tmp_path, 
        windower_id="0"
    )
    assert (tmp_path / "windower_sliding_template.h").exists()
    assert (tmp_path / "windower_sliding_0.h").exists()
    assert (tmp_path / "windower_sliding_0.c").exists()

def test_build_windower_sliding_generates_c_files(tmp_path: Path) -> None:
    settings = _ws(TargetsWindower.Sliding, 100.0, 0.10, 0.05)
    build_windower_sliding(
        settings=settings,
        bitwidth=16,
        signed=True,
        path2save=tmp_path,
        windower_id="0"
    )
    assert (tmp_path / "windower_sliding_template.h").exists()
    assert (tmp_path / "windower_sliding_0.h").exists()
    assert (tmp_path / "windower_sliding_0.c").exists()

def test_build_windower_event_generates_c_files(tmp_path: Path) -> None:
    settings = _ws(TargetsWindower.Event, 100.0, 0.10, 0.05)
    build_windower_event(
        settings=settings,
        threshold=10,
        pre_padding=2,
        bitwidth=16,
        signed=True,
        path2save=tmp_path,
        windower_id="0"
    )
    assert (tmp_path / "windower_event_template.h").exists()
    assert (tmp_path / "windower_event_0.h").exists()
    assert (tmp_path / "windower_event_0.c").exists()

@pytest.mark.parametrize("target", ["mcu", "pc"])
@pytest.mark.parametrize("bitwidth,signed,c_type,np_dtype", MY_CONFIGS)
def test_generated_windower_sequence_c_matches_python(
    target: str,
    bitwidth: int,
    signed: bool,
    np_dtype: type,
    c_type: str,
) -> None:
    settings = SettingsWindow(
        method_window=TargetsWindower.Sequence,
        method_thr=TargetsThreshold.Constant,
        method_input=TargetsEventPreprocessors.Normal,
        sampling_rate=100.0,
        window_sec=0.10,
        overlap_sec=0.0,
        pre_time=0.0,
        threshold=10.0,
    )
    sequencer = WindowSequencer(settings)
    backup = get_path_to_project("build_test") / "build_sequence"
    with temporary_directory(backup) as tmpdir:
        output_dir = tmpdir / "src"
        sequencer.create_design(
            target=target,
            bitwidth=bitwidth,
            signed=signed,
            id="0",
            path2save=output_dir,
        )

        window_length = settings.window_length
        num_shift = window_length

        adapter = tmpdir / "adapter.h"
        adapter.write_text(f"_Bool calc_windower_sliding_0({c_type} data, {c_type} *out);\n")
        loader = CompileLoader(
            headers=str(adapter),
            sources=[str(output_dir / "windower_sliding_0.c")],
            build_dir=str(tmpdir / "cffi-build"),
            module_name=f"windower_equivalence_{uuid4().hex}",
        )
        loader.load()

        num_samples = window_length + 3 * num_shift
        samples = np.arange(num_samples, dtype=np_dtype)
        py_windows = WindowSequencer(settings).sequence(samples)

        ffi = loader.ffi()
        calc_windower_sequence = loader.get("calc_windower_sliding_0")
        out = ffi.new(f"{c_type}[{window_length}]")
        c_windows = []
        for data in samples.tolist():
            if calc_windower_sequence(data, out):
                c_windows.append([int(out[j]) for j in range(window_length)])

        assert len(c_windows) == len(py_windows), (
            f"Output count mismatch: python={len(py_windows)}, c={len(c_windows)}"
        )
        for k, (py_win, c_win) in enumerate(zip(py_windows.tolist(), c_windows)):
            for j, (pv, cv) in enumerate(zip(py_win, c_win)):
                passed, reason = compare_values(int(pv), cv)
                assert passed, f"window {k}, sample {j}: python={int(pv)}, c={cv}: {reason}"

@pytest.mark.parametrize("target", ["mcu", "pc"])
@pytest.mark.parametrize("bitwidth,signed,c_type,np_dtype", MY_CONFIGS)
def test_generated_windower_sliding_c_matches_python(
    target: str,
    bitwidth: int,
    signed: bool,
    np_dtype: type,
    c_type: str,
) -> None:
    settings = SettingsWindow(
        method_window=TargetsWindower.Sliding,
        method_thr=TargetsThreshold.Constant,
        method_input=TargetsEventPreprocessors.Normal,
        sampling_rate=100.0,
        window_sec=0.10,
        overlap_sec=0.05,
        pre_time=0.0,
        threshold=10.0,
    )
    sequencer = WindowSequencer(settings)
    backup = get_path_to_project("build_test") / "build_sliding"
    with temporary_directory(backup) as tmpdir:
        output_dir = tmpdir / "src"
        sequencer.create_design(
            target=target,
            bitwidth=bitwidth,
            signed=signed,
            id="0",
            path2save=output_dir,
        )

        window_length = settings.window_length
        num_shift = window_length - settings.overlap_length

        adapter = tmpdir / "adapter.h"
        adapter.write_text(f"_Bool calc_windower_sliding_0({c_type} data, {c_type} *out);\n")
        loader = CompileLoader(
            headers=str(adapter),
            sources=[str(output_dir / "windower_sliding_0.c")],
            build_dir=str(tmpdir / "cffi-build"),
            module_name=f"windower_equivalence_{uuid4().hex}",
        )
        loader.load()

        num_samples = window_length + 3 * num_shift
        samples = np.arange(num_samples, dtype=np_dtype)
        n_skip = -(-settings.overlap_length // num_shift)  # ceil: skip pre-padded windows
        py_windows = WindowSequencer(settings).slide(samples)[n_skip:]

        ffi = loader.ffi()
        calc_windower_sequence = loader.get("calc_windower_sliding_0")
        out = ffi.new(f"{c_type}[{window_length}]")
        c_windows = []
        for data in samples.tolist():
            if calc_windower_sequence(data, out):
                c_windows.append([int(out[j]) for j in range(window_length)])

        assert len(c_windows) == len(py_windows), (
            f"Output count mismatch: python={len(py_windows)}, c={len(c_windows)}"
        )
        for k, (py_win, c_win) in enumerate(zip(py_windows.tolist(), c_windows)):
            for j, (pv, cv) in enumerate(zip(py_win, c_win)):
                passed, reason = compare_values(int(pv), cv)
                assert passed, f"window {k}, sample {j}: python={int(pv)}, c={cv}: {reason}"

@pytest.mark.parametrize("target", ["mcu", "pc"])
@pytest.mark.parametrize("bitwidth,signed,c_type,np_dtype", MY_CONFIGS)
def test_generated_windower_event_c_matches_python(
    target: str,
    bitwidth: int,
    signed: bool,
    np_dtype: type,
    c_type: str,
) -> None:
    thr_int = 10
    pre_int = 0
    settings = SettingsWindow(
        method_window=TargetsWindower.Event,
        method_thr=TargetsThreshold.Constant,
        method_input=TargetsEventPreprocessors.Normal,
        sampling_rate=100.0,
        window_sec=0.10,
        overlap_sec=0.0,
        pre_time=0.0,
        threshold=float(thr_int),
    )
    sequencer = WindowSequencer(settings)
    backup = get_path_to_project("build_test") / "build_event"
    with temporary_directory(backup) as tmpdir:
        output_dir = tmpdir / "src"
        sequencer.create_design(
            target=target,
            bitwidth=bitwidth,
            signed=signed,
            id="0",
            path2save=output_dir,
            threshold=thr_int,
            pre_samples=pre_int,
        )

        window_length = settings.window_length

        adapter = tmpdir / "adapter.h"
        adapter.write_text(f"_Bool calc_windower_event_0({c_type} data, {c_type} *out);\n")
        loader = CompileLoader(
            headers=str(adapter),
            sources=[str(output_dir / "windower_event_0.c")],
            build_dir=str(tmpdir / "cffi-build"),
            module_name=f"windower_equivalence_{uuid4().hex}",
        )
        loader.load()

        num_samples = window_length + 30
        samples = np.ones(num_samples, dtype=np_dtype)
        samples[3] = 12
        samples[4] = 13
        samples[5] = 10
        samples[20] = 11
        samples[22] = 11
        py_windows = _event_windower_py(samples, window_length, pre_int, thr_int)

        ffi = loader.ffi()
        calc_windower_event = loader.get("calc_windower_event_0")
        out = ffi.new(f"{c_type}[{window_length}]")
        c_windows = []
        for data in samples.tolist():
            if calc_windower_event(data, out):
                c_windows.append([int(out[j]) for j in range(window_length)])

        assert len(c_windows) == len(py_windows), (
            f"Output count mismatch: python={len(py_windows)}, c={len(c_windows)}"
        )
        for k, (py_win, c_win) in enumerate(zip(py_windows, c_windows)):
            for j, (pv, cv) in enumerate(zip(py_win, c_win)):
                passed, reason = compare_values(int(pv), cv)
                assert passed, f"window {k}, sample {j}: python={int(pv)}, c={cv}: {reason}"
