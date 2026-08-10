from pathlib import Path
from shutil import which
from uuid import uuid4

import numpy as np
import pytest
from elasticai.equichecker import CompileLoader, compare_values

from elasticai.creator_plugins.windower.src.c_compile import build_windower
from elasticai.preprocessor.windower.window import SettingsWindow, WindowSequencer

pytestmark = pytest.mark.skipif(which("cc") is None, reason="requires a C compiler")

WINDOWER_CONFIGS = [
    pytest.param(
        SettingsWindow(sampling_rate=100.0, window_sec=0.10, overlap_sec=0.05),
        16, True, "signed short", np.int16,
        id="int16_window10_overlap5",
    ),
    pytest.param(
        SettingsWindow(sampling_rate=100.0, window_sec=0.08, overlap_sec=0.04),
        8, True, "signed char", np.int8,
        id="int8_window8_overlap4",
    ),
    pytest.param(
        SettingsWindow(sampling_rate=100.0, window_sec=0.06, overlap_sec=0.0),
        8, False, "unsigned char", np.uint8,
        marks=pytest.mark.filterwarnings(
            "ignore::RuntimeWarning"
            # window_size=6 triggers gaussian(6, int(0.16*6)=0) in
            # transformation_window_method, which is called eagerly for all
            # window types even though method="" selects np.ones().
        ),
        id="uint8_window6_nooverlap",
    ),
]


def test_build_windower_generates_c_files(tmp_path: Path) -> None:
    settings = SettingsWindow(sampling_rate=100.0, window_sec=0.10, overlap_sec=0.05)
    build_windower(settings=settings, bitwidth=16, signed=True, path2save=tmp_path, windower_id="0")
    assert (tmp_path / "windower_template.h").exists()
    assert (tmp_path / "windower_0.h").exists()
    assert (tmp_path / "windower_0.c").exists()


@pytest.mark.parametrize("settings,bitwidth,signed,c_type,np_dtype", WINDOWER_CONFIGS)
def test_windower_c_matches_python(
    tmp_path: Path,
    settings: SettingsWindow,
    bitwidth: int,
    signed: bool,
    c_type: str,
    np_dtype: type,
) -> None:
    output_dir = tmp_path / "src"
    build_windower(
        settings=settings,
        bitwidth=bitwidth,
        signed=signed,
        path2save=output_dir,
        windower_id="0",
        define_path=".",
    )

    window_length = settings.window_length
    num_shift = window_length - settings.overlap_length

    adapter = tmp_path / "adapter.h"
    adapter.write_text(f"_Bool calc_windower_0({c_type} data, {c_type} *out);\n")
    loader = CompileLoader(
        headers=str(adapter),
        sources=[str(output_dir / "windower_0.c")],
        build_dir=str(tmp_path / "cffi-build"),
        module_name=f"windower_equivalence_{uuid4().hex}",
    )
    loader.load()

    num_samples = window_length + 3 * num_shift
    samples = np.arange(num_samples, dtype=np_dtype)

    # Python reference: WindowSequencer.slide() from window.py.
    # slide() pre-pads with overlap_length zeros before the signal, which produces
    # ceil(overlap_length / num_shift) extra windows at the start containing those
    # zeros. The C function outputs its first window only after window_length real
    # samples have arrived (no zeros). Skip the pre-padded Python windows to align.
    n_skip = -(-settings.overlap_length // num_shift)  # ceil division
    py_windows = WindowSequencer(settings).slide(samples)[n_skip:]

    # C: feed signal sample by sample, collect windows on true return
    ffi = loader.ffi()
    calc_windower = loader.get("calc_windower_0")
    out = ffi.new(f"{c_type}[{window_length}]")
    c_windows = []
    for data in samples.tolist():
        if calc_windower(data, out):
            c_windows.append([int(out[j]) for j in range(window_length)])

    assert len(c_windows) == len(py_windows), (
        f"Output count mismatch: python={len(py_windows)}, c={len(c_windows)}"
    )
    for k, (py_win, c_win) in enumerate(zip(py_windows.tolist(), c_windows)):
        for j, (pv, cv) in enumerate(zip(py_win, c_win)):
            passed, reason = compare_values(int(pv), cv)
            assert passed, f"window {k}, sample {j}: python={int(pv)}, c={cv}: {reason}"
