from pathlib import Path
from shutil import which
from uuid import uuid4

import pytest
from elasticai.equichecker import CompileLoader, compare_values

from elasticai.creator_plugins.windower.src.c_compile import build_windower
from elasticai.preprocessor.windower.window import SettingsWindow

pytestmark = pytest.mark.skipif(which("cc") is None, reason="requires a C compiler")

WINDOWER_CONFIGS = [
    pytest.param(
        SettingsWindow(sampling_rate=100.0, window_sec=0.10, overlap_sec=0.05),
        16, True, "signed short",
        id="int16_window10_overlap5",
    ),
    pytest.param(
        SettingsWindow(sampling_rate=100.0, window_sec=0.08, overlap_sec=0.04),
        8, True, "signed char",
        id="int8_window8_overlap4",
    ),
    pytest.param(
        SettingsWindow(sampling_rate=100.0, window_sec=0.06, overlap_sec=0.0),
        8, False, "unsigned char",
        id="uint8_window6_nooverlap",
    ),
]


def _py_windower(samples: list[int], window_length: int, overlap_length: int) -> list[tuple[int, list[int]]]:
    """Python reference: mirrors calc_next_datum_windower_ from windower_template.h."""
    num_shift = window_length - overlap_length
    buf = [0] * window_length
    count = 0
    shift_cnt = 0
    results: list[tuple[int, list[int]]] = []

    for i, data in enumerate(samples):
        for j in range(1, window_length):
            buf[j - 1] = buf[j]
        buf[window_length - 1] = data

        if count < window_length:
            count += 1
            if count == window_length:
                shift_cnt = 0
                results.append((i, list(buf)))
        else:
            shift_cnt += 1
            if shift_cnt >= num_shift:
                shift_cnt = 0
                results.append((i, list(buf)))

    return results


def test_build_windower_generates_c_files(tmp_path: Path) -> None:
    settings = SettingsWindow(sampling_rate=100.0, window_sec=0.10, overlap_sec=0.05)
    build_windower(settings=settings, bitwidth=16, signed=True, path2save=tmp_path, windower_id="0")
    assert (tmp_path / "windower_template.h").exists()
    assert (tmp_path / "windower_0.h").exists()
    assert (tmp_path / "windower_0.c").exists()


@pytest.mark.parametrize("settings,bitwidth,signed,c_type", WINDOWER_CONFIGS)
def test_windower_c_matches_python(
    tmp_path: Path,
    settings: SettingsWindow,
    bitwidth: int,
    signed: bool,
    c_type: str,
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

    adapter = tmp_path / "adapter.h"
    adapter.write_text(f"_Bool calc_windower_0({c_type} data, {c_type} *out);\n")
    loader = CompileLoader(
        headers=str(adapter),
        sources=[str(output_dir / "windower_0.c")],
        build_dir=str(tmp_path / "cffi-build"),
        module_name=f"windower_equivalence_{uuid4().hex}",
    )
    loader.load()

    num_shift = window_length - settings.overlap_length
    num_samples = window_length + 3 * num_shift
    samples = list(range(num_samples))

    py_results = _py_windower(samples, window_length, settings.overlap_length)

    ffi = loader.ffi()
    calc_windower = loader.get("calc_windower_0")
    out = ffi.new(f"{c_type}[{window_length}]")

    c_results: list[tuple[int, list[int]]] = []
    for i, data in enumerate(samples):
        if calc_windower(data, out):
            c_results.append((i, [int(out[j]) for j in range(window_length)]))

    assert len(c_results) == len(py_results), (
        f"Output count mismatch: python={len(py_results)}, c={len(c_results)}"
    )
    for k, ((pi, py_win), (ci, c_win)) in enumerate(zip(py_results, c_results)):
        assert pi == ci, f"window {k}: trigger index mismatch python={pi}, c={ci}"
        for j, (pv, cv) in enumerate(zip(py_win, c_win)):
            passed, reason = compare_values(pv, cv)
            assert passed, f"window {k}, sample {j}: python={pv}, c={cv}: {reason}"
