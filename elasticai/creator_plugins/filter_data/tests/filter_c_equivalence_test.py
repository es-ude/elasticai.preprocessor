from pathlib import Path
from shutil import which
from uuid import uuid4

import numpy as np
import pytest
from elasticai.equichecker import CompileLoader, compare_values

from elasticai.preprocessor.filter import Filtering, SettingsFilter

pytestmark = pytest.mark.skipif(which("cc") is None, reason="requires a C compiler")

INPUT_DATA = np.array(
    [
        0,
        64,
        -32,
        48,
        -64,
        16,
        32,
        -48,
        64,
        0,
        -16,
        48,
        -32,
        64,
        -64,
        32,
        16,
        -8,
        24,
        -40,
        8,
        0,
        12,
        -20,
    ],
    dtype=np.int8,
)


def assert_c_filter_equivalent(
    settings: SettingsFilter,
    tmp_path: Path,
    source_name: str,
    function_name: str,
) -> None:
    filtering = Filtering(settings)
    output_dir = tmp_path / "src"
    filtering.create_design("mcu", 8, "0", output_dir, signed=True)

    adapter = tmp_path / "adapter.h"
    adapter.write_text(f"signed char {function_name}(signed char data);\n")

    loader = CompileLoader(
        headers=str(adapter),
        sources=[str(output_dir / source_name)],
        build_dir=str(tmp_path / "cffi-build"),
        module_name=f"filter_equivalence_{uuid4().hex}",
    )
    loader.load()
    c_filter = loader.get(function_name)

    expected = filtering.filt(INPUT_DATA.astype(float)).astype(np.int8)
    for index, (input_value, python_value) in enumerate(zip(INPUT_DATA, expected, strict=True)):
        c_value = int(c_filter(int(input_value)))
        passed, reason = compare_values(int(python_value), c_value)
        assert passed, (
            f"{settings.type}/{settings.b_type} mismatch at sample {index}: "
            f"input={int(input_value)}, python={int(python_value)}, c={c_value}: {reason}"
        )


def test_fir_full_c_matches_python(tmp_path: Path) -> None:
    settings = SettingsFilter(1.0, 1000.0, 6, [100.0], "fir", "butter", "lowpass")
    assert_c_filter_equivalent(settings, tmp_path, "filter_fir_low0.c", "calc_filter_fir_LOW0")


def test_fir_optimized_c_matches_python(tmp_path: Path) -> None:
    settings = SettingsFilter(1.0, 1000.0, 7, [100.0], "fir", "butter", "lowpass")
    assert_c_filter_equivalent(settings, tmp_path, "filter_fir_low0.c", "calc_filter_fir_LOW0")


def test_fir_allpass_c_matches_python(tmp_path: Path) -> None:
    settings = SettingsFilter(1.0, 1000.0, 4, [50.0], "fir", "butter", "allpass")
    assert_c_filter_equivalent(
        settings,
        tmp_path,
        "filter_fir_all0.c",
        "calc_filter_fir_all_ALL0",
    )


def test_iir_c_matches_python(tmp_path: Path) -> None:
    settings = SettingsFilter(1.0, 1000.0, 2, [100.0], "iir", "butter", "lowpass")
    assert_c_filter_equivalent(settings, tmp_path, "filter_iir_low0.c", "calc_filter_iir_LOW0")
