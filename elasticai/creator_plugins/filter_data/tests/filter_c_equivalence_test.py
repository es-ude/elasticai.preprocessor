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

INTEGER_CONFIGS = [
    pytest.param(8, np.int8, "signed char", id="int8"),
    pytest.param(32, np.int32, "signed int", id="int32"),
]


@pytest.mark.parametrize("target", ["mcu", "pc"])
def test_create_design_generates_fir_c_files(tmp_path: Path, target: str) -> None:
    settings = SettingsFilter(1.0, 1000.0, 6, [100.0], "fir", "butter", "lowpass")
    Filtering(settings).create_design(target, bitwidth=8, id="0", path2save=tmp_path)
    assert (tmp_path / "filter_fir_low0.c").exists()
    assert (tmp_path / "filter_fir_low0.h").exists()
    assert (tmp_path / "filter_fir_template.h").exists()


@pytest.mark.parametrize("target", ["mcu", "pc"])
def test_create_design_generates_fir_allpass_c_files(tmp_path: Path, target: str) -> None:
    settings = SettingsFilter(1.0, 1000.0, 4, [50.0], "fir", "butter", "allpass")
    Filtering(settings).create_design(target, bitwidth=8, id="0", path2save=tmp_path)
    assert (tmp_path / "filter_fir_all0.c").exists()
    assert (tmp_path / "filter_fir_all0.h").exists()
    assert (tmp_path / "filter_fir_all_template.h").exists()


@pytest.mark.parametrize("target", ["mcu", "pc"])
def test_create_design_generates_iir_c_files(tmp_path: Path, target: str) -> None:
    settings = SettingsFilter(1.0, 1000.0, 2, [100.0], "iir", "butter", "lowpass")
    Filtering(settings).create_design(target, bitwidth=8, id="0", path2save=tmp_path)
    assert (tmp_path / "filter_iir_low0.c").exists()
    assert (tmp_path / "filter_iir_low0.h").exists()
    assert (tmp_path / "filter_iir_template.h").exists()


@pytest.mark.parametrize("target", ["mcu", "pc"])
def test_create_design_generates_mavg_c_files(tmp_path: Path, target: str) -> None:
    settings = SettingsFilter(1.0, 1000.0, 4, [], "fir", "butter", "moving_average")
    Filtering(settings).create_design(target, bitwidth=8, id="0", path2save=tmp_path)
    assert (tmp_path / "filter_mavg_0.c").exists()
    assert (tmp_path / "filter_mavg_0.h").exists()
    assert (tmp_path / "filter_mavg_template.h").exists()


def assert_c_filter_equivalent(
    settings: SettingsFilter,
    tmp_path: Path,
    source_name: str,
    function_name: str,
    bitwidth: int,
    numpy_dtype: type[np.generic],
    c_type: str,
) -> None:
    filtering = Filtering(settings)
    output_dir = tmp_path / "src"
    filtering.create_design("mcu", bitwidth, "0", output_dir, signed=True)

    adapter = tmp_path / "adapter.h"
    adapter.write_text(f"_Bool {function_name}({c_type} data, {c_type} *out);\n")

    loader = CompileLoader(
        headers=str(adapter),
        sources=[str(output_dir / source_name)],
        build_dir=str(tmp_path / "cffi-build"),
        module_name=f"filter_equivalence_{uuid4().hex}",
    )
    loader.load()
    c_filter = loader.get(function_name)
    out = loader.ffi().new(f"{c_type} *")

    expected = filtering.filt(INPUT_DATA.astype(float)).astype(numpy_dtype)
    for index, (input_value, python_value) in enumerate(zip(INPUT_DATA, expected, strict=True)):
        c_filter(int(input_value), out)
        c_value = int(out[0])
        passed, reason = compare_values(int(python_value), c_value)
        assert passed, (
            f"{settings.type}/{settings.b_type} mismatch at sample {index}: "
            f"input={int(input_value)}, python={int(python_value)}, c={c_value}: {reason}"
        )


@pytest.mark.parametrize("bitwidth,numpy_dtype,c_type", INTEGER_CONFIGS)
def test_mavg_c_matches_python(
    tmp_path: Path,
    bitwidth: int,
    numpy_dtype: type[np.generic],
    c_type: str,
) -> None:
    settings = SettingsFilter(1.0, 1000.0, 4, [], "fir", "butter", "moving_average")
    assert_c_filter_equivalent(
        settings,
        tmp_path,
        "filter_mavg_0.c",
        "calc_moving_average_MAVG0",
        bitwidth,
        numpy_dtype,
        c_type,
    )


@pytest.mark.parametrize("bitwidth,numpy_dtype,c_type", INTEGER_CONFIGS)
def test_fir_full_c_matches_python(
    tmp_path: Path,
    bitwidth: int,
    numpy_dtype: type[np.generic],
    c_type: str,
) -> None:
    settings = SettingsFilter(1.0, 1000.0, 6, [100.0], "fir", "butter", "lowpass")
    assert_c_filter_equivalent(
        settings,
        tmp_path,
        "filter_fir_low0.c",
        "calc_filter_fir_LOW0",
        bitwidth,
        numpy_dtype,
        c_type,
    )


@pytest.mark.parametrize("bitwidth,numpy_dtype,c_type", INTEGER_CONFIGS)
def test_fir_optimized_c_matches_python(
    tmp_path: Path,
    bitwidth: int,
    numpy_dtype: type[np.generic],
    c_type: str,
) -> None:
    settings = SettingsFilter(1.0, 1000.0, 7, [100.0], "fir", "butter", "lowpass")
    assert_c_filter_equivalent(
        settings,
        tmp_path,
        "filter_fir_low0.c",
        "calc_filter_fir_LOW0",
        bitwidth,
        numpy_dtype,
        c_type,
    )


@pytest.mark.parametrize("bitwidth,numpy_dtype,c_type", INTEGER_CONFIGS)
def test_fir_allpass_c_matches_python(
    tmp_path: Path,
    bitwidth: int,
    numpy_dtype: type[np.generic],
    c_type: str,
) -> None:
    settings = SettingsFilter(1.0, 1000.0, 4, [50.0], "fir", "butter", "allpass")
    assert_c_filter_equivalent(
        settings,
        tmp_path,
        "filter_fir_all0.c",
        "calc_filter_fir_all_ALL0",
        bitwidth,
        numpy_dtype,
        c_type,
    )


@pytest.mark.parametrize("bitwidth,numpy_dtype,c_type", INTEGER_CONFIGS)
def test_iir_c_matches_python(
    tmp_path: Path,
    bitwidth: int,
    numpy_dtype: type[np.generic],
    c_type: str,
) -> None:
    settings = SettingsFilter(1.0, 1000.0, 2, [100.0], "iir", "butter", "lowpass")
    assert_c_filter_equivalent(
        settings,
        tmp_path,
        "filter_iir_low0.c",
        "calc_filter_iir_LOW0",
        bitwidth,
        numpy_dtype,
        c_type,
    )
