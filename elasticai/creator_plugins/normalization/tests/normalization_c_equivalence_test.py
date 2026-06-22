from pathlib import Path
from shutil import which
from uuid import uuid4

import numpy as np
import pytest
from elasticai.equichecker import CompileLoader, compare_values

from elasticai.preprocessor.normalization import DataNormalization

pytestmark = pytest.mark.skipif(which("cc") is None, reason="requires a C compiler")

INTEGER_CONFIGS = [
    pytest.param(8, np.int8, "signed char", id="int8"),
    pytest.param(32, np.int32, "signed int", id="int32"),
]


@pytest.mark.parametrize("target", ["mcu", "pc"])
def test_create_design_generates_minmax_c_files(tmp_path: Path, target: str) -> None:
    normalizer = DataNormalization("minmax", peak_mode=2)

    normalizer.create_design(target, 8, "0", tmp_path, signed=True)

    assert (tmp_path / "normalization_minmax_0.c").exists()
    assert (tmp_path / "normalization_minmax_0.h").exists()
    assert (tmp_path / "normalization_minmax_template.h").exists()


def test_create_design_rejects_unknown_target(tmp_path: Path) -> None:
    normalizer = DataNormalization("minmax", peak_mode=2)

    with pytest.raises(ValueError, match="Target unknown is not supported"):
        normalizer.create_design("unknown", 8, "0", tmp_path)


def test_create_design_rejects_fpga_target(tmp_path: Path) -> None:
    normalizer = DataNormalization("minmax", peak_mode=2)

    with pytest.raises(NotImplementedError, match="FPGA normalization"):
        normalizer.create_design("fpga", 8, "0", tmp_path)


def test_create_design_rejects_other_normalization_methods(tmp_path: Path) -> None:
    normalizer = DataNormalization("zscore")

    with pytest.raises(NotImplementedError, match="only minmax"):
        normalizer.create_design("mcu", 8, "0", tmp_path)


def test_create_design_rejects_other_peak_modes(tmp_path: Path) -> None:
    normalizer = DataNormalization("minmax", peak_mode=0)

    with pytest.raises(NotImplementedError, match="peak_mode=2"):
        normalizer.create_design("mcu", 8, "0", tmp_path)


def test_create_design_rejects_global_scaling(tmp_path: Path) -> None:
    normalizer = DataNormalization("minmax", do_global_scaling=True)

    with pytest.raises(NotImplementedError, match="global scaling"):
        normalizer.create_design("mcu", 8, "0", tmp_path)


@pytest.mark.parametrize("bitwidth,numpy_dtype,c_type", INTEGER_CONFIGS)
def test_generated_minmax_c_matches_python_frames(
    tmp_path: Path,
    bitwidth: int,
    numpy_dtype: type[np.generic],
    c_type: str,
) -> None:
    normalizer = DataNormalization("minmax", peak_mode=2)
    output_dir = tmp_path / "src"
    normalizer.create_design("mcu", bitwidth, "0", output_dir, signed=True)

    adapter = tmp_path / "adapter.h"
    adapter.write_text(
        f"void normalize_minmax_0(const {c_type} *input, float *output, unsigned int length);\n"
    )
    loader = CompileLoader(
        headers=str(adapter),
        sources=[str(output_dir / "normalization_minmax_0.c")],
        build_dir=str(tmp_path / "cffi-build"),
        module_name=f"normalization_equivalence_{uuid4().hex}",
    )
    loader.load()

    info = np.iinfo(numpy_dtype)
    frames = [
        [info.min, -64, 0, 32, info.max],
        [1, 2, 3, 4],
        [-1, -2, -3, -4],
        [7],
        [0, 0, 0],
        [12, -3, 5, -9, 1, 8, -2],
    ]
    normalize_c = loader.get("normalize_minmax_0")

    for input_values in frames:
        input_frame = np.array(input_values, dtype=numpy_dtype)
        with np.errstate(divide="ignore", invalid="ignore"):
            expected = normalizer.normalize(input_frame.astype(np.float32))
        c_input = loader.ffi().new(f"{c_type}[]", input_frame.tolist())
        c_output = loader.ffi().new("float[]", len(input_frame))

        normalize_c(c_input, c_output, len(input_frame))

        for index, (expected_value, c_value) in enumerate(zip(expected, c_output, strict=True)):
            passed, reason = compare_values(float(expected_value), float(c_value))
            assert passed, f"frame={input_values}, index={index}: {reason}"


@pytest.mark.parametrize("bitwidth,_,c_type", INTEGER_CONFIGS)
def test_generated_minmax_c_accepts_empty_frame(
    tmp_path: Path,
    bitwidth: int,
    _: type[np.generic],
    c_type: str,
) -> None:
    normalizer = DataNormalization("minmax", peak_mode=2)
    output_dir = tmp_path / "src"
    normalizer.create_design("mcu", bitwidth, "0", output_dir, signed=True)

    adapter = tmp_path / "adapter.h"
    adapter.write_text(
        f"void normalize_minmax_0(const {c_type} *input, float *output, unsigned int length);\n"
    )
    loader = CompileLoader(
        headers=str(adapter),
        sources=[str(output_dir / "normalization_minmax_0.c")],
        build_dir=str(tmp_path / "cffi-build"),
        module_name=f"normalization_empty_{uuid4().hex}",
    )
    loader.load()
    c_input = loader.ffi().new(f"{c_type}[]", [0])
    c_output = loader.ffi().new("float[]", [23.0])

    loader.get("normalize_minmax_0")(c_input, c_output, 0)

    assert float(c_output[0]) == 23.0
