from pathlib import Path
from shutil import which
from uuid import uuid4

import numpy as np
import pytest
from elasticai.equichecker import CompileLoader, compare_values

from elasticai.preprocessor.downsampling import DownSampling, SettingsDownSampling

pytestmark = pytest.mark.skipif(which("cc") is None, reason="requires a C compiler")

INTEGER_CONFIGS = [
    pytest.param(8, np.int8, "signed char", id="int8"),
    pytest.param(32, np.int32, "signed int", id="int32"),
]


@pytest.mark.parametrize("target", ["mcu", "pc"])
def test_create_design_generates_subsampling_c_files(tmp_path: Path, target: str) -> None:
    downsampler = DownSampling(SettingsDownSampling(sampling_rate=1000.0, dsr=3))

    downsampler.create_design(target, 8, "0", tmp_path, signed=True)

    assert (tmp_path / "downsampling_subsampling_0.c").exists()
    assert (tmp_path / "downsampling_subsampling_0.h").exists()
    assert (tmp_path / "downsampling_subsampling_template.h").exists()


def test_create_design_rejects_unknown_target(tmp_path: Path) -> None:
    downsampler = DownSampling(SettingsDownSampling(sampling_rate=1000.0, dsr=3))

    with pytest.raises(ValueError, match="Target unknown is not supported"):
        downsampler.create_design("unknown", 8, "0", tmp_path)


def test_create_design_rejects_fpga_target(tmp_path: Path) -> None:
    downsampler = DownSampling(SettingsDownSampling(sampling_rate=1000.0, dsr=3))

    with pytest.raises(NotImplementedError, match="FPGA downsampling"):
        downsampler.create_design("fpga", 8, "0", tmp_path)


def test_create_design_rejects_invalid_downsampling_ratio(tmp_path: Path) -> None:
    downsampler = DownSampling(SettingsDownSampling(sampling_rate=1000.0, dsr=0))

    with pytest.raises(ValueError, match="dsr must be >= 1"):
        downsampler.create_design("mcu", 8, "0", tmp_path)


@pytest.mark.parametrize("bitwidth,numpy_dtype,c_type", INTEGER_CONFIGS)
@pytest.mark.parametrize("augment", [False, True])
def test_generated_subsampling_c_matches_python_frame(
    tmp_path: Path,
    bitwidth: int,
    numpy_dtype: type[np.generic],
    c_type: str,
    augment: bool,
) -> None:
    settings = SettingsDownSampling(sampling_rate=1000.0, dsr=3)
    downsampler = DownSampling(settings)
    output_dir = tmp_path / "src"
    downsampler.create_design("mcu", bitwidth, "0", output_dir, signed=True)

    adapter = tmp_path / "adapter.h"
    adapter.write_text(
        f"unsigned int get_downsampling_subsampling_output_length_0(unsigned int input_length);\n"
        f"void downsample_subsampling_0("
        f"const {c_type} *input, {c_type} *output, unsigned int input_length, unsigned char augment);\n"
    )
    loader = CompileLoader(
        headers=str(adapter),
        sources=[str(output_dir / "downsampling_subsampling_0.c")],
        build_dir=str(tmp_path / "cffi-build"),
        module_name=f"downsampling_subsampling_equivalence_{uuid4().hex}",
    )
    loader.load()

    input_frame = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=numpy_dtype)
    expected = downsampler.do_subsampling(input_frame.reshape(1, -1), augment=augment)
    expected_frame = expected.reshape(-1).astype(numpy_dtype)
    output_length = int(loader.get("get_downsampling_subsampling_output_length_0")(len(input_frame)))
    output_count = output_length * settings.dsr if augment else output_length

    c_input = loader.ffi().new(f"{c_type}[]", input_frame.tolist())
    c_output = loader.ffi().new(f"{c_type}[]", output_count)

    loader.get("downsample_subsampling_0")(c_input, c_output, len(input_frame), int(augment))

    for index, (expected_value, c_value) in enumerate(
        zip(expected_frame.tolist(), c_output, strict=True)
    ):
        passed, reason = compare_values(int(expected_value), int(c_value))
        assert passed, f"index={index}: {reason}"
