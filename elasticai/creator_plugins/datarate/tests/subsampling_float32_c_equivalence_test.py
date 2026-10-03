from pathlib import Path
from shutil import which
from uuid import uuid4

import numpy as np
import pytest
from elasticai.equichecker import CompileLoader

from elasticai.preprocessor.downsampling import DownSampling, SettingsDownSampling

pytestmark = pytest.mark.skipif(which("cc") is None, reason="requires a C compiler")


def _load_downsampling(tmp_path: Path, factor: int):
    downsampler = DownSampling(SettingsDownSampling(sampling_rate=1000.0, dsr=factor))
    output_dir = tmp_path / "src"
    downsampler.create_design_float32("mcu", "classifier", output_dir)

    adapter = tmp_path / "adapter.h"
    adapter.write_text(
        """
unsigned int get_downsampling_float32_classifier_output_length(
    unsigned int input_length
);
_Bool downsample_float32_classifier(
    const float *input,
    unsigned int input_length,
    float *output,
    unsigned int output_capacity
);
"""
    )
    loader = CompileLoader(
        headers=str(adapter),
        sources=[str(output_dir / "downsampling_float32_classifier.c")],
        build_dir=str(tmp_path / "cffi-build"),
        module_name=f"downsampling_float32_{uuid4().hex}",
    )
    loader.load()
    return downsampler, loader


@pytest.mark.parametrize("target", ["mcu", "pc"])
def test_create_design_float32_generates_expected_files(tmp_path: Path, target: str) -> None:
    downsampler = DownSampling(SettingsDownSampling(sampling_rate=1000.0, dsr=2))

    downsampler.create_design_float32(target, "classifier", tmp_path)

    assert {file.name for file in tmp_path.iterdir()} == {
        "downsampling_float32_classifier.c",
        "downsampling_float32_classifier.h",
        "downsampling_float32_template.h",
    }


def test_create_design_float32_skips_factor_one(tmp_path: Path) -> None:
    downsampler = DownSampling(SettingsDownSampling(sampling_rate=1000.0, dsr=1))

    downsampler.create_design_float32("mcu", "classifier", tmp_path)

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("factor", [0, -1, 1.5, True])
def test_create_design_float32_rejects_invalid_factor(tmp_path: Path, factor: object) -> None:
    downsampler = DownSampling(SettingsDownSampling(sampling_rate=1000.0, dsr=factor))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="positive integer"):
        downsampler.create_design_float32("mcu", "classifier", tmp_path)


def test_create_design_float32_rejects_unsupported_target(tmp_path: Path) -> None:
    downsampler = DownSampling(SettingsDownSampling(sampling_rate=1000.0, dsr=2))

    with pytest.raises(ValueError, match="Target fpga is not supported"):
        downsampler.create_design_float32("fpga", "classifier", tmp_path)


@pytest.mark.parametrize("factor", [2, 3])
def test_generated_float32_subsampling_matches_python_for_even_and_odd_lengths(
    tmp_path: Path, factor: int
) -> None:
    downsampler, loader = _load_downsampling(tmp_path, factor)
    input_frame = np.asarray([0.25, -1.0, 2.5, 3.0, -4.75, 8.0, 9.5], dtype=np.float32)
    expected = downsampler.do_subsampling(input_frame, augment=False)
    get_output_length = loader.get("get_downsampling_float32_classifier_output_length")
    downsample = loader.get("downsample_float32_classifier")
    output_length = int(get_output_length(input_frame.size))
    assert int(get_output_length(1000)) == 1000 // factor + (1000 % factor != 0)
    c_input = loader.ffi().new("float[]", input_frame.tolist())
    c_output = loader.ffi().new("float[]", output_length)

    assert downsample(c_input, input_frame.size, c_output, output_length)

    actual = np.asarray([float(c_output[index]) for index in range(output_length)], dtype=np.float32)
    np.testing.assert_array_equal(actual, expected)


def test_generated_float32_subsampling_validates_capacity_and_supports_in_place(
    tmp_path: Path,
) -> None:
    downsampler, loader = _load_downsampling(tmp_path, factor=2)
    input_frame = np.asarray([0.25, -1.0, 2.5, 3.0, -4.75], dtype=np.float32)
    expected = downsampler.do_subsampling(input_frame, augment=False)
    downsample = loader.get("downsample_float32_classifier")
    c_frame = loader.ffi().new("float[]", input_frame.tolist())

    assert not downsample(c_frame, input_frame.size, c_frame, expected.size - 1)
    assert downsample(c_frame, input_frame.size, c_frame, expected.size)

    actual = np.asarray([float(c_frame[index]) for index in range(expected.size)], dtype=np.float32)
    np.testing.assert_array_equal(actual, expected)

    assert downsample(loader.ffi().NULL, 0, loader.ffi().NULL, 0)
