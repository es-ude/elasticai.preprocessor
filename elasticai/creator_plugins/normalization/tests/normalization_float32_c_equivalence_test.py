from pathlib import Path
from shutil import which
from uuid import uuid4

import numpy as np
import pytest
from elasticai.equichecker import CompileLoader

from elasticai.preprocessor.normalization import DataNormalization, SettingsNormalization

pytestmark = pytest.mark.skipif(which("cc") is None, reason="requires a C compiler")


def _settings(method: str, peak_mode: int = 2) -> SettingsNormalization:
    return SettingsNormalization(method=method, peak_mode=peak_mode)


def _load_normalization(tmp_path: Path, method: str):
    normalizer = DataNormalization(settings=_settings(method))
    output_dir = tmp_path / "src"
    normalizer.create_design_float32("mcu", "0", output_dir)

    function_name = f"normalize_{method}_float32_0"
    adapter = tmp_path / "adapter.h"
    adapter.write_text(f"void {function_name}(const float *input, float *output, unsigned int length);\n")
    loader = CompileLoader(
        headers=str(adapter),
        sources=[str(output_dir / f"normalization_{method}_float32_0.c")],
        build_dir=str(tmp_path / "cffi-build"),
        module_name=f"normalization_{method}_float32_{uuid4().hex}",
    )
    loader.load()
    return normalizer, loader, loader.get(function_name)


@pytest.mark.parametrize("target", ["mcu", "pc"])
@pytest.mark.parametrize("method", ["minmax", "zscore"])
def test_create_design_float32_generates_expected_files(tmp_path: Path, target: str, method: str) -> None:
    DataNormalization(settings=_settings(method)).create_design_float32(target, "0", tmp_path)

    assert {file.name for file in tmp_path.iterdir()} == {
        f"normalization_{method}_float32_0.c",
        f"normalization_{method}_float32_0.h",
        f"normalization_{method}_template.h",
    }


@pytest.mark.parametrize("method", ["minmax", "zscore"])
def test_float32_normalization_matches_python_per_channel_and_stays_finite(
    tmp_path: Path, method: str
) -> None:
    normalizer, loader, normalize_c = _load_normalization(tmp_path, method)
    frames = np.stack(
        [
            np.linspace(-4.0, 4.0, 50, dtype=np.float32),
            np.linspace(1000.0, -125.0, 50, dtype=np.float32),
            np.full(50, 0.1, dtype=np.float32),
            np.zeros(50, dtype=np.float32),
        ],
    )
    expected = normalizer.normalize(frames)
    actual = np.empty_like(frames)

    for channel, frame in enumerate(frames):
        c_input = loader.ffi().new("float[]", frame.tolist())
        c_output = loader.ffi().new("float[]", frame.size)
        normalize_c(c_input, c_output, frame.size)
        actual[channel] = [float(c_output[index]) for index in range(frame.size)]

    assert np.all(np.isfinite(expected))
    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)

    if method == "minmax":
        np.testing.assert_array_equal(actual[2], np.ones(frames.shape[1], dtype=np.float32))
    else:
        np.testing.assert_array_equal(actual[2], np.zeros(frames.shape[1], dtype=np.float32))
    np.testing.assert_array_equal(actual[3], np.zeros(frames.shape[1], dtype=np.float32))


@pytest.mark.parametrize("method", ["minmax", "zscore"])
def test_float32_normalization_supports_in_place_and_empty_frames(tmp_path: Path, method: str) -> None:
    normalizer, loader, normalize_c = _load_normalization(tmp_path, method)
    frame = np.asarray([-3.0, -1.0, 2.0, 5.0], dtype=np.float32)
    expected = normalizer.normalize(frame)
    c_frame = loader.ffi().new("float[]", frame.tolist())

    normalize_c(c_frame, c_frame, frame.size)

    actual = np.asarray([float(c_frame[index]) for index in range(frame.size)], dtype=np.float32)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)

    sentinel = loader.ffi().new("float[]", [23.0])
    normalize_c(loader.ffi().NULL, sentinel, 0)
    assert float(sentinel[0]) == 23.0


def test_create_design_float32_rejects_unsupported_configuration(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Target fpga is not supported"):
        DataNormalization(settings=_settings("minmax")).create_design_float32("fpga", "0", tmp_path)

    with pytest.raises(NotImplementedError, match="only minmax and zscore"):
        DataNormalization(settings=_settings("norm")).create_design_float32("mcu", "0", tmp_path)

    with pytest.raises(NotImplementedError, match="peak_mode=2"):
        DataNormalization(settings=_settings("minmax", peak_mode=0)).create_design_float32(
            "mcu", "0", tmp_path
        )
