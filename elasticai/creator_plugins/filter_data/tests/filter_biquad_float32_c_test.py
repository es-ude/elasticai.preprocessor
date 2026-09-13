from pathlib import Path
from shutil import which
from uuid import uuid4

import numpy as np
import pytest
from elasticai.equichecker import CompileLoader

from elasticai.preprocessor.filter import Filtering, SettingsFilter

pytestmark = pytest.mark.skipif(which("cc") is None, reason="requires a C compiler")


def _bandpass_settings() -> SettingsFilter:
    return SettingsFilter(1.0, 1000.0, 2, [225.0, 375.0], "iir", "butter", "bandpass")


def _input_data() -> np.ndarray:
    time = np.arange(256, dtype=np.float32) / np.float32(1000.0)
    return (
        1000.0 * np.sin(2.0 * np.pi * 50.0 * time) + 2000.0 * np.sin(2.0 * np.pi * 300.0 * time)
    ).astype(np.float32)


@pytest.mark.parametrize("target", ["mcu", "pc"])
def test_build_generates_float32_iir_files(tmp_path: Path, target: str) -> None:
    Filtering(_bandpass_settings()).create_design_float32(target, "0", tmp_path)

    assert {file.name for file in tmp_path.iterdir()} == {
        "filter_iir_float32_band_0.c",
        "filter_iir_float32_band_0.h",
        "filter_iir_float32_template.h",
    }


def test_float32_iir_matches_python_and_has_resettable_independent_state(tmp_path: Path) -> None:
    filtering = Filtering(_bandpass_settings())
    output_dir = tmp_path / "src"
    filtering.create_design_float32("mcu", "0", output_dir)

    adapter = tmp_path / "adapter.h"
    adapter.write_text(
        """
typedef struct {
    unsigned short tap_start;
    float taps[4];
} IirFilterFloat32State_band_0;

void reset_filter_iir_float32_band_0(IirFilterFloat32State_band_0 *state);
_Bool filt_iir_float32_band_0(
    IirFilterFloat32State_band_0 *state,
    float data,
    float *out
);
"""
    )
    loader = CompileLoader(
        headers=str(adapter),
        sources=[str(output_dir / "filter_iir_float32_band_0.c")],
        build_dir=str(tmp_path / "cffi-build"),
        module_name=f"filter_float32_{uuid4().hex}",
    )
    loader.load()

    reset_filter = loader.get("reset_filter_iir_float32_band_0")
    filter_sample = loader.get("filt_iir_float32_band_0")
    ffi = loader.ffi()
    state_a = ffi.new("IirFilterFloat32State_band_0 *")
    state_b = ffi.new("IirFilterFloat32State_band_0 *")
    output_a = ffi.new("float *")
    output_b = ffi.new("float *")
    data = _input_data()
    expected = filtering.filt(data).astype(np.float32)

    def process(state, output) -> np.ndarray:
        values = []
        for sample in data:
            assert filter_sample(state, float(sample), output)
            values.append(float(output[0]))
        return np.asarray(values, dtype=np.float32)

    reset_filter(state_a)
    first_run = process(state_a, output_a)
    reset_filter(state_a)
    second_run = process(state_a, output_a)
    np.testing.assert_allclose(first_run, expected, atol=1e-4, rtol=2e-5)
    np.testing.assert_allclose(second_run, first_run, atol=0.0, rtol=0.0)

    reset_filter(state_a)
    reset_filter(state_b)
    interleaved_a = []
    interleaved_b = []
    for sample in data:
        assert filter_sample(state_a, float(sample), output_a)
        assert filter_sample(state_b, float(sample), output_b)
        interleaved_a.append(float(output_a[0]))
        interleaved_b.append(float(output_b[0]))

    np.testing.assert_allclose(interleaved_a, expected, atol=1e-4, rtol=2e-5)
    np.testing.assert_allclose(interleaved_b, expected, atol=1e-4, rtol=2e-5)
