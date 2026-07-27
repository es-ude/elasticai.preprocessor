from pathlib import Path
from shutil import which
from uuid import uuid4

import numpy as np
import pytest
from elasticai.equichecker import CompileLoader, compare_values

from elasticai.preprocessor.downsampling import DownSampling, SettingsDownSampling, TargetsDownSampling

pytestmark = pytest.mark.skipif(which("cc") is None, reason="requires a C compiler")

INTEGER_CONFIGS = [
    pytest.param(8, np.int8, "signed char", id="int8"),
    pytest.param(32, np.int32, "signed int", id="int32"),
]


@pytest.mark.parametrize("target", ["mcu", "pc"])
def test_create_design_generates_cic_c_files(tmp_path: Path, target: str) -> None:
    downsampler = DownSampling(SettingsDownSampling(sampling_rate=1000.0, dsr=4))

    downsampler.create_design(
        method=TargetsDownSampling.CIC,
        target=target,
        bitwidth=32,
        id="0",
        path2save=tmp_path,
        signed=True,
        num_stages=3,
    )

    assert (tmp_path / "downsampling_cic_0.c").exists()
    assert (tmp_path / "downsampling_cic_0.h").exists()
    assert (tmp_path / "downsampling_cic_template.h").exists()


def test_create_design_rejects_invalid_downsampling_ratio(tmp_path: Path) -> None:
    downsampler = DownSampling(SettingsDownSampling(sampling_rate=1000.0, dsr=0))

    with pytest.raises(ValueError, match="dsr must be >= 1"):
        downsampler.create_design(
            method=TargetsDownSampling.CIC,
            target="mcu",
            bitwidth=8,
            id="0",
            path2save=tmp_path,
        )


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth,numpy_dtype,c_type", INTEGER_CONFIGS)
def test_generated_cic_c_matches_python_frame(
    tmp_path: Path,
    bitwidth: int,
    numpy_dtype: type[np.generic],
    c_type: str,
) -> None:
    num_stages = 3
    settings = SettingsDownSampling(sampling_rate=1000.0, dsr=4)
    downsampler = DownSampling(settings)
    output_dir = tmp_path / "src"
    downsampler.create_design(
        method=TargetsDownSampling.CIC,
        target="mcu",
        bitwidth=bitwidth,
        id="0",
        path2save=output_dir,
        signed=True,
        num_stages=num_stages,
    )

    adapter = tmp_path / "adapter.h"
    adapter.write_text(f"_Bool calc_cic_0({c_type} data, {c_type} *out);\n")
    loader = CompileLoader(
        headers=str(adapter),
        sources=[str(output_dir / "downsampling_cic_0.c")],
        build_dir=str(tmp_path / "cffi-build"),
        module_name=f"downsampling_cic_equivalence_{uuid4().hex}",
    )
    loader.load()

    input_frame = np.array([10] * 16, dtype=numpy_dtype)
    expected_float = downsampler.do_cic(input_frame.astype(float), num_stages=num_stages)
    expected = np.array([int(v) for v in expected_float], dtype=numpy_dtype)

    out = loader.ffi().new(f"{c_type} *")
    c_results = []
    for sample in input_frame.tolist():
        if loader.get("calc_cic_0")(sample, out):
            c_results.append(int(out[0]))

    for index, (expected_value, c_value) in enumerate(zip(expected.tolist(), c_results, strict=True)):
        passed, reason = compare_values(int(expected_value), c_value)
        assert passed, f"index={index}: {reason}"


@pytest.mark.parametrize("bitwidth,numpy_dtype,c_type", INTEGER_CONFIGS)
def test_generated_cic_c_matches_python_sinewave(
    tmp_path: Path,
    bitwidth: int,
    numpy_dtype: type[np.generic],
    c_type: str,
) -> None:
    num_stages = 3
    settings = SettingsDownSampling(sampling_rate=1000.0, dsr=4)
    downsampler = DownSampling(settings)
    output_dir = tmp_path / "src"
    downsampler.create_design(
        method=TargetsDownSampling.CIC,
        target="mcu",
        bitwidth=bitwidth,
        id="0",
        path2save=output_dir,
        signed=True,
        num_stages=num_stages,
    )

    adapter = tmp_path / "adapter.h"
    adapter.write_text(f"_Bool calc_cic_0({c_type} data, {c_type} *out);\n")
    loader = CompileLoader(
        headers=str(adapter),
        sources=[str(output_dir / "downsampling_cic_0.c")],
        build_dir=str(tmp_path / "cffi-build"),
        module_name=f"downsampling_cic_sinewave_{uuid4().hex}",
    )
    loader.load()

    amplitude = 100 if bitwidth == 8 else 10000
    t = np.arange(64) / settings.sampling_rate
    input_frame = (np.sin(2 * np.pi * 10 * t) * amplitude).astype(numpy_dtype)

    expected_float = downsampler.do_cic(input_frame.astype(float), num_stages=num_stages)
    expected = np.array([int(v) for v in expected_float], dtype=numpy_dtype)

    out = loader.ffi().new(f"{c_type} *")
    c_results = []
    for sample in input_frame.tolist():
        if loader.get("calc_cic_0")(sample, out):
            c_results.append(int(out[0]))

    for index, (expected_value, c_value) in enumerate(zip(expected.tolist(), c_results, strict=True)):
        passed, reason = compare_values(int(expected_value), c_value)
        assert passed, f"index={index}: {reason}"
