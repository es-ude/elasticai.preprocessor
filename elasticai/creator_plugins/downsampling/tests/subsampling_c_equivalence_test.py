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
def test_create_design_generates_subsampling_c_files(tmp_path: Path, target: str) -> None:
    downsampler = DownSampling(SettingsDownSampling(sampling_rate=1000.0, dsr=3))

    downsampler.create_design(
        method=TargetsDownSampling.Subsampling,
        target=target,
        bitwidth=8,
        id="0",
        path2save=tmp_path,
        signed=True,
    )

    assert (tmp_path / "downsampling_subsampling_0.c").exists()
    assert (tmp_path / "downsampling_subsampling_0.h").exists()
    assert (tmp_path / "downsampling_subsampling_template.h").exists()


def test_create_design_rejects_invalid_downsampling_ratio(tmp_path: Path) -> None:
    downsampler = DownSampling(SettingsDownSampling(sampling_rate=1000.0, dsr=0))

    with pytest.raises(ValueError, match="dsr must be >= 1"):
        downsampler.create_design(
            method=TargetsDownSampling.Subsampling,
            target="mcu",
            bitwidth=8,
            id="0",
            path2save=tmp_path,
        )


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
    downsampler.create_design(
        method=TargetsDownSampling.Subsampling,
        target="mcu",
        bitwidth=bitwidth,
        id="0",
        path2save=output_dir,
        signed=True,
    )

    adapter = tmp_path / "adapter.h"
    adapter.write_text(f"_Bool calc_do_subsampling_0({c_type} data, {c_type} *out);\n")
    loader = CompileLoader(
        headers=str(adapter),
        sources=[str(output_dir / "downsampling_subsampling_0.c")],
        build_dir=str(tmp_path / "cffi-build"),
        module_name=f"downsampling_subsampling_equivalence_{uuid4().hex}",
    )
    loader.load()

    dsr = settings.dsr
    input_frame = np.arange(9, dtype=numpy_dtype)  # 9 = 3 vollständige Fenster à dsr=3
    if augment:
        expected = input_frame  # alle Elemente jedes Fensters in Reihenfolge
    else:
        expected = input_frame[0::dsr]  # erstes Element jedes Fensters

    out = loader.ffi().new(f"{c_type}[{dsr}]")
    c_results = []
    for sample in input_frame.tolist():
        if loader.get("calc_do_subsampling_0")(sample, out):
            if augment:
                c_results.extend(int(out[i]) for i in range(dsr))
            else:
                c_results.append(int(out[0]))

    for index, (expected_value, c_value) in enumerate(
        zip(expected.tolist(), c_results, strict=True)
    ):
        passed, reason = compare_values(int(expected_value), c_value)
        assert passed, f"augment={augment}, index={index}: {reason}"


@pytest.mark.parametrize("bitwidth,numpy_dtype,c_type", INTEGER_CONFIGS)
def test_generated_subsampling_c_matches_python_sinewave(
    tmp_path: Path,
    bitwidth: int,
    numpy_dtype: type[np.generic],
    c_type: str,
) -> None:
    settings = SettingsDownSampling(sampling_rate=1000.0, dsr=3)
    downsampler = DownSampling(settings)
    output_dir = tmp_path / "src"
    downsampler.create_design(
        method=TargetsDownSampling.Subsampling,
        target="mcu",
        bitwidth=bitwidth,
        id="0",
        path2save=output_dir,
        signed=True,
    )

    adapter = tmp_path / "adapter.h"
    adapter.write_text(f"_Bool calc_do_subsampling_0({c_type} data, {c_type} *out);\n")
    loader = CompileLoader(
        headers=str(adapter),
        sources=[str(output_dir / "downsampling_subsampling_0.c")],
        build_dir=str(tmp_path / "cffi-build"),
        module_name=f"downsampling_subsampling_sinewave_{uuid4().hex}",
    )
    loader.load()

    dsr = settings.dsr
    amplitude = 100 if bitwidth == 8 else 10000
    t = np.arange(60) / settings.sampling_rate  # 60 = 20 vollständige Fenster à dsr=3
    input_frame = (np.sin(2 * np.pi * 10 * t) * amplitude).astype(numpy_dtype)
    expected = input_frame[0::dsr]

    out = loader.ffi().new(f"{c_type}[{dsr}]")
    c_results = []
    for sample in input_frame.tolist():
        if loader.get("calc_do_subsampling_0")(sample, out):
            c_results.append(int(out[0]))

    for index, (expected_value, c_value) in enumerate(
        zip(expected.tolist(), c_results, strict=True)
    ):
        passed, reason = compare_values(int(expected_value), c_value)
        assert passed, f"index={index}: {reason}"
