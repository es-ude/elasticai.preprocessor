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


def _poly_names(take_first_order: bool) -> tuple[str, str]:
    """Return (file_stem, c_function_name) depending on order."""
    if take_first_order:
        return "downsampling_poly_one_0", "calc_do_poly_one_0"
    return "downsampling_poly_two_0", "calc_do_poly_two_0"


@pytest.mark.parametrize("take_first_order", [True, False], ids=["order_one", "order_two"])
@pytest.mark.parametrize("target", ["mcu", "pc"])
def test_create_design_generates_poly_c_files(
    tmp_path: Path, target: str, take_first_order: bool
) -> None:
    downsampler = DownSampling(SettingsDownSampling(sampling_rate=1000.0, dsr=4))
    downsampler.create_design(
        method=TargetsDownSampling.Polyphase,
        take_first_order=take_first_order,
        target=target,
        bitwidth=8,
        id="0",
        path2save=tmp_path,
        signed=True,
    )
    stem, _ = _poly_names(take_first_order)
    assert (tmp_path / f"{stem}.c").exists()
    assert (tmp_path / f"{stem}.h").exists()
    assert (tmp_path / "downsampling_poly_template.h").exists()


@pytest.mark.parametrize("take_first_order", [True, False], ids=["order_one", "order_two"])
def test_create_design_rejects_invalid_downsampling_ratio(
    tmp_path: Path, take_first_order: bool
) -> None:
    downsampler = DownSampling(SettingsDownSampling(sampling_rate=1000.0, dsr=0))
    with pytest.raises(ValueError, match="dsr must be >= 1"):
        downsampler.create_design(
            method=TargetsDownSampling.Polyphase,
            take_first_order=take_first_order,
            target="mcu",
            bitwidth=8,
            id="0",
            path2save=tmp_path,
        )


@pytest.mark.parametrize("take_first_order", [True, False], ids=["order_one", "order_two"])
def test_create_design_rejects_downsampling_ratio_not_bin(
    tmp_path: Path, take_first_order: bool
) -> None:
    downsampler = DownSampling(SettingsDownSampling(sampling_rate=1000.0, dsr=3))
    with pytest.raises(ValueError, match=r"dsr must be 2\^n"):
        downsampler.create_design(
            method=TargetsDownSampling.Polyphase,
            take_first_order=take_first_order,
            target="mcu",
            bitwidth=8,
            id="0",
            path2save=tmp_path,
        )


@pytest.mark.parametrize("take_first_order", [True, False], ids=["order_one", "order_two"])
@pytest.mark.parametrize("bitwidth,numpy_dtype,c_type", INTEGER_CONFIGS)
def test_generated_poly_c_matches_python_frame(
    tmp_path: Path,
    bitwidth: int,
    numpy_dtype: type[np.generic],
    c_type: str,
    take_first_order: bool,
) -> None:
    settings = SettingsDownSampling(sampling_rate=1000.0, dsr=4)
    downsampler = DownSampling(settings)
    output_dir = tmp_path / "src"
    downsampler.create_design(
        method=TargetsDownSampling.Polyphase,
        take_first_order=take_first_order,
        target="mcu",
        bitwidth=bitwidth,
        id="0",
        path2save=output_dir,
        signed=True,
    )

    stem, fn = _poly_names(take_first_order)
    adapter = tmp_path / "adapter.h"
    adapter.write_text(f"_Bool {fn}({c_type} data, {c_type} *out);\n")
    loader = CompileLoader(
        headers=str(adapter),
        sources=[str(output_dir / f"{stem}.c")],
        build_dir=str(tmp_path / "cffi-build"),
        module_name=f"downsampling_poly_equivalence_{uuid4().hex}",
    )
    loader.load()

    input_frame = np.arange(8, dtype=numpy_dtype)
    expected_float = downsampler.do_decimation_polyphase(input_frame.astype(float), take_first_order=take_first_order)
    expected = np.array([int(v) for v in expected_float], dtype=numpy_dtype)

    out = loader.ffi().new(f"{c_type} *")
    c_results = []
    for sample in input_frame.tolist():
        if loader.get(fn)(sample, out):
            c_results.append(int(out[0]))

    for index, (expected_value, c_value) in enumerate(
        zip(expected.tolist(), c_results, strict=True)
    ):
        passed, reason = compare_values(int(expected_value), c_value)
        assert passed, f"index={index}: {reason}"


@pytest.mark.parametrize("take_first_order", [True, False], ids=["order_one", "order_two"])
@pytest.mark.parametrize("bitwidth,numpy_dtype,c_type", INTEGER_CONFIGS)
def test_generated_poly_c_matches_python_sinewave(
    tmp_path: Path,
    bitwidth: int,
    numpy_dtype: type[np.generic],
    c_type: str,
    take_first_order: bool,
) -> None:
    settings = SettingsDownSampling(sampling_rate=1000.0, dsr=4)
    downsampler = DownSampling(settings)
    output_dir = tmp_path / "src"
    downsampler.create_design(
        method=TargetsDownSampling.Polyphase,
        take_first_order=take_first_order,
        target="mcu",
        bitwidth=bitwidth,
        id="0",
        path2save=output_dir,
        signed=True,
    )

    stem, fn = _poly_names(take_first_order)
    adapter = tmp_path / "adapter.h"
    adapter.write_text(f"_Bool {fn}({c_type} data, {c_type} *out);\n")
    loader = CompileLoader(
        headers=str(adapter),
        sources=[str(output_dir / f"{stem}.c")],
        build_dir=str(tmp_path / "cffi-build"),
        module_name=f"downsampling_poly_sinewave_{uuid4().hex}",
    )
    loader.load()

    amplitude = 5 if bitwidth == 8 else 10000
    t = np.arange(64) / settings.sampling_rate
    input_frame = (np.sin(2 * np.pi * 10 * t) * amplitude).astype(numpy_dtype)

    expected_float = downsampler.do_decimation_polyphase(input_frame.astype(float), take_first_order=take_first_order)
    expected = np.array([int(v) for v in expected_float], dtype=numpy_dtype)

    out = loader.ffi().new(f"{c_type} *")
    c_results = []
    for sample in input_frame.tolist():
        if loader.get(fn)(sample, out):
            c_results.append(int(out[0]))

    for index, (expected_value, c_value) in enumerate(
        zip(expected.tolist(), c_results, strict=True)
    ):
        passed, reason = compare_values(int(expected_value), c_value)
        assert passed, f"index={index}: {reason}"
