from pathlib import Path
from shutil import which
from uuid import uuid4

import numpy as np
import pytest
from elasticai.equichecker import CompileLoader, compare_values

from elasticai.preprocessor.thresholding import SettingsThreshold, Thresholding

pytestmark = pytest.mark.skipif(which("cc") is None, reason="requires a C compiler")

INTEGER_CONFIGS = [
    pytest.param(8, np.int8, "signed char", id="int8"),
    pytest.param(32, np.int32, "signed int", id="int32"),
]

THRESHOLDING_CONFIGS = [
    pytest.param(1000.0, 10e-3, "const", "thresholding_constant", "thresholding_constant", id="method_constant"),
    pytest.param(1000.0, 10e-3, "welford", "thresholding_welford", "thresholding_welford", id="method_welford"),
    pytest.param(1000.0, 10e-3, "mavg", "thresholding_mavg", "moving_average", id="method_mavg"),
    pytest.param(512.0, 0.015625, "mavg", "thresholding_mavg_pow2", "moving_average_pow2", id="method_mavg_pow2"),  # window_steps = int(0.015625 * 512) = 8 = 2^3
    pytest.param(1000.0, 10e-3, "mavg_abs", "thresholding_mavg_abs", "moving_average_abs", id="method_mavg_abs"),
    pytest.param(512.0, 0.015625, "mavg_abs", "thresholding_mavg_pow2_abs", "moving_average_pow2_abs", id="method_mavg_pow2_abs"),  # window_steps = 8 = 2^3
]

@pytest.mark.parametrize("target", ["mcu", "pc"])
@pytest.mark.parametrize("sampling_rate,window_sec,method,c_name,c_func", THRESHOLDING_CONFIGS)
def test_create_design_generates_thresholding_c_files(
    tmp_path: Path, 
    target: str, 
    method: str, 
    sampling_rate: float, 
    window_sec: float, 
    c_name: str,
    c_func: str,
) -> None:
    thresholder = Thresholding(
        SettingsThreshold(
            method=method,
            sampling_rate=sampling_rate,
            window_sec=window_sec,
            do_quant=False
        )
    )
    thresholder.create_design(
        method=method,
        gain= 1.0,
        window_size=thresholder._settings.window_steps,
        data=np.array([10,20,40]),
        const_threshold=20,
        id="0",
        target=target,
        bitwidth=8,
        signed=True,
        path2save=tmp_path
    )
    assert (tmp_path / f"{c_name}_0.c").exists()
    assert (tmp_path / f"{c_name}_0.h").exists()
    assert (tmp_path / f"{c_name}_template.h").exists()

@pytest.mark.parametrize("target", ["mcu", "pc"])
@pytest.mark.parametrize("bitwidth,numpy_dtype,c_type", INTEGER_CONFIGS)
@pytest.mark.parametrize("sampling_rate,window_sec,method,c_name,c_func", THRESHOLDING_CONFIGS)
@pytest.mark.parametrize("gain", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
def test_generated_thresholding_c_matches_python_frame(
    tmp_path: Path,
    target: str,
    gain: float,
    bitwidth: int,
    numpy_dtype: type(np.generic),
    c_type: str,
    method: str,
    sampling_rate: float,
    window_sec: float,
    c_name: str,
    c_func: str,
) -> None:
    settings = SettingsThreshold(
            method=method,
            sampling_rate=sampling_rate,
            window_sec=window_sec,
            do_quant=True,
    )
    thresholder = Thresholding(settings)
    data = np.array([0, 43, 81, 110, 125, 124, 109, 80, 42, -1, -44, -82, -110, -125, -124, -109, -80, -41, 2, 45], dtype=numpy_dtype)
    output_dir = tmp_path / "src"
    thresholder.create_design(
        method=method,
        target=target,
        gain=gain,
        window_size=thresholder._settings.window_steps,
        data=data,
        const_threshold=0,
        id="0",
        bitwidth=bitwidth,
        signed=True,
        path2save=output_dir,
    )

    adapter = tmp_path / "adapter.h"
    adapter.write_text(f"_Bool calc_{c_func}_0({c_type} data, {c_type} *out);\n")
    loader = CompileLoader(
        headers=str(adapter),
        sources=[str(output_dir / f"{c_name}_0.c")],
        build_dir = str(tmp_path / "cffi-build"),
        module_name=f"thresholding_equivalence_{uuid4().hex}",
    )
    loader.load()

    thr_kwargs = {"thr_val": 0} if method == "const" else {}
    expected_all = thresholder.get_threshold(data, gain, **thr_kwargs).astype(numpy_dtype)
    ffi = loader.ffi()
    out = ffi.new(f"{c_type}[1]")
    c_results = []
    valid_indices = []
    for idx, sample in enumerate(data.tolist()):
        if loader.get(f"calc_{c_func}_0")(sample, out):
            c_results.append(int(out[0]))
            valid_indices.append(idx)

    expected = expected_all[valid_indices]
    for index, (expected_value, c_value) in enumerate(zip(expected.tolist(), c_results, strict=True)):
        passed, reason = compare_values(int(expected_value), c_value)
        assert passed, f"index={index}: {reason}"
