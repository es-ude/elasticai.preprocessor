from shutil import which
from uuid import uuid4

import numpy as np
import pytest
from elasticai.equichecker import CompileLoader, compare_values

from elasticai.preprocessor import get_path_to_project
from elasticai.preprocessor.thresholding import (
    SettingsThreshold,
    TargetsThreshold,
    Thresholding,
)
from elasticai.preprocessor.translation.cocotb_tmp import temporary_directory

pytestmark = pytest.mark.skipif(which("cc") is None, reason="requires a C compiler")

INTEGER_CONFIGS = [
    pytest.param(8, np.int8, "signed char", id="int8"),
    pytest.param(32, np.int32, "signed int", id="int32"),
]

THRESHOLDING_CONFIGS = [
    pytest.param(1000.0, 10e-3, TargetsThreshold.Constant, "thresholding_constant", id="method_constant"),
    pytest.param(1000.0, 10e-3, TargetsThreshold.Welford, "thresholding_welford", id="method_welford"),
    pytest.param(1000.0, 10e-3, TargetsThreshold.MovingAverage, "thresholding_mavg", id="method_mavg"),
    pytest.param(
        1000.0,
        10e-3,
        TargetsThreshold.MovingAverageAbsolute,
        "thresholding_mavg_abs",
        id="method_mavg_abs",
    ),
    pytest.param(
        512.0, 0.015625, TargetsThreshold.MovingAverage, "thresholding_mavg_pow2", id="method_mavg_pow2"
    ),
    pytest.param(
        512.0,
        0.015625,
        TargetsThreshold.MovingAverageAbsolute,
        "thresholding_mavg_pow2_abs",
        id="method_mavg_pow2_abs",
    ),  # window_steps = 8
]


@pytest.mark.parametrize("target", ["mcu", "pc"])
@pytest.mark.parametrize("bitwidth,numpy_dtype,c_type", INTEGER_CONFIGS)
@pytest.mark.parametrize("is_signed", [True])
@pytest.mark.parametrize("sampling_rate,window_sec,method,c_name", THRESHOLDING_CONFIGS)
def test_generated_thresholding_c_matches_python_frame(
    target: str,
    bitwidth: int,
    is_signed: bool,
    numpy_dtype: type(np.generic),
    c_type: str,
    method: TargetsThreshold,
    sampling_rate: float,
    window_sec: float,
    c_name: str,
) -> None:

    settings = SettingsThreshold(
        method=method,
        sampling_rate=sampling_rate,
        window_sec=window_sec,
        thr_val=0.0,
        do_quant=True,
    )
    thresholder = Thresholding(settings)
    data = np.array(
        [0, 43, 81, 110, 125, 124, 109, 80, 42, -1, -44, -82, -110, -125, -124, -109, -80, -41, 2, 45],
        dtype=numpy_dtype,
    )

    backup = get_path_to_project("build_test") / f"build_{c_name}"
    with temporary_directory(backup) as tmpdir:
        output_dir = tmpdir / "src"
        thresholder.create_design(
            id="0",
            target=target,
            bitwidth=bitwidth,
            signed=is_signed,
            data=data,
            path2save=output_dir,
        )

        adapter = tmpdir / "adapter.h"
        adapter.write_text(f"_Bool calc_{c_name}_0({c_type} data, {c_type} *out);\n")
        loader = CompileLoader(
            headers=str(adapter),
            sources=[str(output_dir / f"{c_name}_0.c")],
            build_dir=str(tmpdir / "cffi-build"),
            module_name=f"thresholding_equivalence_{uuid4().hex}",
        )
        loader.load()

        expected_all = thresholder.get_threshold(data).astype(numpy_dtype)
        ffi = loader.ffi()
        out = ffi.new(f"{c_type}[1]")
        c_results = []
        valid_indices = []
        for idx, sample in enumerate(data.tolist()):
            if loader.get(f"calc_{c_name}_0")(sample, out):
                c_results.append(int(out[0]))
                valid_indices.append(idx)

        expected = expected_all[valid_indices]
        for index, (expected_value, c_value) in enumerate(zip(expected.tolist(), c_results, strict=True)):
            passed, reason = compare_values(int(expected_value), c_value)
            assert passed, f"index={index}: {reason}"
