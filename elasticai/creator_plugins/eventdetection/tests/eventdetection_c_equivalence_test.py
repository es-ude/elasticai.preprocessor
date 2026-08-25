from pathlib import Path
from shutil import which
from uuid import uuid4

import numpy as np
import pytest
from elasticai.equichecker import CompileLoader, compare_values

from elasticai.preprocessor.eventdetection import EventDetection, SettingsEventDetection

pytestmark = pytest.mark.skipif(which("cc") is None, reason="requires a C compiler")

INTEGER_CONFIGS = [
    pytest.param(8, np.int8, "signed char", id="int8"),
    pytest.param(32, np.int32, "signed int", id="int32"),
]

HYSTERESIS_TYPE_CONFIGS = [
    pytest.param("eventdetection_normal", "normal", id="no hysteresis"),
    pytest.param("eventdetection_pos_hyst", "pos_hyst", id="positive hysteresis"),
    pytest.param("eventdetection_neg_hyst", "neg_hyst", id="negative hysteresis"),
    pytest.param("eventdetection_double_hyst", "double_hyst", id="double hysteresis"),
]


@pytest.mark.parametrize("target", ["mcu", "pc"])
def test_create_design_generates_eventdetection_c_files(tmp_path: Path, target: str) -> None:
    eventdetector = EventDetection(
        SettingsEventDetection(hysteresis=0.25, hysteresis_type="eventdetection_double_hyst")
    )
    eventdetector.create_design(
        target=target,
        bitwidth=8,
        id="0",
        path2save=tmp_path,
        signed=True,
        hysteresis=0.25,
        hysteresis_type="eventdetection_double_hyst",
    )
    assert (tmp_path / "eventdetection_0.c").exists()
    assert (tmp_path / "eventdetection_0.h").exists()
    assert (tmp_path / "eventdetection_template.h").exists()


@pytest.mark.parametrize("bitwidth,numpy_dtype,c_type", INTEGER_CONFIGS)
@pytest.mark.parametrize("c_hysteresis,py_hysteresis", HYSTERESIS_TYPE_CONFIGS)
def test_generated_eventdetection_c_matches_python_frame(
    tmp_path: Path,
    bitwidth: int,
    numpy_dtype: type(np.generic),
    c_type: str,
    c_hysteresis: str,
    py_hysteresis: str,
) -> None:
    settings = SettingsEventDetection(hysteresis=0.25, hysteresis_type=py_hysteresis)
    eventdetector = EventDetection(settings)
    output_dir = tmp_path / "src"
    eventdetector.create_design(
        target="mcu",
        bitwidth=bitwidth,
        id="0",
        path2save=output_dir,
        hysteresis=0.25,
        hysteresis_type=c_hysteresis,
        signed=True,
    )

    adapter = tmp_path / "adapter.h"
    adapter.write_text(f"_Bool is_event_0({c_type} data, {c_type} threshold);\n")
    loader = CompileLoader(
        headers=str(adapter),
        sources=[str(output_dir / "eventdetection_0.c")],
        build_dir=str(tmp_path / "cffi-build"),
        module_name=f"eventdetection_equivalence_{uuid4().hex}",
    )
    loader.load()

    input_frame = np.array(
        [90, 100, 125, 100, 99, 75, 74, 100, 35, 40, 52, 60, 65, 65, 50, 100, 35, 90], dtype=numpy_dtype
    )
    thresholds_frame = np.array(
        [100, 100, 100, 100, 100, 100, 100, 100, 52, 52, 52, 52, 52, 80, 80, 80, 100, 80],
        dtype=numpy_dtype,
    )
    expected = eventdetector.detect_event(input_frame, thresholds_frame).astype(numpy_dtype)

    c_results = []
    for idx, sample in enumerate(input_frame.tolist()):
        c_results.append(loader.get("is_event_0")(sample, thresholds_frame[idx]))

    for index, (expected_value, c_value) in enumerate(zip(expected.tolist(), c_results, strict=True)):
        passed, reason = compare_values(expected_value, c_value)
        assert passed, f"index={index}: {reason}"
