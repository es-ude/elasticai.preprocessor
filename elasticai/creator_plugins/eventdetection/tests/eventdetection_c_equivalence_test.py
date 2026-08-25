from pathlib import Path
from shutil import which
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
import pytest
from elasticai.creator.arithmetic import int_arithmetic
from elasticai.equichecker import CompileLoader, compare_values

from elasticai.preprocessor import get_path_to_project
from elasticai.preprocessor.eventdetection import (
    EventDetection,
    SettingsEventDetection,
    TargetsEventDetection,
)
from elasticai.preprocessor.translation.cocotb_tmp import temporary_directory


def plot_comparator_output(
    input_frame: np.ndarray, thresholds_frame: np.ndarray, output_c: np.ndarray, output_np: np.ndarray
) -> None:
    fig, ax1 = plt.subplots()

    ax1.plot(input_frame, color="r", marker=".", label="input")
    ax1.plot(thresholds_frame, color="b", marker=".", label="threshold")
    ax1.set_ylabel("input / threshold")

    ax2 = ax1.twinx()
    ax2.plot(output_c, color="g", marker=".", label="output (C)")
    ax2.plot(output_np, color="m", marker=".", label="output (NP)")
    ax2.set_ylabel("output")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.tight_layout()
    plt.show()


pytestmark = pytest.mark.skipif(which("cc") is None, reason="requires a C compiler")

INTEGER_CONFIGS = [
    pytest.param(8, np.int8, "signed char", id="int8"),
    pytest.param(32, np.int32, "signed int", id="int32"),
]


HYSTERESIS_TYPE_CONFIGS = [
    pytest.param("normal", id="no hysteresis"),
    pytest.param("pos_hyst", id="positive hysteresis"),
    pytest.param("neg_hyst", id="negative hysteresis"),
    pytest.param("double_hyst", id="double hysteresis"),
]


@pytest.mark.parametrize("target", ["mcu", "pc"])
def test_create_design_generates_eventdetection_c_files(tmp_path: Path, target: str) -> None:
    eventdetector = EventDetection(
        SettingsEventDetection(
            window_size=10,
            type=TargetsEventDetection.DoubleHyst,
            out_invert=False,
        )
    )
    eventdetector.create_design(
        target=target,
        bitwidth=8,
        id="0",
        path2save=tmp_path,
        signed=True,
    )
    assert (tmp_path / "eventdetection_0.c").exists()
    assert (tmp_path / "eventdetection_0.h").exists()
    assert (tmp_path / "eventdetection_template.h").exists()


@pytest.mark.parametrize("bitwidth,numpy_dtype,c_type", INTEGER_CONFIGS)
@pytest.mark.parametrize("c_hysteresis", HYSTERESIS_TYPE_CONFIGS)
@pytest.mark.parametrize("is_signed", [True])
@pytest.mark.parametrize("out_invert", [False, True])
def test_generated_eventdetection_c_matches_python_frame(
    tmp_path: Path,
    bitwidth: int,
    numpy_dtype: type(np.generic),
    c_type: str,
    c_hysteresis: str,
    is_signed: bool,
    out_invert: bool,
) -> None:
    block_plot = False
    settings = SettingsEventDetection(
        window_size=10, type=TargetsEventDetection(c_hysteresis), out_invert=out_invert
    )
    eventdetector = EventDetection(settings)

    backup = get_path_to_project("build_test") / f"build_{c_hysteresis}"
    with temporary_directory(backup) as tmpdir:
        output_dir = tmpdir / "src"
        eventdetector.create_design(
            target="mcu",
            bitwidth=bitwidth,
            id="0",
            path2save=output_dir,
            signed=is_signed,
        )

        adapter = tmpdir / "adapter.h"
        adapter.write_text(f"_Bool is_event_0({c_type} data, {c_type} threshold);\n")
        loader = CompileLoader(
            headers=str(adapter),
            sources=[str(output_dir / "eventdetection_0.c")],
            build_dir=str(tmpdir / "cffi-build"),
            module_name=f"eventdetection_equivalence_{uuid4().hex}",
        )
        loader.load()

        arith = int_arithmetic(total_bits=bitwidth, signed=is_signed)
        input_frame = np.linspace(
            start=arith.minimum_as_integer,
            stop=arith.maximum_as_integer,
            num=2**8,
            dtype=numpy_dtype,
        )
        input_frame = np.concat((input_frame, input_frame[::-1]), axis=-1)
        thresholds_frame = np.random.randint(
            low=arith.minimum_as_integer + bitwidth, high=arith.maximum_as_integer - bitwidth
        ) + np.zeros_like(input_frame)
        expected = eventdetector.get_events(input_frame, thresholds_frame).astype(numpy_dtype)

        c_results = []
        for sample, thr in zip(input_frame.tolist(), thresholds_frame.tolist()):
            c_results.append(loader.get("is_event_0")(sample, thr))

        if block_plot:
            plot_comparator_output(
                input_frame=input_frame,
                thresholds_frame=thresholds_frame,
                output_c=np.array(c_results),
                output_np=expected,
            )

        for index, (expected_value, c_value) in enumerate(zip(expected.tolist(), c_results, strict=True)):
            passed, reason = compare_values(expected_value, c_value)
            assert passed, f"index={index}: {reason}"
