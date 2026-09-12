import cocotb
import numpy as np
import pytest
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles
from elasticai.creator.arithmetic import int_arithmetic
from elasticai.creator.testing import CocotbTestFixture, eai_testbench
from elasticai.creator_plugins.mac import load_and_plugin

from elasticai.preprocessor.eventdetection import (
    EventDetection,
    SettingsEventDetection,
    TargetsEventDetection,
)
from elasticai.preprocessor.translation.cocotb_tmp import temporary_directory


def get_test_signal() -> tuple[list[int], list[int], list[int]]:
    data_in = [-8, -4, 0, 3, 4, 5, 7]
    threshold = [4, 4, 4, 4, 4, 4, 4]
    check = [0, 0, 0, 0, 1, 1, 1]
    return data_in, threshold, check


@cocotb.test()
@eai_testbench
async def signed_threshold_test(
    dut,
    bitwidth: int,
    data_in: list[int],
    threshold: list[int],
    check: list[int],
):
    period_clk = 5

    dut.CLK_SYS.value = 0
    dut.RSTN.value = 0
    dut.EN.value = 0
    dut.DO_CALC.value = 0
    dut.DATA_IN.value = 0
    dut.THR.value = 0

    cocotb.start_soon(Clock(dut.CLK_SYS, period_clk, unit="ns").start())
    await ClockCycles(dut.CLK_SYS, 4)
    for idx in range(4):
        dut.RSTN.value = idx % 2
        await ClockCycles(dut.CLK_SYS, 3)
    dut.RSTN.value = 1

    dut.EN.value = 1
    assert dut.DVALID.value == 0
    await ClockCycles(dut.CLK_SYS, 2)

    assert len(data_in) == len(check)
    for val, thr, expected in zip(data_in, threshold, check):
        dut.DATA_IN.value = val
        dut.THR.value = thr
        dut.DO_CALC.value = 1
        await ClockCycles(dut.CLK_SYS, 2)

        assert int(dut.IS_EVNT.value) == expected
        assert int(dut.DVALID.value) == 1

        # Calculation finished
        dut.DO_CALC.value = 0
        await ClockCycles(dut.CLK_SYS, 2)

        # Result should be held
        assert int(dut.IS_EVNT.value) == expected
        assert int(dut.DVALID.value) == 0


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [4])
def test_eventdetection_signed_build(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
):
    backup = cocotb_test_fixture.get_artifact_dir()
    with temporary_directory(backup) as tmpdir:
        build_dir = tmpdir / "verilog"

        load_and_plugin(
            type="eventdetector_normal_signed",
            id="0",
            params={"BITWIDTH": bitwidth, "OUT_INVERT": 0},
            packages=["eventdetection"],
            path2save=build_dir,
        )

        data_in, threshold, check = get_test_signal()
        cocotb_test_fixture.write(
            {
                "data_in": data_in,
                "threshold": threshold,
                "check": check,
            }
        )
        cocotb_test_fixture.set_top_module_name("EVENTDETECTOR_NORMAL_SIGNED_0")
        cocotb_test_fixture.clear_srcs()
        cocotb_test_fixture.add_srcs_from_dir(path=tmpdir, glob_pattern="verilog/*.v")
        cocotb_test_fixture.run(
            params={},
            defines={},
        )


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [4])
def test_eventdetection_sub_build(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
):
    backup = cocotb_test_fixture.get_artifact_dir()
    with temporary_directory(backup) as tmpdir:
        build_dir = tmpdir / "verilog"

        load_and_plugin(
            type="eventdetector_sub_signed",
            id="0",
            params={"BITWIDTH": bitwidth, "OUT_INVERT": 0},
            packages=["eventdetection"],
            path2save=build_dir,
        )

        data_in, threshold, check = get_test_signal()
        cocotb_test_fixture.write(
            {
                "data_in": data_in,
                "threshold": threshold,
                "check": check,
            }
        )

        cocotb_test_fixture.set_top_module_name("EVENTDETECTOR_SUB_SIGNED_0")
        cocotb_test_fixture.clear_srcs()
        cocotb_test_fixture.add_srcs_from_dir(path=tmpdir, glob_pattern="verilog/*.v")
        cocotb_test_fixture.run(
            params={},
            defines={},
        )


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [4])
def test_eventdetection_signed_sub_codesign(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
) -> None:
    sets = SettingsEventDetection(window_size=10, type=TargetsEventDetection.Normal, out_invert=True)
    dut = EventDetection(settings=sets)

    arith = int_arithmetic(total_bits=bitwidth, signed=True)
    data_in = np.random.randint(
        low=arith.minimum_as_integer, high=arith.maximum_as_integer + 1, size=(100,)
    )
    threshold = np.zeros_like(data_in) + int((arith.maximum_as_integer - arith.minimum_as_integer) / 2)
    expected = dut.get_events(xin=data_in, threshold=threshold)

    backup = cocotb_test_fixture.get_artifact_dir()
    with temporary_directory(backup) as tmpdir:
        build_dir = tmpdir / "verilog"

        dut.create_design(target="fpga", bitwidth=bitwidth, id="1", path2save=build_dir, signed=True)
        cocotb_test_fixture.write(
            {
                "data_in": data_in.tolist(),
                "threshold": threshold.tolist(),
                "check": expected.tolist(),
            }
        )

        cocotb_test_fixture.set_top_module_name("EVENTDETECTOR_SUB_SIGNED_1")
        cocotb_test_fixture.clear_srcs()
        cocotb_test_fixture.add_srcs_from_dir(path=tmpdir, glob_pattern="verilog/*.v")
        cocotb_test_fixture.run(
            params={},
            defines={},
        )
