import cocotb
import pytest

from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, ReadOnly, RisingEdge
import numpy as np

from elasticai.creator.testing import CocotbTestFixture, eai_testbench

from elasticai.preprocessor.translation.cocotb_tmp import temporary_directory
from elasticai.preprocessor.eventdetection import (
    EventDetection,
    SettingsEventDetection,
    TargetsEventDetection,
)


def build_testdata() -> tuple[list[int], list[int]]:
    data_in = [
        100,  # below THR_ON  -> off
        105,  # below THR_ON  -> off
        109,  # just below THR_ON -> off
        110,  # reaches THR_ON -> on
        100,  # between thresholds -> stays on
        90,   # exactly THR_OFF -> stays on
        89,   # below THR_OFF -> off
        105,  # between thresholds -> stays off
        110,  # reaches THR_ON again -> on
        100,  # between thresholds -> stays on
    ]

    check = [
        0,
        0,
        0,
        1,
        1,
        1,
        0,
        0,
        1,
        1,
    ]

    return data_in, check


@cocotb.test()
@eai_testbench
async def double_hyst_tb(
    dut,
    bitwidth: int,
    data_in: list[int],
    check: list[int]
):
    period_clk = 5

    THR_ON = 110
    THR_OFF = 90

    # Initialize signals
    dut.CLK_SYS.value = 0
    dut.RSTN.value = 0
    dut.EN.value = 0
    dut.DO_CALC.value = 0
    dut.DATA_IN.value = 0
    dut.THR_ON.value = THR_ON
    dut.THR_OFF.value = THR_OFF

    # Start clock and making reset
    cocotb.start_soon(
        Clock(
            dut.CLK_SYS,
            period_clk,
            unit="ns",
        ).start()
    )

    # Make reset
    await ClockCycles(dut.CLK_SYS, 4)

    # During reset the event output must be 0
    await ReadOnly()
    assert int(dut.IS_EVNT.value) == 0
    await FallingEdge(dut.CLK_SYS)
    dut.RSTN.value = 1
    await RisingEdge(dut.CLK_SYS)

    # Enable module
    await FallingEdge(dut.CLK_SYS)
    dut.EN.value = 1
    assert int(dut.DVALID.value) == 0

    # Apply test values
    for val, expected in zip(data_in, check):

        # Apply new input before the next rising edge
        await FallingEdge(dut.CLK_SYS)
        dut.DATA_IN.value = val
        dut.DO_CALC.value = 1

        # The module performs the comparison here
        await RisingEdge(dut.CLK_SYS)
        await ReadOnly()

        print(
            "DATA_IN =", val,
            "THR_ON =", THR_ON,
            "THR_OFF =", THR_OFF,
            "EXPECTED =", expected,
            "IS_EVNT =", int(dut.IS_EVNT.value),
            "DVALID =", int(dut.DVALID.value),
        )

        assert int(dut.IS_EVNT.value) == expected
        assert int(dut.DVALID.value) == 1

        # Finish calculation
        await FallingEdge(dut.CLK_SYS)
        dut.DO_CALC.value = 0
        await RisingEdge(dut.CLK_SYS)
        await ReadOnly()

        # With DO_CALC = 0 the event state must be held
        assert int(dut.IS_EVNT.value) == expected
        assert int(dut.DVALID.value) == 0

@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [8])
def test_template(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
):
    data_in, check = build_testdata()

    backup = cocotb_test_fixture.get_artifact_dir()
    with temporary_directory(backup):
            cocotb_test_fixture.write({"data_in": data_in, "check": check})
            cocotb_test_fixture.set_top_module_name("EVENTDETECTION_DOUBLE_HYST")    # Hier muss Name mit Verilog Modul übereinstimmen
            cocotb_test_fixture.clear_srcs()
            cocotb_test_fixture.add_srcs_from_package(
                "eventdetection",
                "verilog/eventdetection_double_hyst.v",
            )
    
            cocotb_test_fixture.run(
                params={
                    "BITWIDTH": bitwidth,
                },
                defines={},
            )

@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [8])
def test_build(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int, 
):
    dut = EventDetection(
        SettingsEventDetection(
            window_size=10,
            type=TargetsEventDetection.DoubleHyst,
            out_invert=False
        )
    )

    data_in, check = build_testdata()

    backup = cocotb_test_fixture.get_artifact_dir()
    with temporary_directory(backup) as tmpdir:
        build_dir = tmpdir / "verilog"

        dut.create_design(
            target="fpga",
            bitwidth=bitwidth,
            id="",
            path2save=build_dir,
            signed=False
        )

        cocotb_test_fixture.write({"data_in": data_in, "check": check})
        cocotb_test_fixture.set_top_module_name("EVENTDETECTION_DOUBLE_HYST")
        cocotb_test_fixture.clear_srcs()
        cocotb_test_fixture.add_srcs_from_dir(path=tmpdir, glob_pattern="verilog/*.v")
        cocotb_test_fixture.run(
            params={},
            defines={},
        )

@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [8])
def test_double_hyst_build_equal(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
):
    threshold = 100

    dut = EventDetection(
        SettingsEventDetection(
            window_size=10,
            type=TargetsEventDetection.DoubleHyst,
            out_invert=False
        )
    )

    data_in, _ = build_testdata()
    data_checked = dut.get_events(
        xin=np.asarray(data_in),
        threshold=np.full(len(data_in), threshold),
    ).astype(int).tolist()

    backup = cocotb_test_fixture.get_artifact_dir()
    with temporary_directory(backup) as tmpdir:
        build_dir = tmpdir / "verilog"

        dut.create_design(
            target="fpga",
            bitwidth=bitwidth,
            id="",
            path2save=build_dir,
            signed=False,
        )

        cocotb_test_fixture.write({"data_in": data_in, "check": data_checked,})
        cocotb_test_fixture.set_top_module_name("EVENTDETECTION_DOUBLE_HYST")
        cocotb_test_fixture.clear_srcs()
        cocotb_test_fixture.add_srcs_from_dir(path=tmpdir,glob_pattern="verilog/*.v")
        cocotb_test_fixture.run(
            params={},
            defines={},
        )
