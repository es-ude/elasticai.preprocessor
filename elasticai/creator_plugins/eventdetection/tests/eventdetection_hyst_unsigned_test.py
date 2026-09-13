import cocotb
import numpy as np
import pytest

from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, ReadOnly, RisingEdge

from elasticai.creator.testing import CocotbTestFixture, eai_testbench

from elasticai.preprocessor.translation.cocotb_tmp import temporary_directory
from elasticai.preprocessor.eventdetection import (
    EventDetection,
    SettingsEventDetection,
    TargetsEventDetection,
)


def build_testdata(
    threshold: int,
    window_size: int,
) -> tuple[list[int], list[int]]:
    data_in = [
        threshold,
        threshold + window_size - 1,
        threshold + window_size,
        threshold,
        threshold - window_size,
        threshold - window_size - 1,
        threshold,
        threshold + window_size - 1,
        threshold + window_size,
        threshold,
    ]

    check = [
        0,
        0,
        1,
        1,
        1,
        0,
        0,
        0,
        1,
        1,
    ]

    return data_in, check


@cocotb.test()
@eai_testbench
async def hysteresis_tb(
    dut,
    bitwidth: int,
    data_in: list[int],
    check: list[int],
    thr_on: int,
    thr_off: int,
):
    period_clk = 5

    # Initialize signals
    dut.CLK_SYS.value = 0
    dut.RSTN.value = 0
    dut.EN.value = 0
    dut.DO_CALC.value = 0
    dut.DATA_IN.value = 0
    dut.THR_ON.value = thr_on
    dut.THR_OFF.value = thr_off

    # Start clock
    cocotb.start_soon(
        Clock(
            dut.CLK_SYS,
            period_clk,
            unit="ns",
        ).start()
    )

    # Reset
    await ClockCycles(dut.CLK_SYS, 4)

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

        await FallingEdge(dut.CLK_SYS)
        dut.DATA_IN.value = val
        dut.DO_CALC.value = 1

        await RisingEdge(dut.CLK_SYS)
        await ReadOnly()

        print(
            "DATA_IN =", val,
            "THR_ON =", thr_on,
            "THR_OFF =", thr_off,
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

        # Result must be held
        assert int(dut.IS_EVNT.value) == expected
        assert int(dut.DVALID.value) == 0

@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [8])
def test_template(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
):
    threshold = 100
    window_size = 10

    thr_on = threshold + window_size
    thr_off = threshold - window_size

    data_in, check = build_testdata(
        threshold=threshold,
        window_size=window_size,
    )

    backup = cocotb_test_fixture.get_artifact_dir()

    with temporary_directory(backup):
            cocotb_test_fixture.write(
            {
                "data_in": data_in,
                "check": check,
                "thr_on": thr_on,
                "thr_off": thr_off,
            }
        )
            cocotb_test_fixture.set_top_module_name("UNSIGNED_HYSTERESIS") 
            cocotb_test_fixture.clear_srcs()
            cocotb_test_fixture.add_srcs_from_package(
                "eventdetection",
                "verilog/eventdetection_hyst_unsigned.v",
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
    threshold = 100
    window_size = 10

    thr_on = threshold + window_size
    thr_off = threshold - window_size

    data_in, check = build_testdata(
        threshold=threshold,
        window_size=window_size,
    )

    dut = EventDetection(
        SettingsEventDetection(
            window_size=window_size,
            type=TargetsEventDetection.DoubleHyst,
            out_invert=False,
        )
    )

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

        cocotb_test_fixture.write(
            {
                "data_in": data_in,
                "check": check,
                "thr_on": thr_on,
                "thr_off": thr_off,
            }
        )
        cocotb_test_fixture.set_top_module_name("EVENTDETECTION_HYST_UNSIGNED")
        cocotb_test_fixture.clear_srcs()
        cocotb_test_fixture.add_srcs_from_dir(path=tmpdir, glob_pattern="verilog/*.v")
        cocotb_test_fixture.run(
            params={},
            defines={},
        )

@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [8])
def test_pos_hyst_build_equal(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
):
    threshold = 100
    window_size = 10

    dut = EventDetection(
        SettingsEventDetection(
            window_size=window_size,
            type=TargetsEventDetection.PosHyst,
            out_invert=False,
        )
    )

    data_in, _ = build_testdata(threshold=threshold, window_size=window_size,)

    data_checked = dut.get_events(
        xin=np.asarray(data_in),
        threshold=np.full(len(data_in),threshold),
    ).astype(int).tolist()

    thr_on, thr_off = dut._type_hysteresis(
        threshold
    )

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

        cocotb_test_fixture.write(
            {
                "data_in": data_in,
                "check": data_checked,
                "thr_on": thr_on,
                "thr_off": thr_off,
            }
        )
        cocotb_test_fixture.set_top_module_name("EVENTDETECTION_HYST_UNSIGNED")
        cocotb_test_fixture.clear_srcs()
        cocotb_test_fixture.add_srcs_from_dir(path=tmpdir,glob_pattern="verilog/*.v")
        cocotb_test_fixture.run(
            params={},
            defines={},
        )

@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [8])
def test_neg_hyst_build_equal(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
):
    threshold = 100
    window_size = 10

    dut = EventDetection(
        SettingsEventDetection(
            window_size=window_size,
            type=TargetsEventDetection.NegHyst,
            out_invert=False,
        )
    )

    data_in, _ = build_testdata(threshold=threshold, window_size=window_size,)

    data_checked = dut.get_events(
        xin=np.asarray(data_in),
        threshold=np.full(len(data_in),threshold),
    ).astype(int).tolist()

    thr_on, thr_off = dut._type_hysteresis(
        threshold
    )

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

        cocotb_test_fixture.write(
            {
                "data_in": data_in,
                "check": data_checked,
                "thr_on": thr_on,
                "thr_off": thr_off,
            }
        )
        cocotb_test_fixture.set_top_module_name("EVENTDETECTION_HYST_UNSIGNED")
        cocotb_test_fixture.clear_srcs()
        cocotb_test_fixture.add_srcs_from_dir(path=tmpdir,glob_pattern="verilog/*.v")
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
    window_size = 10

    dut = EventDetection(
        SettingsEventDetection(
            window_size=window_size,
            type=TargetsEventDetection.DoubleHyst,
            out_invert=False,
        )
    )

    data_in, _ = build_testdata(threshold=threshold, window_size=window_size,)

    data_checked = dut.get_events(
        xin=np.asarray(data_in),
        threshold=np.full(len(data_in),threshold),
    ).astype(int).tolist()

    thr_on, thr_off = dut._type_hysteresis(
        threshold
    )

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

        cocotb_test_fixture.write(
            {
                "data_in": data_in,
                "check": data_checked,
                "thr_on": thr_on,
                "thr_off": thr_off,
            }
        )
        cocotb_test_fixture.set_top_module_name("EVENTDETECTION_HYST_UNSIGNED")
        cocotb_test_fixture.clear_srcs()
        cocotb_test_fixture.add_srcs_from_dir(path=tmpdir,glob_pattern="verilog/*.v")
        cocotb_test_fixture.run(
            params={},
            defines={},
        )
