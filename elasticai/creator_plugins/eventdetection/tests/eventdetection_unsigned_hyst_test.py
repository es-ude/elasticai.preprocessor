import cocotb
import numpy as np
import pytest
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge
from elasticai.creator.testing import CocotbTestFixture, eai_testbench

from elasticai.preprocessor.eventdetection import (
    EventDetection,
    SettingsEventDetection,
    TargetsEventDetection,
)
from elasticai.preprocessor.translation.cocotb_tmp import temporary_directory


def build_testdata(
    threshold: int,
    window_size: int,
) -> tuple[list[int], list[int], list[int]]:
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

    thr = [threshold for _ in data_in]

    return data_in, thr, check


@cocotb.test()
@eai_testbench
async def hysteresis_tb(
    dut,
    bitwidth: int,
    data_in: list[int],
    threshold: list[int],
    check: list[int],
):
    period_clk = 5

    # Initialize signals
    dut.CLK_SYS.value = 0
    dut.RSTN.value = 0
    dut.EN.value = 0
    dut.DO_CALC.value = 0
    dut.DATA_IN.value = 0
    dut.THR.value = 0

    # Start clock
    cocotb.start_soon(
        Clock(
            dut.CLK_SYS,
            period_clk,
            unit="ns",
        ).start()
    )
    await RisingEdge(dut.CLK_SYS)
    assert int(dut.IS_EVNT.value) == 0
    assert int(dut.DVALID.value) == 0

    for idx in range(4):
        dut.RSTN.value = idx % 2
        await ClockCycles(dut.CLK_SYS, 2)
    assert int(dut.IS_EVNT.value) == int(dut.OUT_INVERT.value)
    dut.EN.value = 1

    # Apply test values
    for val, thr, expected in zip(data_in, threshold, check):
        dut.DATA_IN.value = val
        dut.THR.value = thr
        dut.DO_CALC.value = 1

        await RisingEdge(dut.CLK_SYS)
        assert dut.DVALID.value == 0
        dut.DO_CALC.value = 0
        await ClockCycles(dut.CLK_SYS, 2)
        assert dut.DVALID.value == 1
        assert int(dut.IS_EVNT.value) == expected
        if not int(dut.IS_EVNT.value) == expected:
            print(
                "DATA_IN =",
                val,
                "EXPECTED =",
                expected,
                "IS_EVNT =",
                int(dut.IS_EVNT.value),
            )


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [8])
def test_template(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
):
    threshold = 100
    window_size = 10

    data_in, thr, check = build_testdata(
        threshold=threshold,
        window_size=window_size,
    )

    backup = cocotb_test_fixture.get_artifact_dir()
    with temporary_directory(backup):
        cocotb_test_fixture.write(
            {
                "data_in": data_in,
                "threshold": thr,
                "check": check,
            }
        )
        cocotb_test_fixture.set_top_module_name("EVENTDETECTOR_HYSTERESE_UNSIGNED")
        cocotb_test_fixture.clear_srcs()
        cocotb_test_fixture.add_srcs_from_package(
            "eventdetection",
            "verilog/eventdetector_hyst_unsigned.v",
        )
        cocotb_test_fixture.run(
            params={
                "BITWIDTH": bitwidth,
                "THR_ON": window_size,
                "THR_OFF": window_size,
                "OUT_INVERT": 0,
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

    data_in, thr, check = build_testdata(
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
                "threshold": thr,
                "check": check,
            }
        )
        cocotb_test_fixture.set_top_module_name("EVENTDETECTOR_HYST_UNSIGNED")
        cocotb_test_fixture.clear_srcs()
        cocotb_test_fixture.add_srcs_from_dir(path=tmpdir, glob_pattern="verilog/*.v")
        cocotb_test_fixture.run(
            params={},
            defines={},
        )


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [6, 8, 12])
def test_equal(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
) -> None:
    threshold = 2 ** (bitwidth - 1)
    window_size = 10

    for vtype in [TargetsEventDetection.PosHyst]:
        dut = EventDetection(
            SettingsEventDetection(
                window_size=window_size,
                type=vtype,
                out_invert=True,
            )
        )
        data_in, thr, _ = build_testdata(
            threshold=threshold,
            window_size=window_size,
        )
        data_checked = (
            dut.get_events(
                xin=np.asarray(data_in),
                threshold=np.asarray(thr),
            )
            .astype(int)
            .tolist()
        )

        backup = cocotb_test_fixture.get_artifact_dir()
        with temporary_directory(backup) as tmpdir:
            build_dir = tmpdir / "verilog"

            dut.create_design(
                target="fpga",
                bitwidth=bitwidth,
                id=f"{vtype.value}",
                path2save=build_dir,
                signed=False,
            )

            cocotb_test_fixture.write(
                {
                    "data_in": data_in,
                    "threshold": thr,
                    "check": data_checked,
                }
            )
            cocotb_test_fixture.set_top_module_name(f"EVENTDETECTOR_HYST_UNSIGNED_{vtype.value.upper()}")
            cocotb_test_fixture.clear_srcs()
            cocotb_test_fixture.add_srcs_from_dir(path=tmpdir, glob_pattern="verilog/*.v")
            cocotb_test_fixture.run(
                params={},
                defines={},
            )
