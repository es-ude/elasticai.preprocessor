from pathlib import Path

import cocotb
import pytest
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge
from cocotb.utils import get_sim_time
from elasticai.creator.testing import CocotbTestFixture, eai_testbench

from elasticai.preprocessor import sequential


@cocotb.test()
@eai_testbench
async def sequential_verilog_tb(
    dut,
    bitwidth: int,
    size_input: int,
    size_output: int,
):
    period_clk = 2

    dut.CLK_SYS.value = 0
    dut.RSTN.value = 0
    dut.EN.value = 0
    dut.DO_CALC.value = 0
    dut.DATA_IN.value = 0

    # Start clock and make reset
    cocotb.start_soon(Clock(dut.CLK_SYS, period_clk, unit="ns").start())
    await ClockCycles(dut.CLK_SYS, 4)
    for idx in range(8):
        dut.RSTN.value = idx % 2
        await ClockCycles(dut.CLK_SYS, 2)
    dut.RSTN.value = 1

    await ClockCycles(dut.CLK_SYS, 2)
    assert dut.DATA_VALID.value == 1
    dut.EN.value = 1

    await ClockCycles(dut.CLK_SYS, 4)
    for idx in range(8):
        dut.DO_CALC.value = 1
        await ClockCycles(dut.CLK_SYS, 1)
        dut.DO_CALC.value = 0

        dt0 = get_sim_time(unit="ns")
        await RisingEdge(dut.DATA_VALID)
        dt1 = get_sim_time(unit="ns")
        num_clocks = int((dt1 - dt0) / period_clk)
        assert num_clocks == 1  # dut.SIZE_OUTPUT.value.to_unsigned() - 1
        await ClockCycles(dut.CLK_SYS, 4)


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth, size_input, size_output", [(8, 4, 2)])
def test_template(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
    size_input: int,
    size_output: int,
) -> None:
    path2template = Path(sequential.__file__).parent

    cocotb_test_fixture.set_top_module_name("PREPROCESSOR")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_dir(path=path2template, glob_pattern="design/sequential.v")
    cocotb_test_fixture.run(
        params={
            "BITWIDTH": bitwidth,
            "SIZE_INPUT": 4,
            "SIZE_OUTPUT": 2,
        },
        defines={},
    )
