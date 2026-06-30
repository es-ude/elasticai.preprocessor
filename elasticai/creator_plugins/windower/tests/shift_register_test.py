import cocotb
import numpy as np
import pytest
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge
from elasticai.creator.testing import CocotbTestFixture, eai_testbench


@cocotb.test()
@eai_testbench
async def shifting_data(dut, bitwidth: int, elements: int):
    period_clk = 5

    data_in_array = [np.random.randint(low=0, high=2**bitwidth - 1) for _ in range(elements)]
    data_in_array = [val if val >= 0 else 0 for val in data_in_array]
    data_in_array = [2**bitwidth - 1 if val >= 2**bitwidth - 1 else val for val in data_in_array]
    print(data_in_array)

    dut.CLK_SYS.value = 0
    dut.RSTN.value = 0
    dut.EN.value = 0
    dut.DO_SHIFT.value = 0
    dut.DATA_IN.value = 0

    # Start clock and making reset
    cocotb.start_soon(Clock(dut.CLK_SYS, period_clk, unit="ns").start())
    for _ in range(8):
        await RisingEdge(dut.CLK_SYS)
    for idx in range(4):
        await RisingEdge(dut.CLK_SYS)
        dut.RSTN.value = idx % 2
        await RisingEdge(dut.CLK_SYS)
    dut.RSTN.value = 1
    for _ in range(2):
        await RisingEdge(dut.CLK_SYS)
    await FallingEdge(dut.CLK_SYS)

    # Set Trigger
    ite = 0
    dut.EN.value = 1
    assert dut.DVALID.value == 0
    for idx in range(3 * elements):
        ite += 1
        await RisingEdge(dut.CLK_SYS)
        dut.DO_SHIFT.value = 1
        dut.DATA_IN.value = data_in_array[idx % elements]
        await RisingEdge(dut.CLK_SYS)
        assert dut.DVALID.value == 0
        dut.DO_SHIFT.value = 0
        for _ in range(2):
            await RisingEdge(dut.CLK_SYS)
        if ite <= elements:
            assert dut.DATA_OUT.value == 0
        else:
            assert dut.DATA_OUT.value == data_in_array[(ite - 1) % elements]
        assert dut.DVALID.value == 1


# @pytest.mark.simulation
@pytest.mark.parametrize(
    ["bitwidth", "elements"],
    [
        (4, 6),
        (8, 16),
        (10, 31),
        (16, 256),
    ],
)
def test_shift_register(cocotb_test_fixture: CocotbTestFixture, bitwidth: int, elements: int):
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_package("windower", "verilog/shift_register.v")

    cocotb_test_fixture.set_top_module_name("SHIFT_REGISTER")
    cocotb_test_fixture.run(params={"BITWIDTH": bitwidth, "SAMPLES": elements}, defines={})
