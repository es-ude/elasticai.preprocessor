import cocotb
import numpy as np
import pytest
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge
from elasticai.creator.testing import CocotbTestFixture, eai_testbench


@cocotb.test()
@eai_testbench
async def both_register_tb(dut, bitwidth: int, samples: int):
    period_clk = 5

    data_in_array = [
        int(2 ** (bitwidth - 1) * (1 + np.cos(2 * np.pi * idx / samples))) for idx in range(samples)
    ]
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
    for idx in range(3 * samples):
        ite += 1
        await RisingEdge(dut.CLK_SYS)
        dut.DO_SHIFT.value = 1
        dut.DATA_IN.value = data_in_array[idx % samples]
        await RisingEdge(dut.CLK_SYS)
        assert dut.DVALID.value == 0
        dut.DO_SHIFT.value = 0
        for _ in range(2):
            await RisingEdge(dut.CLK_SYS)
        assert dut.DATA_BUF0.value == dut.DATA_BUF1.value
        assert dut.DATA_OUT0.value == dut.DATA_OUT1.value
        assert dut.DVALID.value == 1


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth, samples", [(8, 12), (6, 128)])
def test_output_shift_and_ringbuffer(cocotb_test_fixture: CocotbTestFixture, bitwidth: int, samples: int):
    cocotb_test_fixture.set_top_module_name("BOTH_REGISTER")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_package("windower", "verilog/both_register.v")
    cocotb_test_fixture.add_srcs_from_package("windower", "verilog/shift_register.v")
    cocotb_test_fixture.add_srcs_from_package("windower", "verilog/ring_buffer.v")
    cocotb_test_fixture.run(params={"BITWIDTH": bitwidth, "SAMPLES": samples}, defines={})
