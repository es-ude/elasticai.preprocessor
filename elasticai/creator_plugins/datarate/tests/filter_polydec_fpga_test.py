import pytest
import cocotb
import numpy as np

from cocotb.clock import Clock
from cocotb.triggers import Timer, RisingEdge, FallingEdge
from elasticai.creator.testing import CocotbTestFixture, eai_testbench
from elasticai.creator_plugins.datarate.utils import load_and_plugin


@cocotb.test()
@eai_testbench
async def polyphase_access(dut, bitwidth: int, poly_order: int):
    period_clk = 5
    period_smp = 10
    num_periods = 2
    mid_cm = 2 ** (bitwidth -1)
    sig_in = np.array(mid_cm + (mid_cm -2) * np.sin(np.linspace(start=0, stop=num_periods*2*np.pi, num=22, endpoint=True, dtype=float)), dtype=int)
    gain_cic = 2 ** poly_order

    dut.CLK_SYS.value = 0
    dut.CLK_HGH.value = 0
    dut.RSTN.value = 1
    dut.EN.value = 0
    dut.DATA_IN.value = 0

    # Start clock and make reset
    assert period_clk <= period_smp
    cocotb.start_soon(Clock(dut.CLK_SYS, period_clk, unit='ns').start())
    await Timer(4 * period_clk, unit='ns')
    for idx in range(4):
        await RisingEdge(dut.CLK_SYS)
        dut.RSTN.value = idx % 2
    await RisingEdge(dut.CLK_SYS)
    dut.RSTN.value = 1
    await Timer(4 * period_clk, unit='ns')

    # Apply data and test
    dut.DATA_IN.value = int(sig_in[0])
    dut.EN.value = 1
    cocotb.start_soon(Clock(dut.CLK_HGH, period_smp, unit='ns').start())
    for val in sig_in:
        dut.DATA_IN.value = int(val)

        await FallingEdge(dut.CLK_LOW)
        if poly_order > 0:
            await FallingEdge(dut.CLK_LOW)
        if poly_order > 1:
            await FallingEdge(dut.CLK_LOW)
        assert dut.DATA_OUT.value in range(int(int(val) * gain_cic - 1), int(int(val) * gain_cic + 1))


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth, poly_order", [
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 0),
        (2, 1),
        (2, 2),
        (2, 3),
        (4, 0),
        (4, 2),
        (8, 0),
        (8, 3),
        (12, 1),
        (12, 2),
        (16, 0),
        (16, 3)
    ]
)

def test_filter_polydec_fpga(cocotb_test_fixture: CocotbTestFixture, bitwidth: int, poly_order: int):
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_package("datarate", "verilog/filter_polydec_fpga.v")
    cocotb_test_fixture.set_top_module_name("FILTER_POLYDEC_FPGA")
    cocotb_test_fixture.run(params={"BITWIDTH": bitwidth, "POLY_ORDER": poly_order}, defines={})


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth, poly_order", [
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 0),
        (2, 1),
        (2, 2),
        (2, 3),
        (4, 0),
        (4, 2),
        (8, 0),
        (8, 3),
        (12, 1),
        (12, 2),
        (16, 0),
        (16, 3)
    ]
)

def test_filter_polydec_fpga_build(cocotb_test_fixture: CocotbTestFixture, bitwidth: int, poly_order: int):
    artifact_dir = cocotb_test_fixture.get_artifact_dir()
    build_dir = artifact_dir / "verilog"

    load_and_plugin(
        type="filter_poly_fpga",
        id=id,
        params={"BITWIDTH": bitwidth, "POLY_ORDER": poly_order},
        packages=["datarate"],
        path2save=build_dir,
        )
    cocotb_test_fixture.set_top_module_name("FILTER_POLYDEC_FPGA")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")
    cocotb_test_fixture.run(params={}, defines={})