import pytest
import cocotb
import numpy as np

from cocotb.clock import Clock
from cocotb.triggers import Timer, FallingEdge
from elasticai.creator.testing import CocotbTestFixture, eai_testbench
from elasticai.creator_plugins.datarate.utils import load_and_plugin


@cocotb.test()
@eai_testbench
async def polyphase_access(dut, bitwidth: int, poly_order: int):
    period_smp = 10
    num_periods = 2
    mid_cm = 2 ** (bitwidth -1) 
    sig_in = np.array(mid_cm + (mid_cm -2) * np.sin(np.linspace(start=0, stop=num_periods*2*np.pi, num=22, endpoint=True, dtype=float)), dtype=int)
    gain_cic = 2 ** poly_order

    dut.CLK_HGH.value = 0
    dut.RSTN.value = 1
    dut.EN.value = 0
    dut.DATA_IN.value = 0

    # Start clock and make reset
    await Timer(4 * period_smp, unit='ns')
    for idx in range(4):
        await Timer(4 * period_smp, unit='ns')
        dut.RSTN.value = idx % 2
    await Timer(4 * period_smp, unit='ns')
    dut.RSTN.value = 1
    await Timer(4 * period_smp, unit='ns')

    # Apply data and test
    dut.DATA_IN.value = int(sig_in[0])
    dut.EN.value = 1
    cocotb.start_soon(Clock(dut.CLK_HGH, period_smp, unit='ns').start())
    for val in sig_in:
        dut.DATA_IN.value = int(val)

        await FallingEdge(dut.CLK_LOW)
        if poly_order > 0: 
            await FallingEdge(dut.CLK_LOW)
        if poly_order> 1: 
            await FallingEdge(dut.CLK_LOW)
        assert dut.DATA_OUT.value in range(int(int(val) * gain_cic - 1), int(int(val) * gain_cic + 1))


#  --------------- (1) Template Test ------------------

@pytest.mark.simulation
#@pytest.mark.parametrize("bitwidth, poly_order", [(1, 0), (1, 2), (2, 1), (2, 3), (4, 0), (4, 2), (8, 1), (8, 2), (12, 1), (12, 3), (16, 1), (16, 2)])
@pytest.mark.parametrize("bitwidth", [1, 2, 4, 8, 12, 16])
@pytest.mark.parametrize("poly_order", [0, 1, 2, 3])

def test_filter_polydec_asic(cocotb_test_fixture: CocotbTestFixture, bitwidth: int, poly_order: int):
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_package("datarate", "verilog/filter_polydec_asic.v")
    cocotb_test_fixture.set_top_module_name("FILTER_POLYDEC_ASIC")
    cocotb_test_fixture.run(params={"BITWIDTH": bitwidth, "POLY_ORDER": poly_order}, defines={})


#  --------------- (2) Build Test ------------------ (Kann ich es bauen und das Gebaute testen?)

@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [1, 2, 4, 8, 12, 16])
@pytest.mark.parametrize("poly_order", [0, 1, 2, 3])    

def test_filter_polydec_asic_build(cocotb_test_fixture: CocotbTestFixture, bitwidth: int, poly_order: int):
    build_dir = cocotb_test_fixture.get_artifact_dir() / "verilog"

    load_and_plugin(
        type="filter_poly_asic",
        id="",
        params={"BITWIDTH": bitwidth, "POLY_ORDER": poly_order},
        packages=["datarate"],
        path2save=build_dir,
        )
    
    cocotb_test_fixture.set_top_module_name("FILTER_POLYDEC_ASIC")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")
    cocotb_test_fixture.run(params={}, defines={})

#  --------------- (3) Äquivalenz Test ------------------
