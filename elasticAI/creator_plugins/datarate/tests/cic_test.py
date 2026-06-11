import random

import cocotb
import numpy as np
import pytest
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer
from elasticai.creator.testing import CocotbTestFixture, eai_testbench

from elasticai.creator_plugins.datarate.utils import load_and_plugin


@cocotb.test()
@eai_testbench
async def cic_access(dut, bitwidth: int, dec_rate: int, n_dec: int):
    period_clk = 5
    period_smp = 10
    valrange = int(2**bitwidth)
    gain_cic = int(n_dec * np.log2(dec_rate))

    dut.CLK_SYS.value = 0
    dut.CLK_SMP.value = 0
    dut.RSTN.value = 1
    dut.EN.value = 0
    dut.DATA_IN.value = 0

    # Start clock and make reset
    assert period_clk <= period_smp
    cocotb.start_soon(Clock(dut.CLK_SYS, period_clk, unit="ns").start())
    await Timer(4 * period_clk, unit="ns")
    for idx in range(4):
        await RisingEdge(dut.CLK_SYS)
        dut.RSTN.value = idx % 2
    await RisingEdge(dut.CLK_SYS)
    dut.RSTN.value = 1
    await Timer(4 * period_clk, unit="ns")
    dut.EN.value = 1

    # Apply data and test
    cocotb.start_soon(Clock(dut.CLK_SMP, period_smp, unit="ns").start())
    for _ in range(1):
        val_in = random.randint(0, valrange - 1)
        dut.DATA_IN.value = val_in

        await FallingEdge(dut.DEC_CLK)
        await Timer(period_clk, unit="ns")
        assert dut.DATA_OUT.value.to_unsigned() in range(
            int(val_in * gain_cic / 2 - 1), int(val_in * gain_cic / 2 + 1)
        )
        await FallingEdge(dut.DEC_CLK)
        await Timer(period_clk, unit="ns")
        assert dut.DATA_OUT.value.to_unsigned() in range(
            int(val_in * gain_cic - 1), int(val_in * gain_cic + 1)
        )


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [2, 6, 8, 12, 16])
@pytest.mark.parametrize("dec_rate", [2])
@pytest.mark.parametrize("n_dec", [2])
def test_filter_cic(cocotb_test_fixture: CocotbTestFixture, bitwidth: int, dec_rate: int, n_dec: int):
    cocotb_test_fixture.set_top_module_name("FILTER_CIC")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_package("datarate", "verilog/cic.v")
    cocotb_test_fixture.run(
        params={"BITWIDTH": bitwidth, "DEC_RATE": dec_rate, "N_DEC": n_dec}, defines={}
    )


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [2, 8])
@pytest.mark.parametrize("dec_rate", [2])
@pytest.mark.parametrize("n_dec", [2])
def test_filter_cic_build(
    cocotb_test_fixture: CocotbTestFixture, bitwidth: int, dec_rate: int, n_dec: int
):
    artifact_dir = cocotb_test_fixture.get_artifact_dir()
    build_dir = artifact_dir / "verilog"

    load_and_plugin(
        type="cic",
        id="0",
        params={"BITWIDTH": bitwidth, "DEC_RATE": dec_rate, "N_DEC": n_dec},
        packages=["datarate"],
        path2save=build_dir,
    )

    """testwidth = [1, 2, 3, 4]
    cocotb_test_fixture.write( # write um Signale einzugeben
        {"testwidth": testwidth}
    )"""

    cocotb_test_fixture.set_top_module_name("CIC_0")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")
    cocotb_test_fixture.run(params={}, defines={})


#  --------------- (3) Äquivalenz Test ------------------
