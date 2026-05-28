import random
import cocotb
import numpy as np
from pathlib import Path
from cocotb.clock import Clock
from cocotb.triggers import Timer, RisingEdge, FallingEdge

from elasticai.creator.file_generation import find_project_root
from elasticai.creator.testing.cocotb_runner import run_cocotb_sim_for_src_dir
import elasticai.creator_plugins.datarate as test_dut


cocotb_settings = dict(
    src_files=["filter_cic.v"],
    top_module_name='FILTER_CIC',
    path2src=Path(test_dut.__file__).parent / 'verilog',
    cocotb_test_module="elasticai.creator_plugins.datarate.tests.filter_cic_tb",
    params={'BITWIDTH': 6, 'DEC_RATE': 2, 'N_DEC': 4},
    waveform_save_dst=find_project_root() / "build_temp"
)


@cocotb.test()
async def cic_access(dut):
    period_clk = 5
    period_smp = 10
    valrange = 2 ** int(dut.BITWIDTH.value)
    gain_cic = int(int(dut.N_DEC.value) * np.log2(int(dut.DEC_RATE.value)))

    dut.CLK_SYS.value = 0
    dut.CLK_SMP.value = 0
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
    dut.EN.value = 1

    # Apply data and test
    cocotb.start_soon(Clock(dut.CLK_SMP, period_smp, unit='ns').start())
    for _ in range(1):
        val_in = random.randint(0, valrange - 1)
        dut.DATA_IN.value = val_in

        await FallingEdge(dut.DEC_CLK)
        await Timer(period_clk, unit='ns')
        assert dut.DATA_OUT.value.to_unsigned() in range(int(val_in * gain_cic / 2 - 1), int(val_in * gain_cic / 2 + 1))
        await FallingEdge(dut.DEC_CLK)
        await Timer(period_clk, unit='ns')
        assert dut.DATA_OUT.value.to_unsigned() in range(int(val_in * gain_cic - 1), int(val_in * gain_cic + 1))


if __name__ == "__main__":
    run_cocotb_sim_for_src_dir(**cocotb_settings)
