import cocotb
import numpy as np
from pathlib import Path
from cocotb.clock import Clock
from cocotb.triggers import Timer, FallingEdge

import elasticai.creator_plugins.datarate as test_dut
from elasticai.creator.testing.cocotb_runner import run_cocotb_sim_for_src_dir 


cocotb_settings = dict(
    src_files=["filter_polydec_asic.v"],
    top_module_name='FILTER_POLYDEC_ASIC',
    path2src=Path(test_dut.__file__).parent / 'verilog',
    cocotb_test_module="elasticai.creator_plugins.datarate.tests.filter_polydec_asic_tb",
    params={'BITWIDTH': 6, 'POLY_ORDER': 0}
)


@cocotb.test()
async def polyphase_access(dut):
    period_smp = 10
    num_periods = 2
    mid_cm = 2 ** (int(dut.BITWIDTH.value) -1)
    sig_in = np.array(mid_cm + (mid_cm -2) * np.sin(np.linspace(start=0, stop=num_periods*2*np.pi, num=22, endpoint=True, dtype=float)), dtype=int)
    gain_cic = 2 ** int(dut.POLY_ORDER.value)

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
        if int(dut.POLY_ORDER.value) > 0:
            await FallingEdge(dut.CLK_LOW)
        if int(dut.POLY_ORDER.value) > 1:
            await FallingEdge(dut.CLK_LOW)
        assert dut.DATA_OUT.value in range(int(int(val) * gain_cic - 1), int(int(val) * gain_cic + 1))


if __name__ == "__main__":
    run_cocotb_sim_for_src_dir(**cocotb_settings)
