import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge
from pathlib import Path
import numpy as np

from elasticai.creator.testing.cocotb_runner import run_cocotb_sim_for_src_dir
import elasticai.creator_plugins.filter_data as test_dut
from elasticai.creator_plugins.helper import calc_mavg


cocotb_settings = dict(
    src_files=["filter_mov_avg_norm.v"],
    top_module_name="MOVING_AVERAGE",
    cocotb_test_module="elasticai.creator_plugins.filters.tests.filter_mov_avg_norm_tb",
    path2src=Path(test_dut.__file__).parent / "verilog",
    params={"BITWIDTH": 8, "LENGTH": 4},
)


@cocotb.test()
async def filter_fir_mavg_pow2_tb(dut):
    period_clk = 5
    period_data = 100
    num_repeats = 4
    do_signed = False

    used_bitwidth = int(dut.BITWIDTH.value)
    used_adrwidth = 4
    data_in_array = [
        np.random.randint(low=0, high=2**used_bitwidth - 1)
        for _ in range(used_adrwidth)
    ]
    # data_in_array = [int(2**(used_bitwidth-1) * (1 + np.cos(2 * np.pi * idx / used_adrwidth))) for idx in range(used_adrwidth)]
    data_in_array = [val if val >= 0 else 0 for val in data_in_array]
    data_in_array = [
        2**used_bitwidth - 1 if val >= 2**used_bitwidth - 1 else val
        for val in data_in_array
    ]

    # --- Moving mean buffer
    mavg_buffer = [0 for _ in range(used_adrwidth)]

    # --- Control signals
    dut.CLK_SYS.value = 0
    dut.RSTN.value = 0
    dut.EN.value = 0
    dut.DO_CALC.value = 0
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

    # Set Data on Trigger
    dut.EN.value = 1
    assert dut.DVALID.value == 0
    for _ in range(8):
        await RisingEdge(dut.CLK_SYS)
    cocotb.start_soon(Clock(dut.DO_CALC, period_data, unit="ns").start())
    ite = 0
    for idx in range(num_repeats):
        await RisingEdge(dut.CLK_SYS)
        assert dut.DVALID.value == (idx > 0)
        for val in data_in_array:
            await RisingEdge(dut.DO_CALC)
            dut.DATA_IN.value = val
            test_val = calc_mavg(mavg_buffer, used_bitwidth, do_signed)
            mavg_buffer[ite % used_adrwidth] = val
            ite += 1

            await FallingEdge(dut.DVALID)
            await FallingEdge(dut.CLK_SYS)
            assert dut.DVALID.value == 0

            await RisingEdge(dut.DVALID)
            assert dut.DVALID.value == 1
            print(
                int(dut.DATA_OUT.value),
                calc_mavg(mavg_buffer, used_bitwidth, do_signed),
            )
            assert dut.DATA_OUT.value == test_val


if __name__ == "__main__":
    run_cocotb_sim_for_src_dir(**cocotb_settings)
