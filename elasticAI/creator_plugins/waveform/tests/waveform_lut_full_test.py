import random

import cocotb
import pytest
from cocotb.clock import Clock, Timer
from cocotb.triggers import RisingEdge
from elasticai.creator.arithmetic import int_arithmetic, int_converter
from elasticai.creator.testing import CocotbTestFixture, eai_testbench

from elasticai.creator_plugins.waveform.utils import load_and_plugin, prepare_waveform


@cocotb.test()
@eai_testbench
async def wvf_lut_read(
    dut, bitwidth: int, num_params: int, num_trials: int, waveform: list[int], check: list[int]
):
    period_clk = 5
    mode_trgg = False if "WAIT_CYC" in dir(dut) else True
    mode_accs = True if "LUT_DATA_EXT" in dir(dut) else False

    if mode_trgg:
        dut.TRGG_CNT.value = 0
    else:
        dut.WAIT_CYC.value = 2 ** (dut.WAIT_WIDTH.value.to_unsigned() - 1) - 1
    if mode_accs:
        sum_val = 0
        for v in reversed(waveform):
            sum_val = sum_val << bitwidth
            sum_val |= v & ((1 << bitwidth) - 1)
        dut.LUT_DATA_EXT.value = sum_val

    dut.CLK_SYS.value = 0
    dut.RSTN.value = 0
    dut.EN_FLAG.value = 0

    # Start clock and make reset
    cocotb.start_soon(Clock(dut.CLK_SYS, period_clk, unit="ns").start())
    for idx in range(4):
        await RisingEdge(dut.CLK_SYS)
    for idx in range(4):
        await RisingEdge(dut.CLK_SYS)
        dut.RSTN.value = idx % 2
    await RisingEdge(dut.CLK_SYS)
    dut.RSTN.value = 1
    for idx in range(4):
        await RisingEdge(dut.CLK_SYS)

    # Make test (several shots, full cycle through RAM)
    dut.EN_FLAG.value = 1
    if mode_trgg:
        cocotb.start_soon(Clock(dut.TRGG_CNT, 20 * period_clk, unit="ns").start())
        await Timer(period_clk, unit="ns")

    ram_data_out = list()
    for num_ite in range(num_trials):
        cnt_ram = num_params if num_ite == 0 or num_ite == num_trials else num_params - 1
        for idx in range(cnt_ram):
            if mode_trgg:
                await RisingEdge(dut.TRGG_CNT)
            else:
                for _ in range(dut.WAIT_CYC.value):
                    await RisingEdge(dut.CLK_SYS)

            dut.EN_FLAG.value = num_ite < num_trials
            ram_data_out.append(dut.LUT_OUT.value.to_signed())
        assert dut.LUT_END.value == 1
    assert ram_data_out[:num_params] == waveform
    assert ram_data_out == check


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth, num_params, num_trials", [(6, 23, 3)])
def test_waveform_lut_full_normal(
    cocotb_test_fixture: CocotbTestFixture, bitwidth: int, num_params: int, num_trials: int
):
    waveform = [0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 31, 29, 26, 23, 20, 17, 14, 11, 8, 5, 2, 0]
    check = [waveform[0]]
    for _ in range(num_trials):
        check.extend(waveform[1:])

    cocotb_test_fixture.set_top_module_name("LUT_WAVEFORM_FULL")
    cocotb_test_fixture.write({"waveform": waveform, "check": check})

    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_package("waveform", "verilog/waveform_lut_full.v")
    cocotb_test_fixture.run(
        params={
            "BITWIDTH": bitwidth,
            "WAIT_WIDTH": bitwidth,
            "LUTWIDTH": num_params,
        },
        defines={},
    )


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth, num_params, num_trials", [(6, 23, 3)])
def test_waveform_lut_full_normal_build(
    cocotb_test_fixture: CocotbTestFixture, bitwidth: int, num_params: int, num_trials: int
):

    conv = int_converter(total_bits=bitwidth, signed=True)

    data = prepare_waveform(
        waveform="SINE_FULL",
        bitwidth=bitwidth,
        num_params=num_params,
    )

    check = [data[0]]
    data0 = data.copy()
    data0.reverse()
    for _ in range(num_trials):
        check.extend(data0[1:])

    build_dir = cocotb_test_fixture.get_artifact_dir() / "verilog"
    load_and_plugin(
        type="waveform_lut_full",
        id="0",
        params={
            "BITWIDTH": bitwidth,
            "WAIT_WIDTH": bitwidth,
            "LUTWIDTH": num_params,
            "LUT_DATA": conv.integer_to_decimal_string_array_verilog(data),
        },
        path2save=build_dir,
    )

    cocotb_test_fixture.write({"waveform": data0, "check": check})
    cocotb_test_fixture.set_top_module_name("WAVEFORM_LUT_FULL_0")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")
    cocotb_test_fixture.run(params={}, defines={})


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth, num_params, num_trials", [(6, 23, 3)])
def test_waveform_lut_full_ext_trigger(
    cocotb_test_fixture: CocotbTestFixture, bitwidth: int, num_params: int, num_trials: int
):
    waveform = [0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 31, 29, 26, 23, 20, 17, 14, 11, 8, 5, 2, 0]
    check = [waveform[0]]
    for _ in range(num_trials):
        check.extend(waveform[1:])

    cocotb_test_fixture.set_top_module_name("LUT_WAVEFORM_FULL")
    cocotb_test_fixture.write({"waveform": waveform, "check": check})

    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_package("waveform", "verilog/waveform_lut_full.v")
    cocotb_test_fixture.run(
        params={
            "BITWIDTH": bitwidth,
            "LUTWIDTH": num_params,
        },
        defines={"TRGG_EXTERNAL": True},
    )


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth, num_params, num_trials", [(8, 11, 3), (16, 33, 5)])
def test_waveform_lut_full_ext_data(
    cocotb_test_fixture: CocotbTestFixture, bitwidth: int, num_params: int, num_trials: int
):
    conv = int_arithmetic(total_bits=bitwidth, signed=True)
    waveform = [
        random.randint(a=conv.minimum_as_integer, b=conv.maximum_as_integer) for _ in range(num_params)
    ]
    check = [waveform[0]]
    for _ in range(num_trials):
        check.extend(waveform[1:])

    cocotb_test_fixture.set_top_module_name("LUT_WAVEFORM_FULL")
    cocotb_test_fixture.write({"waveform": waveform, "check": check})

    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_package("waveform", "verilog/waveform_lut_full.v")
    cocotb_test_fixture.run(
        params={
            "BITWIDTH": bitwidth,
            "WAIT_WIDTH": 4,
            "LUTWIDTH": num_params,
        },
        defines={"ACCESS_EXTERNAL": True},
    )
