import random

import cocotb
import elasticai.creator_plugins.bram as bram
import pytest
from cocotb.clock import Clock, Timer
from cocotb.triggers import RisingEdge
from elasticai.creator.arithmetic import int_arithmetic
from elasticai.creator.testing import CocotbTestFixture, eai_testbench
from elasticai.creator_plugins.bram.utils import translate_path_to_int, write_mem_file

from elasticai.creator_plugins.waveform.tests.waveform_lut_full_test import reconstruct_signal
from elasticai.creator_plugins.waveform.utils import WaveformGenerator, load_and_plugin


@cocotb.test()
@eai_testbench
async def wvf_ram_write_one_value_read_one_trial(
    dut, bitwidth: int, num_params: int, is_signed: bool, check: list[int]
):
    period_clk = 5
    ramrange = num_params
    ram_data_out = [0 for _ in range(ramrange)]
    mode_trgg = False if "WAIT_CYC" in dir(dut) else True

    if mode_trgg:
        dut.TRGG_CNT.value = 0
    else:
        dut.WAIT_CYC.value = bitwidth - 1
    dut.CLK_SYS.value = 0
    dut.RSTN.value = 0
    dut.EN_FLAG.value = 0
    dut.RAM_WE.value = 0
    dut.RAM_ADR.value = 0
    dut.RAM_IN.value = 0

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

    # Test #1: Read single frame
    dut.EN_FLAG.value = 1
    if mode_trgg:
        cocotb.start_soon(Clock(dut.TRGG_CNT, 20 * period_clk, unit="ns").start())
        await Timer(period_clk, unit="ns")
    for idx in range(ramrange):
        if mode_trgg:
            await RisingEdge(dut.TRGG_CNT)
        else:
            for _ in range(dut.WAIT_CYC.value):
                await RisingEdge(dut.CLK_SYS)

        dut.EN_FLAG.value = 0
        if is_signed:
            ram_data_out[idx] = dut.RAM_OUT.value.to_signed()
        else:
            ram_data_out[idx] = dut.RAM_OUT.value.to_unsigned()

    assert dut.RAM_END.value == 1
    if not ram_data_out == check:
        print("REF", check)
        print("OUT", ram_data_out)
    assert ram_data_out == check

    # Test #2 --- Change data
    change_idx = [random.randint(0, ramrange - 1) for _ in range(4)]
    change_val = [random.randint(0, ramrange - 1) for _ in change_idx]

    ram_data_chck0 = [val for val in reversed(dut.BRAM.bram_block.value)]
    for idx, val in zip(change_idx, change_val):
        ram_data_chck0[idx] = val

        dut.RAM_ADR.value = idx
        dut.RAM_IN.value = val

        await RisingEdge(dut.CLK_SYS)
        dut.RAM_WE.value = 1
        await RisingEdge(dut.CLK_SYS)
        dut.RAM_WE.value = 0
        await RisingEdge(dut.CLK_SYS)

        dut.RAM_ADR.value = 0
        dut.RAM_IN.value = 0

    ram_data_chck1 = [val for val in reversed(dut.BRAM.bram_block.value)]
    assert ram_data_chck0 != ram_data_chck1


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth, num_params", [(6, 23)])
@pytest.mark.parametrize("is_signed", [False])
def test_waveform_ram_full_normal(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
    num_params: int,
    is_signed: bool,
):
    waveform = [0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 31, 29, 26, 23, 20, 17, 14, 11, 8, 5, 2, 0]
    build_dir = cocotb_test_fixture.get_artifact_dir()
    path2file = build_dir / "data.mem"
    write_mem_file(path=path2file, data=waveform, bitwidth=bitwidth)

    check = [waveform[0]]
    for _ in range(1):
        check.extend(waveform[1:])

    cocotb_test_fixture.write({"check": check})
    cocotb_test_fixture.set_top_module_name("RAM_WAVEFORM_FULL")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_package("waveform", "verilog/waveform_ram_full.v")
    cocotb_test_fixture.add_srcs_from_package(bram, "verilog/bram_single_port.v")
    cocotb_test_fixture.run(
        params={
            "BITWIDTH": bitwidth,
            "WAIT_WIDTH": bitwidth,
            "RAMWIDTH": num_params,
            "PATH2MEM": translate_path_to_int(path2file),
        },
        defines={},
    )


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth, num_params", [(6, 23)])
@pytest.mark.parametrize("is_signed", [False])
def test_waveform_ram_full_normal_build(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
    num_params: int,
    is_signed: bool,
):
    waveform = [0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 31, 29, 26, 23, 20, 17, 14, 11, 8, 5, 2, 0]
    build_dir = cocotb_test_fixture.get_artifact_dir() / "verilog"
    path2file = build_dir / "data.mem"

    write_mem_file(path=path2file, data=waveform, bitwidth=bitwidth)
    load_and_plugin(
        type="waveform_ram_full",
        id="0",
        params={
            "BITWIDTH": bitwidth,
            "WAIT_WIDTH": bitwidth,
            "RAMWIDTH": num_params,
            "PATH2MEM": translate_path_to_int(path2file),
        },
        path2save=build_dir,
        use_bram=True,
    )

    check = reconstruct_signal(waveform=waveform, num_trials=1)

    cocotb_test_fixture.write({"check": check})
    cocotb_test_fixture.set_top_module_name("WAVEFORM_RAM_FULL_0")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")
    cocotb_test_fixture.run(params={}, defines={})


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth, num_params", [(6, 21)])
@pytest.mark.parametrize("is_signed", [False])
def test_waveform_ram_full_normal_build2(
    cocotb_test_fixture: CocotbTestFixture, bitwidth: int, num_params: int, is_signed: bool
):
    build_dir = cocotb_test_fixture.get_artifact_dir() / "verilog"
    data0 = WaveformGenerator(100.0, False).create_design(
        waveform="SINE_FULL",
        num_params=num_params,
        is_signed=is_signed,
        target="fpga",
        bitwidth=bitwidth,
        id="1",
        path2save=build_dir,
        use_bram=True,
        do_opt=False,
    )
    check = reconstruct_signal(waveform=data0, num_trials=1)
    cocotb_test_fixture.write({"check": check})
    cocotb_test_fixture.set_top_module_name("WAVEFORM_RAM_FULL_1")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")
    cocotb_test_fixture.run(params={}, defines={})


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth, num_params", [(6, 23)])
@pytest.mark.parametrize("is_signed", [False, True])
def test_waveform_ram_full_ext_trigger(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
    num_params: int,
    is_signed: bool,
):
    arith = int_arithmetic(total_bits=bitwidth - 1, signed=is_signed)
    waveform = [
        arith.to_twos(random.randint(a=arith.minimum_as_integer, b=arith.maximum_as_integer))
        for _ in range(num_params)
    ]

    build_dir = cocotb_test_fixture.get_artifact_dir()
    path2file = build_dir / "data.mem"
    write_mem_file(path=path2file, data=waveform, bitwidth=bitwidth)

    check = [waveform[0]]
    for _ in range(1):
        check.extend(waveform[1:])

    cocotb_test_fixture.write({"check": check})
    cocotb_test_fixture.set_top_module_name("RAM_WAVEFORM_FULL")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_package("waveform", "verilog/waveform_ram_full.v")
    cocotb_test_fixture.add_srcs_from_package(bram, "verilog/bram_single_port.v")
    cocotb_test_fixture.run(
        params={
            "BITWIDTH": bitwidth,
            "RAMWIDTH": num_params,
            "PATH2MEM": translate_path_to_int(path2file),
        },
        defines={"TRGG_EXTERNAL": True},
    )
