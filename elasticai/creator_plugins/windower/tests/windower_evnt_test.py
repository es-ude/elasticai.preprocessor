from random import randint

import cocotb
import pytest
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge
from elasticai.creator.testing import CocotbTestFixture, eai_testbench

from elasticai.creator_plugins.windower.utils import load_and_plugin


def build_testdata(bitwidth: int, samples: int) -> list[int]:
    max_value = 2**bitwidth - 1

    data = [randint(0, max_value) for _ in range(samples)]

    return data


@cocotb.test()
@eai_testbench
async def windower_evnt_tb(
    dut,
    bitwidth: int,
    samples: int,
    num_shift: int,
    data_in: list[int],
):
    period_clk = 5

    dut.CLK_SYS.value = 0
    dut.RSTN.value = 0
    dut.EN.value = 0
    dut.IS_EVNT.value = 0
    dut.DO_SHIFT.value = 0
    dut.DATA_IN.value = 0

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

    dut.EN.value = 1
    assert dut.DVALID.value == 0

    for idx in range(8 * samples):
        await RisingEdge(dut.CLK_SYS)
        dut.DO_SHIFT.value = 1
        dut.DATA_IN.value = data_in[idx % samples]

        await RisingEdge(dut.CLK_SYS)
        dut.DO_SHIFT.value = 0

        for _ in range(2):
            await RisingEdge(dut.CLK_SYS)


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [8])
@pytest.mark.parametrize("samples", [32])
@pytest.mark.parametrize("num_shift", [4])
def test_windower_evnt(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
    samples: int,
    num_shift: int,
):
    data_in = build_testdata(bitwidth=bitwidth, samples=samples)
    cocotb_test_fixture.write(
        {
            "data_in": data_in,
        }
    )

    cocotb_test_fixture.set_top_module_name("EVENT_WINDOWER")

    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_package(
        "windower",
        "verilog/ring_buffer.v",
    )
    cocotb_test_fixture.add_srcs_from_package(
        "windower",
        "verilog/windower_evnt.v",
    )

    cocotb_test_fixture.run(
        params={
            "BITWIDTH": bitwidth,
            "SAMPLES": samples,
            "NUM_SHIFT": num_shift,
        },
        defines={},
    )


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [8])
@pytest.mark.parametrize("samples", [32])
@pytest.mark.parametrize("num_shift", [4])
def test_windower_evnt_build(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
    samples: int,
    num_shift: int,
):
    data_in = build_testdata(bitwidth=bitwidth, samples=samples)
    cocotb_test_fixture.write(
        {
            "data_in": data_in,
        }
    )

    build_dir = cocotb_test_fixture.get_artifact_dir() / "verilog"

    load_and_plugin(
        type="windower_evnt",
        id="",
        params={
            "BITWIDTH": bitwidth,
            "SAMPLES": samples,
            "NUM_SHIFT": num_shift,
        },
        packages=["windower"],
        path2save=build_dir,
        add_ringbuffer=True,
    )

    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")
    cocotb_test_fixture.set_top_module_name("EVENT_WINDOWER")
    cocotb_test_fixture.run(params={}, defines={})
