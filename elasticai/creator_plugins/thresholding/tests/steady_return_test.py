import cocotb
import numpy as np

# from elasticai.creator_plugins.helper import calc_mavg
# add this:
import pytest
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
from elasticai.creator.testing import CocotbTestFixture, eai_testbench
from elasticai.creator_plugins.mac import load_and_plugin


def build_test_signal(bitwidth: int, length: int) -> list[int]:
    return [np.random.randint(0, 2**bitwidth - 1) for _ in range(length)]


@cocotb.test()
@eai_testbench
async def steady_return_test(
    dut,
    bitwidth: int,
    data_in: list[int],
):

    check = data_in  # input = output

    period_clk = 5
    period_data = 100

    dut.CLK_SYS.value = 0
    dut.RSTN.value = 0
    dut.EN.value = 0
    dut.DO_CALC.value = 0
    dut.DATA_IN.value = 0

    cocotb.start_soon(Clock(dut.CLK_SYS, period_clk, unit="ns").start())

    for _ in range(8):
        await RisingEdge(dut.CLK_SYS)

    dut.RSTN.value = 1
    dut.EN.value = 1

    for _ in range(2):
        await RisingEdge(dut.CLK_SYS)

    cocotb.start_soon(Clock(dut.DO_CALC, period_data, unit="ns").start())

    await RisingEdge(dut.CLK_SYS)

    for value, expected in zip(data_in, check):
        await RisingEdge(dut.DO_CALC)
        dut.DATA_IN.value = value

        await RisingEdge(dut.DVALID)

        print(
            "IN =",
            value,
            "EXPECTED =",
            expected,
            "OUT =",
            int(dut.DATA_OUT.value),
        )

        assert int(dut.DATA_OUT.value) == expected


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [4])
def test_steady_return(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
):

    data_in = build_test_signal(bitwidth, 20)

    cocotb_test_fixture.write({"data_in": data_in})

    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_package(
        "thresholding",
        "verilog/*.v",
    )

    cocotb_test_fixture.set_top_module_name("STEADY_RETURN")

    cocotb_test_fixture.run(
        params={
            "BITWIDTH": bitwidth,
        },
        defines={},
    )


# --- build test
@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [4])
def test_steady_return_build(cocotb_test_fixture: CocotbTestFixture, bitwidth: int):

    # Directory for artifact
    artifact_dir = cocotb_test_fixture.get_artifact_dir()
    build_dir = artifact_dir / "verilog"

    # generate verilog using plugin
    load_and_plugin(
        type="steady_return",
        id="0",
        params={"BITWIDTH": bitwidth},
        packages=["thresholding"],
        path2save=build_dir,
    )

    # input data
    data_in = build_test_signal(bitwidth, 8)

    cocotb_test_fixture.write(
        {
            "data_in": data_in,
        }
    )

    # start simulation
    cocotb_test_fixture.set_top_module_name("STEADY_RETURN_0")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")

    cocotb_test_fixture.run(
        params={},
        defines={},
    )
