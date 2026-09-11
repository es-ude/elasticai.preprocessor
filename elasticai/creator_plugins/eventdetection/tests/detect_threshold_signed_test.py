import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

import pytest

from elasticai.creator.testing import CocotbTestFixture, eai_testbench
from elasticai.creator_plugins.mac import load_and_plugin


def get_test_signal() -> tuple[list[int], list[int]]:
    """
    Returns input values and expected event detection results
    for THR = 4.
    """
    data_in = [
        0,
        1,
        3,
        4,
        5,
        8,
        15,
    ]

    check = [
        0,  # 0 > 4  -> false
        0,  # 1 > 4  -> false
        0,  # 3 > 4  -> false
        1,  # 4 > 4  -> true
        1,  # 5 > 4  -> true
        1,  # 8 > 4  -> true
        1,  # 15 > 4 -> true
    ]

    return data_in, check


@cocotb.test()
@eai_testbench
async def unsigned_threshold_test(
    dut,
    bitwidth: int,
    data_in: list[int],
    check: list[int],
):
    period_clk = 5

    dut.CLK_SYS.value = 0
    dut.RSTN.value = 0
    dut.EN.value = 0
    dut.DO_CALC.value = 0
    dut.DATA_IN.value = 0
    dut.THR.value = 4

    cocotb.start_soon(
        Clock(dut.CLK_SYS, period_clk, unit="ns").start()
    )

    # Reset
    for _ in range(4):
        await RisingEdge(dut.CLK_SYS)

    dut.RSTN.value = 1

    # Enable module
    dut.EN.value = 1

    for _ in range(2):
        await RisingEdge(dut.CLK_SYS)

    assert len(data_in) == len(check)

    for val, expected in zip(data_in, check):

        dut.DATA_IN.value = val
        dut.DO_CALC.value = 1

        await RisingEdge(dut.CLK_SYS)

        print(
            "IN =", val,
            "THR =", 4,
            "EXPECTED =", expected,
            "OUT =", int(dut.IS_EVNT.value),
            "DVALID =", int(dut.DVALID.value),
        )

        assert int(dut.IS_EVNT.value) == expected
        assert int(dut.DVALID.value) == 1

        # Calculation finished
        dut.DO_CALC.value = 0

        await RisingEdge(dut.CLK_SYS)

        # Result should be held
        assert int(dut.IS_EVNT.value) == expected
        assert int(dut.DVALID.value) == 0


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [4])
def test_unsigned_threshold_build(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
):
    artifact_dir = cocotb_test_fixture.get_artifact_dir()
    build_dir = artifact_dir / "verilog"

    load_and_plugin(
        type="unsigned_threshold",
        id="0",
        params={
            "BITWIDTH": bitwidth,
        },
        packages=["thresholding"],
        path2save=build_dir,
    )

    data_in, check = get_test_signal()

    cocotb_test_fixture.write(
        {
            "data_in": data_in,
            "check": check,
        }
    )

    cocotb_test_fixture.set_top_module_name(
        "UNSIGNED_THRESHOLD_0"
    )

    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir(
        "verilog/*.v"
    )

    cocotb_test_fixture.run(
        params={},
        defines={},
    )