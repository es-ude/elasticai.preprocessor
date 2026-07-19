import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge
from pathlib import Path
import numpy as np

from elasticai.creator.testing.cocotb_runner import run_cocotb_sim_for_src_dir
import elasticai.creator_plugins.filter_data as test_dut
# from elasticai.creator_plugins.helper import calc_mavg

#add this:
import pytest 
from elasticai.creator.testing import CocotbTestFixture, eai_testbench
from elasticai.creator_plugins.mac import load_and_plugin

# --- get deterministic signal for template test
def get_template_signal() -> list[int]:
    # length = 4
    return [ 0, 0, 0, 0 ]

# --- build reference data
def calc_mavg_reference(data_in: list[int], length: int) -> list[int]:
    """
    Reference model matching the Verilog implementation:

    pre_out <= pre_out - old_tap + DATA_IN
    DATA_OUT = pre_out >> log2(LENGTH)
    """

    taps = [0] * length
    pos = 0
    pre_out = 0

    result = []

    for sample in data_in:

        # update FIR buffer
        pre_out = pre_out - taps[pos] + sample
        taps[pos] = sample

        # output before current sample update
        result.append(pre_out // length)

        # Verilog counts backwards
        if pos == 0:
            pos = length - 1
        else:
            pos -= 1

    return result


@cocotb.test()
@eai_testbench
async def filter_fir_mavg_pow2_test(
    dut,
    bitwidth: int,
    length: int,
    data_in: list[int],
    check: list[int],
    num_repeats: int,
    ):
    period_clk = 5
    period_data = 100
    # num_repeats = 4 <- is now parameter
    do_signed = False

    used_bitwidth = bitwidth #int(dut.BITWIDTH.value)
    used_adrwidth = 4

    # --- data_in_array not needed anymore because of input signal
    # data_in_array = [
    #     np.random.randint(low=0, high=2**used_bitwidth - 1)
    #     for _ in range(used_adrwidth)
    # ]
    # # data_in_array = [int(2**(used_bitwidth-1) * (1 + np.cos(2 * np.pi * idx / used_adrwidth))) for idx in range(used_adrwidth)]
    # data_in_array = [val if val >= 0 else 0 for val in data_in_array]
    # data_in_array = [
    #     2**used_bitwidth - 1 if val >= 2**used_bitwidth - 1 else val
    #     for val in data_in_array
    # ]

    # --- Control signals
    dut.CLK_SYS.value = 0
    dut.RSTN.value = 0
    dut.EN.value = 0
    dut.DO_CALC.value = 0
    dut.DATA_IN.value = 0

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
        for val, expected in zip(data_in, check):
            await RisingEdge(dut.DO_CALC)
            dut.DATA_IN.value = val

            await FallingEdge(dut.DVALID)
            await FallingEdge(dut.CLK_SYS)
            assert dut.DVALID.value == 0

            await RisingEdge(dut.DVALID)
            assert dut.DVALID.value == 1

            output = int(dut.DATA_OUT.value)
            print(
                "IN =",
                val,
                "EXPECTED =",
                expected,
                "OUT =",
                output,
            )
            assert output == expected



@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [8])
@pytest.mark.parametrize("length", [4])
def test_mov_avg_pow2(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
    length: int,
	):
    num_repeats = 1

    # deterministic test vector
    data_in = get_template_signal()

    check = calc_mavg_reference(
        data_in,
        length,
    )

    cocotb_test_fixture.write(
        {
            "data_in": data_in,
            "check": check,
            "num_repeats": num_repeats,
        }
    )

    cocotb_test_fixture.set_top_module_name(
        "MOVING_AVERAGE_POW2"
    )
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_package(
        "thresholding",
        "verilog/*.v"
    )
    cocotb_test_fixture.run(
        params={
            "BITWIDTH": bitwidth,
            "LENGTH": length,
        },
        defines={},
    )

# --- build test
@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [4])
@pytest.mark.parametrize("length", [4])
def test_mov_avg_pow2_build(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
    length: int,
):

    num_repeats = 1

    # Directory for artifact
    artifact_dir = cocotb_test_fixture.get_artifact_dir()
    build_dir = artifact_dir / "verilog"

    # generate verilog using plugin
    load_and_plugin(
        type="mov_avg_pow2",
        id="0",
        params={
            "BITWIDTH": bitwidth,
            "LENGTH": length,
        },
        packages=["thresholding"],
        path2save=build_dir,
    )

    # input data
    data_in = get_template_signal()

    # reference output
    check = calc_mavg_reference(
        data_in,
        length,
    )

    cocotb_test_fixture.write(
        {
            "data_in": data_in,
            "check": check,
            "num_repeats": num_repeats,
        }
    )

    # start simulation
    cocotb_test_fixture.set_top_module_name(
        "MOV_AVG_POW2_0"
    )
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir(
        "verilog/*.v"
    )

    cocotb_test_fixture.run(
        params={},
        defines={},
    )
    
# --- Check equivalence to reference function
@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [4])
@pytest.mark.parametrize("length", [4])
def test_mov_avg_pow2_equal(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
    length: int,
):
    num_repeats = 1

    build_dir = (
        cocotb_test_fixture.get_artifact_dir()
        / "verilog"
    )

    # random input vector
    data_in = [
        np.random.randint(
            0,
            2**bitwidth
        )
        for _ in range(20)
    ]

    # reference model
    data_check = calc_mavg_reference(
        data_in,
        length,
    )

    # generate module
    load_and_plugin(
        type="mov_avg_pow2",
        id="1",
        params={
            "BITWIDTH": bitwidth,
            "LENGTH": length,
        },
        packages=["thresholding"],
        path2save=build_dir,
    )

    cocotb_test_fixture.write(
        {
            "data_in": data_in,
            "check": data_check,
            "num_repeats": num_repeats,
        }
    )
    cocotb_test_fixture.set_top_module_name(
        "MOV_AVG_POW2_1"
    )
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir(
        "verilog/*.v"
    )
    cocotb_test_fixture.run(
        params={},
        defines={},
    )