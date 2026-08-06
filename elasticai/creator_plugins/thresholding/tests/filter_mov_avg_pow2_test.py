import cocotb
import numpy as np

# from elasticai.creator_plugins.helper import calc_mavg
# add this:
import pytest
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge
from elasticai.creator.testing import CocotbTestFixture, eai_testbench
from elasticai.creator_plugins.mac import load_and_plugin

from elasticai.preprocessor.thresholding import SettingsThreshold, Thresholding


# --- get deterministic signal for template test
def get_template_signal() -> list[int]:
    # length = 4
    return [0, 0, 0, 0]


# --- build test signal
def build_test_signal(bitwidth: int, length: int) -> list[int]:
    return [np.random.randint(0, 2**bitwidth - 1) for _ in range(length)]


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


def assert_mov_avg_equivalent(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
    length: int,
    id: int,
    data_in: list[int],
):

    module_type = "mov_avg_pow2"

    # Check if length is a power of 2
    if length & (length - 1) != 0:
        raise Exception("Length has to be a power of 2.")

    settings = SettingsThreshold(
        method="mavg",
        module_type=module_type,
        sampling_rate=1.0,
        gain=1.0,
        window_sec=float(length),
    )

    threshold = Thresholding(settings)

    build_dir = cocotb_test_fixture.get_artifact_dir() / "verilog"

    threshold.create_design(
        target="fpga",
        bitwidth=bitwidth,
        id=f"{id}",
        path2save=build_dir,
    )

    check = threshold.get_threshold_list(data_in)

    cocotb_test_fixture.write(
        {
            "data_in": data_in,
            "check": check,
        }
    )

    top_module_name = f"{module_type.upper()}_{id}"
    cocotb_test_fixture.set_top_module_name(top_module_name)
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")

    cocotb_test_fixture.run(
        params={},
        defines={},
    )


@cocotb.test()
@eai_testbench
async def filter_fir_mavg_pow2_test(
    dut,
    bitwidth: int,
    length: int,
    data_in: list[int],
    check: list[int],
):
    period_clk = 5
    period_data = 100

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

    # Synchronisation to first clk
    await RisingEdge(dut.CLK_SYS)

    # Process all data
    for val, expected in zip(data_in, check):
        await RisingEdge(dut.DO_CALC)
        dut.DATA_IN.value = val

        await FallingEdge(dut.DVALID)
        await FallingEdge(dut.CLK_SYS)
        assert dut.DVALID.value == 0

        await RisingEdge(dut.DVALID)
        print(
            "IN =",
            val,
            "EXPECTED =",
            expected,
            "OUT =",
            int(dut.DATA_OUT.value),
        )
        assert int(dut.DATA_OUT.value) == int(expected)


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [8])
@pytest.mark.parametrize("length", [4])
def test_mov_avg_pow2(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
    length: int,
):

    # deterministic test vector
    data_in = get_template_signal()

    check = data_in

    cocotb_test_fixture.write(
        {
            "data_in": data_in,
            "check": check,
        }
    )

    cocotb_test_fixture.set_top_module_name("MOVING_AVERAGE_POW2")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_package("thresholding", "verilog/*.v")
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
    check = data_in

    cocotb_test_fixture.write(
        {
            "data_in": data_in,
            "check": check,
        }
    )

    # start simulation
    cocotb_test_fixture.set_top_module_name("MOV_AVG_POW2_0")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")

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
    id = 1

    data_in = build_test_signal(bitwidth, length)

    assert_mov_avg_equivalent(cocotb_test_fixture, bitwidth, length, id, data_in)


# --- Check equivalence with extended input
@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [4])
@pytest.mark.parametrize("length", [16])
def test_mov_avg_pow2_equal_extended(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
    length: int,
):

    id = 2
    num_repeats = 4

    data_in = build_test_signal(
        bitwidth=bitwidth,
        length=length,
    )
    data_in = data_in * num_repeats

    assert_mov_avg_equivalent(cocotb_test_fixture, bitwidth, length, id, data_in)
