import cocotb
import numpy as np
import pytest
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, RisingEdge
from elasticai.creator.arithmetic import int_arithmetic
from elasticai.creator.testing import CocotbTestFixture, eai_testbench
from elasticai.creator_plugins.mac import load_and_plugin

from elasticai.preprocessor.thresholding import SettingsThreshold, Thresholding
from elasticai.preprocessor.translation.cocotb_tmp import temporary_directory


@cocotb.test()
@eai_testbench  # add this
async def filter_moving_average_test(
    dut,
    bitwidth: int,
    length: int,
    data_in: list[int],  # new input parameter
    check: list[int],  # check signal
):
    period_clk = 5
    period_data = 100

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
    await ClockCycles(dut.CLK_SYS, 8)
    await FallingEdge(dut.CLK_SYS)

    # Set Data on Trigger
    dut.EN.value = 1
    assert dut.DVALID.value == 0
    await ClockCycles(dut.CLK_SYS, 8)

    # Process all data
    dout = []
    cocotb.start_soon(Clock(dut.DO_CALC, period_data, unit="ns").start())
    for val in data_in:
        dut.DATA_IN.value = val
        await RisingEdge(dut.DO_CALC)

        await FallingEdge(dut.DO_CALC)
        dout.append(int(dut.DATA_OUT.value.to_signed()))

    await ClockCycles(dut.CLK_SYS, int(period_data / period_clk) - 2)

    passed = True
    for val, exp in zip(dout, check):
        passed = passed and val in [exp - 1, exp, exp + 1]

    if not passed:
        print("IN:", len(data_in), data_in)
        print("OUT:", len(dout), dout)
        print("CHECK:", len(check), check)
    assert passed


# --- template test
@pytest.mark.simulation
@pytest.mark.parametrize(
    "bitwidth, length",
    [
        (8, 4),
    ],
)
def test_template(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
    length: int,
):

    # --- Build test data
    cnv = int_arithmetic(total_bits=bitwidth, signed=True)
    data_in = np.linspace(
        cnv.minimum_as_integer, cnv.maximum_as_integer, num=16, dtype=int, endpoint=True
    ).tolist()
    check_data = [31, 59, 82, 101, 84, 67, 50, 33, 20, 16, 20, 33, 50, 67, 84, 101]

    backup = cocotb_test_fixture.get_artifact_dir()
    with temporary_directory(backup):
        cocotb_test_fixture.write(
            {
                "data_in": data_in,
                "check": check_data,
            }
        )
        cocotb_test_fixture.clear_srcs()
        cocotb_test_fixture.add_srcs_from_package("thresholding", "verilog/mov_avg_abs_pow2.v")
        cocotb_test_fixture.set_top_module_name("MOVING_AVERAGE")
        cocotb_test_fixture.run(
            params={
                "BITWIDTH": bitwidth,
                "LENGTH": length,
            },
            defines={},
        )


# --- build test
@pytest.mark.simulation
@pytest.mark.parametrize(
    "bitwidth, length",
    [
        (8, 4),
    ],
)
def test_build(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
    length: int,
):
    backup = cocotb_test_fixture.get_artifact_dir()
    with temporary_directory(backup) as tmpdir:
        build_dir = tmpdir / "verilog"

        load_and_plugin(
            type="mov_avg_abs_pow2",
            id="0",
            params={
                "BITWIDTH": bitwidth,
                "LENGTH": length,
            },
            packages=["thresholding"],
            path2save=build_dir,
        )

        # --- input data
        cnv = int_arithmetic(total_bits=bitwidth, signed=True)
        data_in = np.linspace(
            cnv.minimum_as_integer, cnv.maximum_as_integer, num=16, dtype=int, endpoint=True
        ).tolist()
        check_data = [31, 59, 82, 101, 84, 67, 50, 33, 20, 16, 20, 33, 50, 67, 84, 101]

        cocotb_test_fixture.write(
            {
                "data_in": data_in,
                "check": check_data,
            }
        )

        # start test
        cocotb_test_fixture.set_top_module_name("MOV_AVG_ABS_POW2_0")
        cocotb_test_fixture.clear_srcs()
        cocotb_test_fixture.add_srcs_from_dir(path=tmpdir, glob_pattern="verilog/*.v")
        cocotb_test_fixture.run(
            params={},
            defines={},
        )


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth, length", [(8, 4), (6, 8), (10, 32)])
def test_build_equal(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
    length: int,
):

    id = 1
    cnv = int_arithmetic(total_bits=bitwidth, signed=True)
    data_in = np.array(
        [np.random.randint(cnv.minimum_as_integer, cnv.maximum_as_integer) for _ in range(8 * length)]
    )

    settings = SettingsThreshold(
        method="mavg_abs",
        sampling_rate=1.0,
        window_sec=float(length),
        thr_val=0.0,
        do_quant=True,
    )
    threshold = Thresholding(settings)
    checked_data = threshold.get_threshold(data_in)

    backup = cocotb_test_fixture.get_artifact_dir()
    with temporary_directory(backup) as tmpdir:
        build_dir = tmpdir / "verilog"

        threshold.create_design(
            data=data_in,
            target="fpga",
            bitwidth=bitwidth,
            id=f"{id}",
            signed=True,
            path2save=build_dir,
        )

        cocotb_test_fixture.write(
            {
                "data_in": data_in.tolist(),
                "check": checked_data.tolist(),
            }
        )

        top_module = f"MOV_AVG_ABS_POW2_{id}"
        cocotb_test_fixture.set_top_module_name(top_module)
        cocotb_test_fixture.clear_srcs()
        cocotb_test_fixture.add_srcs_from_dir(path=tmpdir, glob_pattern="verilog/*.v")

        cocotb_test_fixture.run(
            params={},
            defines={},
        )
