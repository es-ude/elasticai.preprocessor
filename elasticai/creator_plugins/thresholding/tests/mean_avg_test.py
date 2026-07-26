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
from elasticai.preprocessor.thresholding import Thresholding, SettingsThreshold

# --- get signal for template test
def get_template_signal() -> list[int]:
    return [ 0, 0, 0, 0 ]

# --- build test signal
def build_test_signal(bitwidth: int, length: int) -> list[int]:
    return [
        np.random.randint(0, 2**bitwidth - 1)
        for _ in range(length)
    ]


def assert_mean_avg_equivalent(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
    countwidth: int,
    id: int,
    data_in: list[int], ):

    module_type = "mean_avg"

    settings = SettingsThreshold(
        method="abs_mean",
        module_type = module_type,
        sampling_rate=1.0,
        gain=1.0,
        window_sec=float(len(data_in)),
    )

    threshold = Thresholding(settings)

    build_dir = (
        cocotb_test_fixture.get_artifact_dir()
        / "verilog"
    )

    threshold.create_design(
        target="fpga",
        bitwidth=bitwidth,
        countwidth = countwidth,
        id=f'{id}',
        path2save=build_dir,
    )

    
    check = threshold.get_threshold_list(data_in)

    cocotb_test_fixture.write(
        {
            "data_in": data_in,
            "check": check,
        }
    )

    top_module_name = f'{module_type.upper()}_{id}'
    cocotb_test_fixture.set_top_module_name(top_module_name)
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")

    cocotb_test_fixture.run(
        params={},
        defines={},
    )

# --- Testbench -------------------------------------
@cocotb.test()
@eai_testbench
async def mean_avg_test(
    dut,
    bitwidth: int,
    countwidth: int,
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

    assert len(data_in) == len(check)

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
            "IN =", val,
            "EXPECTED =", expected,
            "OUT =", int(dut.DATA_OUT.value),                
        )
        assert int(dut.DATA_OUT.value) == int(expected)


# --- template test
@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [8])
@pytest.mark.parametrize("countwidth", [8])
def test_mean_avg(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
    countwidth: int,
	):

    # --- Build test data
    data_in = get_template_signal()

    check_data = data_in

    cocotb_test_fixture.write(
        {
            "data_in": data_in,
            "check": check_data,
        }
    )
    cocotb_test_fixture.clear_srcs()    #modul sources werden frei gegeben um neu geladen zu werden
    cocotb_test_fixture.add_srcs_from_package("thresholding","verilog/*.v")
    cocotb_test_fixture.set_top_module_name("MEAN_AVERAGE")   
    cocotb_test_fixture.run(params={
        "BITWIDTH": bitwidth,
        "COUNTWIDTH": countwidth,
    }, 
    defines={}
    )

# --- build test
@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [4])
@pytest.mark.parametrize("countwidth", [8])
def test_mean_avg_build(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
    countwidth: int,
    ):

    # Directory for artifact
    artifact_dir = cocotb_test_fixture.get_artifact_dir()
    build_dir = artifact_dir / "verilog"

    load_and_plugin(
        type="mean_avg",
        id="0",  
        params={
            "BITWIDTH": bitwidth,
            "COUNTWIDTH": countwidth,
        },
        packages=["thresholding"],
        path2save=build_dir,
    )

    # --- input data
    data_in = get_template_signal()
    check = data_in

    cocotb_test_fixture.write(
        {
            "data_in": data_in,
            "check": check,
        }
    )

    #start test
    cocotb_test_fixture.set_top_module_name("MEAN_AVG_0")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")
    cocotb_test_fixture.run(
        params={},
        defines={},
    )

# --- Check equivalence to reference function
@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [4])
@pytest.mark.parametrize("countwidth", [8])
def test_mean_avg_equal(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
    countwidth: int,
):
    id = 1

    length = 16

    data_in = build_test_signal(bitwidth, length)

    assert_mean_avg_equivalent(
        cocotb_test_fixture,
        bitwidth,
        countwidth,
        id,
        data_in)