import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge
from pathlib import Path
import numpy as np

import pytest #add this
from elasticai.creator.testing import CocotbTestFixture, eai_testbench
from elasticai.creator_plugins.mac import load_and_plugin

from elasticai.creator.testing.cocotb_runner import run_cocotb_sim_for_src_dir
import elasticai.creator_plugins.filter_data as test_dut
from elasticai.preprocessor.thresholding import Thresholding, SettingsThreshold
# from elasticai.creator_plugins.helper import calc_mavg

# --- get signal for template test
def get_template_signal() -> list[int]:
    return [ 0, 0, 0, 0 ]

# --- build test signal
def build_test_signal(bitwidth: int, length: int) -> list[int]:
    return [
        np.random.randint(0, 2**bitwidth - 1)
        for _ in range(length)
    ]

# --- build check data sliding middle value
def calc_mavg_reference_sliding(data, length):
    taps = [0] * length
    pos = 0
    pre_out = 0
    out = []

    for sample in data:
        # FPGA gibt zuerst den alten Wert aus
        out.append(pre_out // length)

        # intern aktualisieren
        pre_out = pre_out - taps[pos] + sample
        taps[pos] = sample
        pos = (pos + 1) % length

    return out

def assert_mov_avg_equivalent(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
    length: int,
    id: int,
    data_in: list[int], ):

    settings = SettingsThreshold(
        method="mavg",
        sampling_rate=1.0,
        gain=1.0,
        window_sec=float(length),
    )

    threshold = Thresholding(settings)

    build_dir = (
        cocotb_test_fixture.get_artifact_dir()
        / "verilog"
    )

    threshold.create_design(
        target="fpga",
        bitwidth=bitwidth,
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

    top_module = f'MOV_AVG_NORM_{id}'
    cocotb_test_fixture.set_top_module_name(top_module)
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")

    cocotb_test_fixture.run(
        params={},
        defines={},
    )


@cocotb.test()
@eai_testbench  #add this
async def filter_fir_mavg_norm_test(
    dut,
    bitwidth: int,
    length: int,
    data_in: list[int],  #new input parameter
    check: list[int],    #check signal
):
    period_clk = 5
    period_data = 100
    do_signed = False

    used_bitwidth = int(dut.BITWIDTH.value)
    used_adrwidth = 4

    # --- data_in_array not needed anymore because of input signal
    # data_in_array = [
    #     np.random.randint(low=0, high=2**used_bitwidth - 1)
    #     for _ in range(used_adrwidth)
    # ]
    # data_in_array = [int(2**(used_bitwidth-1) * (1 + np.cos(2 * np.pi * idx / used_adrwidth))) for idx in range(used_adrwidth)]
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

    # --- check input
    if(len(data_in) != len(check)):
        raise Exception("wrong input data") 

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
@pytest.mark.parametrize("length", [4])
def test_mov_avg_norm(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
    length: int,
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
    cocotb_test_fixture.set_top_module_name("MOVING_AVERAGE")   
    cocotb_test_fixture.run(params={
        "BITWIDTH": bitwidth,
        "LENGTH": length,
    }, 
    defines={}
    )


# --- build test
@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [4])
@pytest.mark.parametrize("length", [4])
def test_mov_avg_norm_build(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
    length: int,
    ):

    # Directory for artifact
    artifact_dir = cocotb_test_fixture.get_artifact_dir()
    build_dir = artifact_dir / "verilog"

    load_and_plugin(
        type="mov_avg_norm",
        id="0",  
        params={
            "BITWIDTH": bitwidth,
            "LENGTH": length,
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
    cocotb_test_fixture.set_top_module_name("MOV_AVG_NORM_0")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")
    cocotb_test_fixture.run(
        params={},
        defines={},
    )

# --- Check equivalence to reference function (build_test_data)
@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [4])
@pytest.mark.parametrize("length", [4])
def test_mov_avg_norm_equal(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
    length: int,
    ):

    id = 1

    data_in = build_test_signal(
        bitwidth=bitwidth,
        length=20,
    )

    assert_mov_avg_equivalent(
        cocotb_test_fixture,
        bitwidth,
        length,
        id,
        data_in)

# --- Check equivalence with extended input
@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [4])
@pytest.mark.parametrize("length", [20])
def test_mov_avg_norm_equal_extended(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
    length: int,
    ):

    id = 2
    num_repeats = 4

    data_in = build_test_signal(
        bitwidth=bitwidth,
        length= length,
    )
    data_in = data_in * num_repeats 

    assert_mov_avg_equivalent(
        cocotb_test_fixture,
        bitwidth,
        length,
        id,
        data_in)    