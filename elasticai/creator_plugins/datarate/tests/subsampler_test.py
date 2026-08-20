import cocotb
import pytest
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
from elasticai.creator.testing import CocotbTestFixture, eai_testbench
from elasticai.creator_plugins.datarate.utils import load_and_plugin
from elasticai.preprocessor.downsampling import DownSampling, SettingsDownSampling

@cocotb.test()
@eai_testbench
async def subsampler_access(dut, sig_in: list[int], check: list[int]):
    period = 10
    dut.clk.value = 0
    dut.rst_n.value = 0
    dut.in_valid.value = 0
    dut.in_data.value = 0

    cocotb.start_soon(Clock(dut.clk, period, unit="ns").start())
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    data_out = []
    for sample in sig_in:
        dut.in_data.value = sample
        dut.in_valid.value = 1
        await RisingEdge(dut.clk)
        if dut.out_valid.value:
            data_out.append(dut.out_data.value.signed_integer)
    await RisingEdge(dut.clk)
    if dut.out_valid.value:
        value = dut.out_data.value.signed_integer
        data_out.append(value)
        print(f"Output : {value}")

    dut.in_valid.value = 0
    assert data_out == check


sig_in = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
check  = [4, 8, 12, 16]  # N = 4

@pytest.mark.simulation
def test_subsampler(cocotb_test_fixture: CocotbTestFixture):
    cocotb_test_fixture.write({"sig_in": sig_in, "check": check})
    cocotb_test_fixture.set_top_module_name("SUBSAMPLER")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_package("datarate", "verilog/subsampler.v")
    cocotb_test_fixture.run(
        params={"DATA_WIDTH": 16, "N":4,}, defines={}
    )

@pytest.mark.simulation
def test_subsampler_build(
    cocotb_test_fixture: CocotbTestFixture):
    build_dir = cocotb_test_fixture.get_artifact_dir() / "verilog"

    load_and_plugin(
        type="subsampler",
        id="0",
        params={"DATA_WIDTH": 16, "N":4},
        packages=["datarate"],
        path2save=build_dir,
    )

    cocotb_test_fixture.write({"sig_in": sig_in, "check": check})
    cocotb_test_fixture.set_top_module_name("SUBSAMPLER_0")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")
    cocotb_test_fixture.run(params={}, defines={})

@pytest.mark.simulation
def test_subsampler_build_equal(
    cocotb_test_fixture: CocotbTestFixture):
    build_dir = cocotb_test_fixture.get_artifact_dir() / "verilog"
    dut = DownSampling(
        SettingsDownSampling(
            sampling_rate=1000.0,  
            dsr=4,
            type="subsampler",
        )
    )
    
    #Erwarteter Wert aus Python Funktion
    data_checked = (dut.do_subsampling(  
        data=np.asarray(sig_in)
    )).tolist()

    load_and_plugin(
        type="subsampler",
        id="1",
        params={"DATA_WIDTH": 16, "N":4},
        packages=["datarate"],
        path2save=build_dir,
    )

    cocotb_test_fixture.write({"sig_in": sig_in, "check": data_checked})
    cocotb_test_fixture.set_top_module_name("SUBSAMPLER_1")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")
    cocotb_test_fixture.run(params={}, defines={})