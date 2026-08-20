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
async def downsampler_mean_access(dut, sig_in: list[int], check: list[int]):
    period = 10
    dut.clk.value = 0
    dut.rst_n.value = 0
    dut.in_valid.value = 0
    dut.din.value = 0

    cocotb.start_soon(Clock(dut.clk, period, unit="ns").start())
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    data_out = []
    for sample in sig_in:
       dut.din.value = sample
       dut.in_valid.value = 1
       await RisingEdge(dut.clk)
       if dut.out_valid.value:
           data_out.append(dut.dout.value.signed_integer)
    await RisingEdge(dut.clk)
    if dut.out_valid.value:
        value = dut.dout.value.signed_integer
        data_out.append(value)
        print(f"Output : {value}")
    dut.in_valid.value = 0
    assert data_out == check


# po2 = power of 2
sig_in_po2 = [1, 2, 3, 4, 5, 6, 7, 8]
check_po2 = [2, 6] # Mittelwerte: (1+2+3+4)/4 = 2, (5+6+7+8)/4 = 6

@pytest.mark.simulation
def test_downsampler_mean_po2(cocotb_test_fixture: CocotbTestFixture):
    cocotb_test_fixture.write({"sig_in": sig_in_po2, "check": check_po2})
    cocotb_test_fixture.set_top_module_name("DOWNSAMPLER_MEAN")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_package("datarate", "verilog/downsampler_mean.v")
    cocotb_test_fixture.run(
        params={"DATA_WIDTH": 16, "DSR":4,}, defines={}
    )

sig_in = [1, 2, 3, 4, 5, 6, 7, 8, 9]
check = [2, 5, 8] # (1+2+3)/3 = 2, (4+5+6)/3 = 5, (7+8+9)/3 = 8

@pytest.mark.simulation
def test_downsampler_mean(cocotb_test_fixture: CocotbTestFixture):
    cocotb_test_fixture.write({"sig_in": sig_in, "check": check})
    cocotb_test_fixture.set_top_module_name("DOWNSAMPLER_MEAN")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_package("datarate", "verilog/downsampler_mean.v")
    cocotb_test_fixture.run(
        params={"DATA_WIDTH": 16, "DSR":3,}, defines={}
    )

@pytest.mark.simulation
def test_downsampler_mean_build(
    cocotb_test_fixture: CocotbTestFixture):
    build_dir = cocotb_test_fixture.get_artifact_dir() / "verilog"

    load_and_plugin(
        type="downsampler_mean",
        id="0",
        params={},
        packages=["datarate"],
        path2save=build_dir,
    )

    cocotb_test_fixture.write({"sig_in": sig_in, "check": check})
    cocotb_test_fixture.set_top_module_name("DOWNSAMPLER_MEAN_0")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")
    cocotb_test_fixture.run(params={"DATA_WIDTH": 16, "DSR":3}, defines={})


@pytest.mark.simulation
def test_downsampler_mean_build_equal(
    cocotb_test_fixture: CocotbTestFixture):
    build_dir = cocotb_test_fixture.get_artifact_dir() / "verilog"
    dut = DownSampling(
        SettingsDownSampling(
            sampling_rate=1000.0,  
            dsr=3,
            type="downsampler_mean",
        )
    )
    
    #Erwarteter Wert aus Python Funktion
    data_checked = (dut.do_simple(  
        uin=np.asarray(sig_in)
    )).tolist()

    load_and_plugin(
        type="downsampler_mean",
        id="1",
        params={},
        packages=["datarate"],
        path2save=build_dir,
    )

    cocotb_test_fixture.write({"sig_in": sig_in, "check": data_checked})
    cocotb_test_fixture.set_top_module_name("DOWNSAMPLER_MEAN_1")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")
    cocotb_test_fixture.run(params={"DATA_WIDTH": 16, "DSR":3}, defines={})


