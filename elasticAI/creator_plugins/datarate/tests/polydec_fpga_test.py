import cocotb
import numpy as np
import pytest
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer
from elasticai.creator.testing import CocotbTestFixture, eai_testbench

from elasticai.creator_plugins.datarate.utils import load_and_plugin
from elasticai.preprocessor.downsampling import DownSampling, SettingsDownSampling
from elasticai.creator.arithmetic import FxpArithmetic, FxpParams


def build_test_signal(bitwidth: int, num_periods: int = 2, n_samples: int = 22) -> list: 
    mid_cm = 2 ** (bitwidth - 1)

    sig_in = mid_cm + (mid_cm - 2) * np.sin(
        np.linspace(0, num_periods * 2 * np.pi, n_samples, dtype=float)
    )
    return sig_in.astype(int).tolist()


@cocotb.test()
@eai_testbench
async def polyphase_access(dut, bitwidth: int, poly_order: int, sig_in: list[int], check: list[int]):
    period_clk = 5
    period_smp = 10
    gain_cic = 2**poly_order

    dut.CLK_SYS.value = 0
    dut.CLK_HGH.value = 0
    dut.RSTN.value = 1
    dut.EN.value = 0
    dut.DATA_IN.value = 0

    # Start clock and make reset
    assert period_clk <= period_smp
    cocotb.start_soon(Clock(dut.CLK_SYS, period_clk, unit="ns").start())
    await Timer(4 * period_clk, unit="ns")
    for idx in range(4):
        await RisingEdge(dut.CLK_SYS)
        dut.RSTN.value = idx % 2
    await RisingEdge(dut.CLK_SYS)
    dut.RSTN.value = 1
    await Timer(4 * period_clk, unit="ns")

    # Apply data and test
    dut.DATA_IN.value = int(sig_in[0])
    dut.EN.value = 1
    cocotb.start_soon(Clock(dut.CLK_HGH, period_smp, unit="ns").start())
    errors = []
    for val, expected in zip(sig_in, check):
        dut.DATA_IN.value = int(val)

        await FallingEdge(dut.CLK_LOW)
        if poly_order > 0:
            await FallingEdge(dut.CLK_LOW)
        if poly_order > 1:
            await FallingEdge(dut.CLK_LOW)
        input = int(dut.DATA_IN.value)    
        actual = int(dut.DATA_OUT.value)
        target = int(expected) * gain_cic 
        error = abs(actual - target)
        errors.append(error)
        assert error <= 1, (
            f"input={val:4d}, DATA_IN={input:4d}, DATA_OUT={actual:4d}, expected={expected:4d}, erwartet={target:4d}, Fehler={error:4d}, check={check[0]:4d}"
        )
    print(errors)
        

@pytest.mark.simulation
@pytest.mark.parametrize(
    "bitwidth, poly_order",
    [
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 0),
        (2, 1),
        (2, 2),
        (2, 3),
        (4, 0),
        (4, 2),
        (8, 0),
        (8, 3),
        (12, 1),
        (12, 2),
        (16, 0),
        (16, 3),
    ],
)
def test_filter_polydec_fpga(cocotb_test_fixture: CocotbTestFixture, bitwidth: int, poly_order: int):
    data_in = build_test_signal(
        bitwidth=bitwidth,
        num_periods=2,
        n_samples=22,
    )
    cocotb_test_fixture.write({"sig_in": data_in, "check": data_in})
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_package("datarate", "verilog/polydec_fpga.v")
    cocotb_test_fixture.set_top_module_name("FILTER_POLYDEC_FPGA")
    cocotb_test_fixture.run(params={"BITWIDTH": bitwidth, "POLY_ORDER": poly_order}, defines={})


@pytest.mark.simulation
@pytest.mark.parametrize(
    "bitwidth, poly_order",
    [
        (4, 2),
        (16, 3),
    ],
)
def test_filter_polydec_fpga_build(
    cocotb_test_fixture: CocotbTestFixture, bitwidth: int, poly_order: int
):
    build_dir = cocotb_test_fixture.get_artifact_dir() / "verilog"

    data_in = build_test_signal(
        bitwidth=bitwidth,
        num_periods=2,
        n_samples=22,
    )

    load_and_plugin(
        type="polydec_fpga",
        id="0",
        params={"BITWIDTH": bitwidth, "POLY_ORDER": poly_order},
        packages=["datarate"],
        path2save=build_dir,
    )
    cocotb_test_fixture.write({"sig_in": data_in, "check": data_in})
    cocotb_test_fixture.set_top_module_name("POLYDEC_FPGA_0")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")
    cocotb_test_fixture.run(params={}, defines={})


#  --------------- (3) Äquivalenz Test ------------------

@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth, poly_order", [(3, 2)])
def test_filter_polydec_fpga_build_equal(
        cocotb_test_fixture: CocotbTestFixture, bitwidth: int, poly_order: int
):
    build_dir = cocotb_test_fixture.get_artifact_dir() / "verilog"
    dut = DownSampling(
        SettingsDownSampling(
            sampling_rate=1000.0, # Default Settings
            dsr=10,
        )
    )
    # Test-Signal
    data_in = build_test_signal(
        bitwidth=bitwidth,
        num_periods=2,
        n_samples=22,
    )
    
    #Erwarteter Wert aus Python Funktion
    data_checked = (dut.do_decimation_polyphase_order_two(  # sind momentan nicht äquivalent
        uin=data_in
    )).tolist()
    print("Check-Ausgangsdaten:", data_checked)

    load_and_plugin(
        type="polydec_fpga",
        id="1",
        params={"BITWIDTH": bitwidth, "POLY_ORDER": poly_order},
        packages=["datarate"],
        path2save=build_dir,
    )
    
    cocotb_test_fixture.write({"sig_in": data_in, "check": data_checked}) # Eingangsdaten sind Test-Daten, Erwarteter Output ist Ergebnis der Python-Funktion (Fehler bei assert in Testbench)
    cocotb_test_fixture.set_top_module_name("POLYDEC_FPGA_1")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")
    cocotb_test_fixture.run(
        params={},
        defines={},
    )