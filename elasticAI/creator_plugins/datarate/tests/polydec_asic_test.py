import cocotb
import numpy as np
import pytest
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, Timer
from elasticai.creator.testing import CocotbTestFixture, eai_testbench

from elasticai.creator_plugins.datarate.utils import load_and_plugin
from elasticai.preprocessor.downsampling import DownSampling, SettingsDownSampling
from elasticai.creator.arithmetic import FxpArithmetic, FxpParams

# Änderungen:
# externe build_test_signal Funktion
# umgestellt auf externen Input von Eingangssignal und Check Wert
# Umstellung des der Pytests auf Vorgabe über cocotb_test_fixture.write()
# Ich gebe das gleiche generierte Signal in die Testbench und in die Äquivalenzfunktion


def build_test_signal(bitwidth: int, num_periods: int = 2, n_samples: int = 22) -> list: # Vorher in cocotb-Testbench, Signal geht sowohl in TB als auch in Python_Funktion (Äquivalenz)
    mid_cm = 2 ** (bitwidth - 1)

    sig_in = mid_cm + (mid_cm - 2) * np.sin(
        np.linspace(0, num_periods * 2 * np.pi, n_samples, dtype=float)
    )
    return sig_in.astype(int).tolist()


@cocotb.test()
@eai_testbench
async def polyphase_access(dut, bitwidth: int, poly_order: int, sig_in: list[int], check: list[int]): # Neu: externes testsignal und Check Wert
    period_smp = 10
    gain_cic = 2**poly_order

    dut.CLK_HGH.value = 0
    dut.RSTN.value = 1
    dut.EN.value = 0
    dut.DATA_IN.value = 0

    # Start clock and make reset
    await Timer(4 * period_smp, unit="ns")
    for idx in range(4):
        await Timer(4 * period_smp, unit="ns")
        dut.RSTN.value = idx % 2
    await Timer(4 * period_smp, unit="ns")
    dut.RSTN.value = 1
    await Timer(4 * period_smp, unit="ns")

    # Apply data and test
    dut.DATA_IN.value = int(sig_in[0])
    dut.EN.value = 1
    cocotb.start_soon(Clock(dut.CLK_HGH, period_smp, unit="ns").start())
 
    for val, expected in zip(sig_in, check):
        dut.DATA_IN.value = int(val)

        await FallingEdge(dut.CLK_LOW)
        if poly_order > 0:
            await FallingEdge(dut.CLK_LOW)
        if poly_order > 1:
            await FallingEdge(dut.CLK_LOW)   
        assert abs(int(dut.DATA_OUT.value) - int(expected) * gain_cic) <= 1 
        

#  --------------- (1) Template Test ------------------
# Funktioniert das Template überhaupt? Läuft es durch?

@pytest.mark.simulation
@pytest.mark.parametrize(
    "bitwidth, poly_order",
    [(1, 0), (1, 2), (2, 1), (2, 3), (4, 0), (4, 2), (8, 1), (8, 2), (12, 1), (12, 3), (16, 1), (16, 2)],
)
def test_filter_polydec_asic(cocotb_test_fixture: CocotbTestFixture, bitwidth: int, poly_order: int):
    data_in = build_test_signal(
        bitwidth=bitwidth,
        num_periods=2,
        n_samples=22,
    )
    cocotb_test_fixture.write({"sig_in": data_in, "check": data_in}) # Hier check = Eingangssignal
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_package("datarate", "verilog/polydec_asic.v")
    cocotb_test_fixture.set_top_module_name("FILTER_POLYDEC_ASIC")
    cocotb_test_fixture.run(params={"BITWIDTH": bitwidth, "POLY_ORDER": poly_order}, defines={})


#  --------------- (2) Build Test ------------------

@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth, poly_order", [(4, 2)])
def test_filter_polydec_asic_build(
    cocotb_test_fixture: CocotbTestFixture, bitwidth: int, poly_order: int
):
    build_dir = cocotb_test_fixture.get_artifact_dir() / "verilog"

    data_in = build_test_signal(
        bitwidth=bitwidth,
        num_periods=2,
        n_samples=22,
    )

    load_and_plugin(
        type="polydec_asic",
        id="0",
        params={"BITWIDTH": bitwidth, "POLY_ORDER": poly_order},
        packages=["datarate"],
        path2save=build_dir,
    )
    cocotb_test_fixture.write({"sig_in": data_in, "check": data_in})
    cocotb_test_fixture.set_top_module_name("POLYDEC_ASIC_0")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")
    cocotb_test_fixture.run(params={}, defines={})


#  --------------- (3) Äquivalenz Test ------------------

@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth, poly_order", [(3, 2)])
def test_filter_polydec_asic_build_equal(
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

    load_and_plugin(
        type="polydec_asic",
        id="1",
        params={"BITWIDTH": bitwidth, "POLY_ORDER": poly_order},
        packages=["datarate"],
        path2save=build_dir,
    )
    
    cocotb_test_fixture.write({"sig_in": data_in, "check": data_checked}) # Eingangsdaten sind Test-Daten, Erwarteter Output ist Ergebnis der Python-Funktion (Fehler bei assert in Testbench)
    cocotb_test_fixture.set_top_module_name("POLYDEC_ASIC_1")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")
    cocotb_test_fixture.run(
        params={},
        defines={},
    )

