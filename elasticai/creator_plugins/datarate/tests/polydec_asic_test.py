import cocotb
import numpy as np
import pytest
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer
from elasticai.creator.arithmetic import FxpArithmetic, FxpParams
from elasticai.creator.testing import CocotbTestFixture, eai_testbench

from elasticai.creator_plugins.datarate.utils import load_and_plugin
from elasticai.preprocessor.downsampling import DownSampling, SettingsDownSampling


def build_test_signal(bitwidth: int, frac: int = 0, num_periods: int = 2, n_samples: int = 22) -> list:
    arith_data = FxpArithmetic(FxpParams(total_bits=bitwidth, frac_bits=frac, signed=False))
    mid_cm = 2 ** (bitwidth - 1)

    sig_in = mid_cm + (mid_cm - 2) * np.sin(
        np.linspace(0, num_periods * 2 * np.pi, n_samples, dtype=float)
    )
    return [arith_data.cut_as_integer(float(v)) for v in sig_in]


# Feste Werte für den Template-Test
FIXED_SIG_IN = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
FIXED_CHECK = {
    1: [1, 5, 1, 5, 1, 5],  # y_k = x[2k+1] + x[2k]
    2: [1, 7, 3, 7, 3, 7],  # y_k = x[2k+1] + 2*x[2k] + x[2k-2]
}


@cocotb.test()
@eai_testbench
async def polyphase_access(dut, bitwidth: int, poly_order: int, sig_in: list[int], check: list[int]):
    period_smp = 10
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

    data_out = list()
    for val in sig_in:
        dut.DATA_IN.value = int(val)
        await RisingEdge(dut.CLK_HGH)
        if dut.CLK_LOW.value:
            data_out.append(dut.DATA_OUT.value.to_unsigned())

    await RisingEdge(dut.CLK_LOW)
    await Timer(1, unit="ns")
    data_out.append(dut.DATA_OUT.value.to_unsigned())
    await FallingEdge(dut.CLK_LOW)

    if not data_out[1:] == check:
        print(f"INP: ({len(sig_in)}) {sig_in}")
        print(f"OUT: ({len(data_out)}) {data_out}")
        print(f"REF: ({len(check)}) {check}")
    assert data_out[1:] == check


@pytest.mark.simulation
@pytest.mark.parametrize(
    "bitwidth, poly_order",
    [
        (3, 1),
        (3, 2),
    ],
)
def test_filter_polydec_asic(cocotb_test_fixture: CocotbTestFixture, poly_order: int, bitwidth: int):
    sig_in = FIXED_SIG_IN  # Bitwidth nicht mehr berücksichtigt
    check = FIXED_CHECK[poly_order]

    cocotb_test_fixture.write({"sig_in": sig_in, "check": check})
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_package("datarate", "verilog/polydec_asic.v")
    cocotb_test_fixture.set_top_module_name("FILTER_POLYDEC_ASIC")
    cocotb_test_fixture.run(params={"BITWIDTH": bitwidth, "POLY_ORDER": poly_order}, defines={})


@pytest.mark.simulation
@pytest.mark.parametrize(
    "bitwidth, poly_order",
    [
        (3, 1),
    ],
)
def test_filter_polydec_asic_build_first_order(
    cocotb_test_fixture: CocotbTestFixture, poly_order: int, bitwidth: int
):
    build_dir = cocotb_test_fixture.get_artifact_dir() / "verilog"
    sig_in = FIXED_SIG_IN  # Bitwidth nicht mehr berücksichtigt
    check = FIXED_CHECK[poly_order]

    load_and_plugin(
        type="polydec_asic",
        id="0",
        params={"BITWIDTH": bitwidth, "POLY_ORDER": poly_order},
        packages=["datarate"],
        path2save=build_dir,
    )

    cocotb_test_fixture.write({"sig_in": sig_in, "check": check})
    cocotb_test_fixture.set_top_module_name("POLYDEC_ASIC_0")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")
    cocotb_test_fixture.run(params={}, defines={})


@pytest.mark.simulation
@pytest.mark.parametrize(
    "bitwidth, poly_order",
    [
        (3, 2),
    ],
)
def test_filter_polydec_asic_build_second_order(
    cocotb_test_fixture: CocotbTestFixture, poly_order: int, bitwidth: int
):
    build_dir = cocotb_test_fixture.get_artifact_dir() / "verilog"
    sig_in = FIXED_SIG_IN  # Bitwidth nicht mehr berücksichtigt
    check = FIXED_CHECK[poly_order]

    load_and_plugin(
        type="polydec_asic",
        id="0",
        params={"BITWIDTH": bitwidth, "POLY_ORDER": poly_order},
        packages=["datarate"],
        path2save=build_dir,
    )

    cocotb_test_fixture.write({"sig_in": sig_in, "check": check})
    cocotb_test_fixture.set_top_module_name("POLYDEC_ASIC_0")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")
    cocotb_test_fixture.run(params={}, defines={})


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth, poly_order", [(3, 1)])
def test_filter_polydec_asic_build_equal_first_order(
    cocotb_test_fixture: CocotbTestFixture, bitwidth: int, poly_order: int
):
    build_dir = cocotb_test_fixture.get_artifact_dir() / "verilog"
    dut = DownSampling(
        SettingsDownSampling(
            sampling_rate=1000.0,  # Default Settings
            dsr=10,
        )
    )
    # Test-Signal
    data_in = build_test_signal(
        bitwidth=bitwidth,
        num_periods=2,
        n_samples=22,
    )

    # Erwarteter Wert aus Python Funktion
    data_checked = (dut._do_decimation_polyphase_order_one(uin=np.asarray(data_in))).tolist()

    load_and_plugin(
        type="polydec_asic",
        id="1",
        params={"BITWIDTH": bitwidth, "POLY_ORDER": poly_order},
        packages=["datarate"],
        path2save=build_dir,
    )

    cocotb_test_fixture.write({"sig_in": data_in, "check": data_checked})
    cocotb_test_fixture.set_top_module_name("POLYDEC_ASIC_1")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")
    cocotb_test_fixture.run(
        params={},
        defines={},
    )


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth, poly_order", [(3, 2)])
def test_filter_polydec_asic_build_equal_second_order(
    cocotb_test_fixture: CocotbTestFixture, bitwidth: int, poly_order: int
):
    build_dir = cocotb_test_fixture.get_artifact_dir() / "verilog"
    dut = DownSampling(
        SettingsDownSampling(
            sampling_rate=1000.0,  # Default Settings
            dsr=10,
        )
    )
    # Test-Signal
    data_in = build_test_signal(
        bitwidth=bitwidth,
        num_periods=2,
        n_samples=22,
    )

    # Erwarteter Wert aus Python Funktion
    data_checked = (dut._do_decimation_polyphase_order_two(uin=np.asarray(data_in))).tolist()

    load_and_plugin(
        type="polydec_asic",
        id="1",
        params={"BITWIDTH": bitwidth, "POLY_ORDER": poly_order},
        packages=["datarate"],
        path2save=build_dir,
    )

    cocotb_test_fixture.write({"sig_in": data_in, "check": data_checked})
    cocotb_test_fixture.set_top_module_name("POLYDEC_ASIC_1")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")
    cocotb_test_fixture.run(
        params={},
        defines={},
    )
