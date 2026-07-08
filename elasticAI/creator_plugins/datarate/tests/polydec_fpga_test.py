import cocotb
import numpy as np
import pytest
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer
from elasticai.creator.testing import CocotbTestFixture, eai_testbench

from elasticai.creator_plugins.datarate.utils import load_and_plugin
from elasticai.preprocessor.downsampling import DownSampling, SettingsDownSampling
from elasticai.creator.arithmetic import FxpArithmetic, FxpParams


def build_test_signal(
    bitwidth: int, frac: int = 0, num_periods: int = 2, n_samples: int = 22
) -> list:
    arith_data = FxpArithmetic(FxpParams(total_bits=bitwidth, frac_bits=frac, signed=False))
    mid_cm = 2 ** (bitwidth - 1)

    sig_in = mid_cm + (mid_cm - 2) * np.sin(
        np.linspace(0, num_periods * 2 * np.pi, n_samples, dtype=float)
    )
    return [arith_data.cut_as_integer(float(v)) for v in sig_in]


#Feste Werte für den Template-Test
FIXED_SIG_IN = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
FIXED_CHECK = {
    1: [1, 5, 1, 5, 1, 5],   # y_k = x[2k+1] + x[2k]
    2: [1, 7, 3, 7, 3, 7],   # y_k = x[2k+1] + 2*x[2k] + x[2k-2]
}    


@cocotb.test()
@eai_testbench
async def polyphase_access(dut, bitwidth: int, poly_order: int, sig_in: list[int], check: list[int]):
    period_clk = 5
    period_smp = 20

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

    data_out = list()
    for val in sig_in:
        await RisingEdge(dut.CLK_HGH)
        dut.DATA_IN.value = val
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
@pytest.mark.parametrize("bitwidth, poly_order",[(3, 1),(3, 2),],)
def test_filter_polydec_fpga(cocotb_test_fixture: CocotbTestFixture, bitwidth: int, poly_order: int):
    sig_in = FIXED_SIG_IN     # Bitwidth nicht mehr berücksichtigt
    check = FIXED_CHECK[poly_order]

    cocotb_test_fixture.write({"sig_in": sig_in, "check": check})
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_package("datarate", "verilog/polydec_fpga.v")
    cocotb_test_fixture.set_top_module_name("FILTER_POLYDEC_FPGA")
    cocotb_test_fixture.run(params={"BITWIDTH": bitwidth, "POLY_ORDER": poly_order}, defines={})



@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth, poly_order",[(3, 1),],)
def test_filter_polydec_fpga_build_first_order(
    cocotb_test_fixture: CocotbTestFixture, bitwidth: int, poly_order: int
):
    build_dir = cocotb_test_fixture.get_artifact_dir() / "verilog"
    sig_in = FIXED_SIG_IN     # Bitwidth nicht mehr berücksichtigt
    check = FIXED_CHECK[poly_order]

    load_and_plugin(
        type="polydec_fpga",
        id="0",
        params={"BITWIDTH": bitwidth, "POLY_ORDER": poly_order},
        packages=["datarate"],
        path2save=build_dir,
    )

    cocotb_test_fixture.write({"sig_in": sig_in, "check": check})
    cocotb_test_fixture.set_top_module_name("POLYDEC_FPGA_0")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")
    cocotb_test_fixture.run(params={}, defines={})


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth, poly_order",[(3, 2),],)
def test_filter_polydec_fpga_build_second_order(
    cocotb_test_fixture: CocotbTestFixture, bitwidth: int, poly_order: int
):
    build_dir = cocotb_test_fixture.get_artifact_dir() / "verilog"
    sig_in = FIXED_SIG_IN     # Bitwidth nicht mehr berücksichtigt
    check = FIXED_CHECK[poly_order]

    load_and_plugin(
        type="polydec_fpga",
        id="0",
        params={"BITWIDTH": bitwidth, "POLY_ORDER": poly_order},
        packages=["datarate"],
        path2save=build_dir,
    )

    cocotb_test_fixture.write({"sig_in": sig_in, "check": check})
    cocotb_test_fixture.set_top_module_name("POLYDEC_FPGA_0")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")
    cocotb_test_fixture.run(params={}, defines={})



@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth, poly_order", [(3, 1)])
def test_filter_polydec_fpga_build_equal_first_order(
        cocotb_test_fixture: CocotbTestFixture, bitwidth: int, poly_order: int
):
    build_dir = cocotb_test_fixture.get_artifact_dir() / "verilog"
    dut = DownSampling(
        SettingsDownSampling(
            sampling_rate=1000.0,
            dsr=10,
        )
    )
    # Test-Signal
    data_in = build_test_signal(
        bitwidth=bitwidth,
        num_periods=2,
        n_samples=22,
    )

    data_checked = dut.do_decimation_polyphase_order_one(
        uin=np.asarray(data_in)
    ).tolist()

    load_and_plugin(
        type="polydec_fpga",
        id="1",
        params={"BITWIDTH": bitwidth, "POLY_ORDER": poly_order},
        packages=["datarate"],
        path2save=build_dir,
    )

    cocotb_test_fixture.write({"sig_in": data_in, "check": data_checked})
    cocotb_test_fixture.set_top_module_name("POLYDEC_FPGA_1")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")
    cocotb_test_fixture.run(
        params={},
        defines={},
    )


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth, poly_order", [(3, 2)])
def test_filter_polydec_fpga_build_equal_second_order(
        cocotb_test_fixture: CocotbTestFixture, bitwidth: int, poly_order: int
):
    build_dir = cocotb_test_fixture.get_artifact_dir() / "verilog"
    dut = DownSampling(
        SettingsDownSampling(
            sampling_rate=1000.0,
            dsr=10,
        )
    )
    # Test-Signal
    data_in = build_test_signal(
        bitwidth=bitwidth,
        num_periods=2,
        n_samples=22,
    )

    data_checked = dut.do_decimation_polyphase_order_two(
        uin=np.asarray(data_in)
    ).tolist()

    load_and_plugin(
        type="polydec_fpga",
        id="1",
        params={"BITWIDTH": bitwidth, "POLY_ORDER": poly_order},
        packages=["datarate"],
        path2save=build_dir,
    )

    cocotb_test_fixture.write({"sig_in": data_in, "check": data_checked})
    cocotb_test_fixture.set_top_module_name("POLYDEC_FPGA_1")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")
    cocotb_test_fixture.run(
        params={},
        defines={},
    )