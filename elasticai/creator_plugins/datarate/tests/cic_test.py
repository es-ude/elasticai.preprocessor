import random

import cocotb
import numpy as np
import pytest
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer
from elasticai.creator.testing import CocotbTestFixture, eai_testbench

from elasticai.creator_plugins.datarate.utils import load_and_plugin
from elasticai.preprocessor.downsampling import DownSampling, SettingsDownSampling


def build_test_signal(
    bitwidth: int,
    num_samples: int,
    seed: int = 42,
) -> list[int]:
    rng = random.Random(seed)
    valrange = 2**bitwidth
    return [rng.randint(0, valrange - 1) for _ in range(num_samples)]


@cocotb.test()
@eai_testbench
async def cic_access(dut, bitwidth: int, dec_rate: int, n_dec: int, sig_in: list[int], check: list[int]):
    period_clk = 5
    period_smp = 10
    gain_cic = dec_rate**n_dec

    dut.CLK_SYS.value = 0
    dut.CLK_SMP.value = 0
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
    dut.EN.value = 1

    # Apply data and test
    cocotb.start_soon(Clock(dut.CLK_SMP, period_smp, unit="ns").start())

    data_out = list()
    sample_idx = 0
    for out_idx in range(len(check)):
        n_feed = 1 if out_idx == 0 else dec_rate
        for _ in range(n_feed):
            dut.DATA_IN.value = int(sig_in[sample_idx])
            sample_idx += 1
            await RisingEdge(dut.CLK_SMP)

        await FallingEdge(dut.DEC_CLK)
        await Timer(period_clk, unit="ns")
        data_out.append(dut.DATA_OUT.value.to_unsigned())

    ref = [c * gain_cic for c in check]
    if not data_out == ref:
        print(f"INP: ({len(sig_in)}) {sig_in}")
        print(f"OUT: ({len(data_out)}) {data_out}")
        print(f"REF: ({len(check)}) {check}")

    assert data_out == ref


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [2, 6, 8, 12, 16])
@pytest.mark.parametrize("dec_rate", [1])
@pytest.mark.parametrize("n_dec", [2])
def test_filter_cic(cocotb_test_fixture: CocotbTestFixture, bitwidth: int, dec_rate: int, n_dec: int):
    data_in = build_test_signal(
        bitwidth=bitwidth,
        num_samples=20,
    )
    cocotb_test_fixture.write({"sig_in": data_in, "check": data_in})
    cocotb_test_fixture.set_top_module_name("FILTER_CIC")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_package("datarate", "verilog/cic.v")
    cocotb_test_fixture.run(
        params={"BITWIDTH": bitwidth, "DEC_RATE": dec_rate, "N_DEC": n_dec}, defines={}
    )


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [8])
@pytest.mark.parametrize("dec_rate", [1])
@pytest.mark.parametrize("n_dec", [2])
def test_filter_cic_build(
    cocotb_test_fixture: CocotbTestFixture, bitwidth: int, dec_rate: int, n_dec: int
):
    build_dir = cocotb_test_fixture.get_artifact_dir() / "verilog"

    data_in = build_test_signal(
        bitwidth=bitwidth,
        num_samples=20,
    )

    load_and_plugin(
        type="cic",
        id="0",
        params={"BITWIDTH": bitwidth, "DEC_RATE": dec_rate, "N_DEC": n_dec},
        packages=["datarate"],
        path2save=build_dir,
    )
    cocotb_test_fixture.write({"sig_in": data_in, "check": data_in})
    cocotb_test_fixture.set_top_module_name("CIC_0")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")
    cocotb_test_fixture.run(params={}, defines={})


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [8])
@pytest.mark.parametrize("dec_rate", [2])
@pytest.mark.parametrize("n_dec", [2])
def test_filter_cic_build_equal(
    cocotb_test_fixture: CocotbTestFixture, bitwidth: int, dec_rate: int, n_dec: int
):
    build_dir = cocotb_test_fixture.get_artifact_dir() / "verilog"
    dut = DownSampling(
        SettingsDownSampling(
            sampling_rate=1000.0,
            dsr=dec_rate,
        )
    )
    # Test-Signal
    data_in = build_test_signal(
        bitwidth=bitwidth,
        num_samples=20,
    )

    data_checked = dut.do_cic(uin=np.asarray(data_in), num_stages=n_dec).tolist()

    load_and_plugin(
        type="cic",
        id="1",
        params={"BITWIDTH": bitwidth, "DEC_RATE": dec_rate, "N_DEC": n_dec},
        packages=["datarate"],
        path2save=build_dir,
    )
    cocotb_test_fixture.write({"sig_in": data_in, "check": data_checked})
    cocotb_test_fixture.set_top_module_name("CIC_1")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")
    cocotb_test_fixture.run(
        params={},
        defines={},
    )
