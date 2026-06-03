import cocotb
import numpy as np
import pytest
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge
from cocotb.utils import get_sim_time
from elasticai.creator.arithmetic import FxpArithmetic, FxpParams
from elasticai.creator.testing import CocotbTestFixture, eai_testbench

from elasticai.creator_plugins.filter_data.utils import load_and_plugin
from elasticai.preprocessor.filter import Filtering, SettingsFilter


def build_testdata(
    bitwidth: int, frac: int, order: int, taps_signal: int = 10, num_repeats: int = 1
) -> list:
    arith_data = FxpArithmetic(FxpParams(total_bits=bitwidth, frac_bits=frac, signed=True))

    used_taps = taps_signal * order
    waveform = [
        2 ** (bitwidth - frac) * 0.495 * np.cos(2 * np.pi * idx / used_taps) for idx in range(used_taps)
    ]
    data = list()
    for _ in range(num_repeats):
        data.extend(waveform)
    return arith_data.cut_as_integer(data)


def plot_results(data_in, data_out, data_check):
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots()
    ax1.plot(data_in, label="data_in", marker=".", markersize=4)
    ax1.plot(data_out, label="filter_out", marker=".", markersize=4)
    ax1.plot(data_check, label="sim_out", marker=".", markersize=4)
    ax1.set_xlabel("Value")
    ax1.set_ylabel("y1")
    plt.legend()

    ax2 = ax1.twinx()
    ax2.plot(
        np.asarray(data_out) - np.asarray(data_check),
        label="error",
        marker=".",
        markersize=4,
        color="red",
    )
    ax2.set_ylabel("MAE", color="red")

    plt.grid(True)
    plt.show()


@cocotb.test()
@eai_testbench
async def fir_simple_low(dut, bitwidth: int, fracwidth: int, data: list[int], check: list[int]):
    period_clk = 5

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
    for _ in range(2):
        await RisingEdge(dut.CLK_SYS)
    await FallingEdge(dut.CLK_SYS)

    # Set Data on Trigger
    dut.EN.value = 1
    for _ in range(8):
        await RisingEdge(dut.CLK_SYS)
    assert dut.DVALID.value == 0

    data_out = list()
    for val in data:
        dut.DATA_IN.value = val
        dut.DO_CALC.value = 1
        t1 = get_sim_time("ns")
        await RisingEdge(dut.CLK_SYS)
        dut.DO_CALC.value = 0

        await FallingEdge(dut.CLK_SYS)
        assert dut.DVALID.value == 0

        await RisingEdge(dut.DVALID)
        assert dut.DVALID.value == 1
        t2 = get_sim_time("ns")

        runtime = t2 - t1
        assert runtime == period_clk * (1 + 1)
        data_out.append(dut.DATA_OUT.value.to_signed())
        await RisingEdge(dut.CLK_SYS)

    # plot_results(data_in=data, data_out=data_out, data_check=check)
    error = sum([abs(a - b) for a, b in zip(data_out, check)]) / len(data_out)
    assert error <= 0.25


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth, fracwidth", [(8, 7), (12, 4), (8, 4)])
def test_filter_fir_low(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
    fracwidth: int,
):
    dut = Filtering(
        SettingsFilter(
            gain=1.0,
            fs=2e3,
            n_order=1,
            f_filt=[0.99 * 1e3],
            type="fir",
            f_type="butter",
            b_type="simple_low",
        )
    )
    data_in = build_testdata(bitwidth=bitwidth, frac=fracwidth, order=1, taps_signal=10, num_repeats=10)

    arith_data = FxpArithmetic(FxpParams(total_bits=bitwidth, frac_bits=fracwidth, signed=True))
    data_check = dut.filt_quantized(
        np.asarray(data_in) * 2 ** (-fracwidth),
        total_bitwidth=bitwidth,
        fraction_width=fracwidth,
        is_signed=True,
    ).tolist()
    data_check = arith_data.cut_as_integer(data_check)

    cocotb_test_fixture.write({"data": data_in, "check": data_check})
    cocotb_test_fixture.set_top_module_name("FIR_SIMPLE_LOW")
    cocotb_test_fixture.run(
        params={"BITWIDTH": bitwidth},
        defines={},
    )


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth, fracwidth", [(8, 7)])
def test_filter_fir_low_build(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
    fracwidth: int,
):
    load_and_plugin(
        type="fir_low",
        id="0",
        params={
            "BITWIDTH": bitwidth,
        },
        packages=["filter_data"],
        path2save=cocotb_test_fixture.get_artifact_dir() / "verilog",
        add_ringbuffer=False,
        add_mac=False,
        use_dsp_mult=False,
    )

    dut = Filtering(
        SettingsFilter(
            gain=1.0,
            fs=2e3,
            n_order=1,
            f_filt=[0.99 * 1e3],
            type="fir",
            f_type="butter",
            b_type="simple_low",
        )
    )
    data_in = build_testdata(bitwidth=bitwidth, frac=fracwidth, order=1, taps_signal=10, num_repeats=10)

    arith_data = FxpArithmetic(FxpParams(total_bits=bitwidth, frac_bits=fracwidth, signed=True))
    data_check = dut.filt_quantized(
        np.asarray(data_in) * 2 ** (-fracwidth),
        total_bitwidth=bitwidth,
        fraction_width=fracwidth,
        is_signed=True,
    ).tolist()
    data_check = arith_data.cut_as_integer(data_check)

    cocotb_test_fixture.write({"data": data_in, "check": data_check})
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")
    cocotb_test_fixture.set_top_module_name("FIR_LOW_0")
    cocotb_test_fixture.run(
        params={},
        defines={},
    )
