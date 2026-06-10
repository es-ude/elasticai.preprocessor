from math import ceil

import cocotb
import elasticai.creator_plugins.mac as mac
import elasticai.creator_plugins.multipliers as mult
import numpy as np
import pytest
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge
from cocotb.utils import get_sim_time
from elasticai.creator.arithmetic import FxpArithmetic, FxpParams
from elasticai.creator.testing import CocotbTestFixture, eai_testbench

import elasticai.creator_plugins.windower as windower
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
async def filter_biquad(
    dut,
    bitwidth: int,
    fracwidth: int,
    order: int,
    num_mult: int,
    data: list[int],
    check: list[int],
):
    period_clk = 5
    # --- Control signals
    dut.CLK_SYS.value = 0
    dut.RSTN.value = 0
    dut.EN.value = 0
    dut.DO_CALC.value = 0
    dut.DATA_IN.value = 0
    # --- Start clock and making reset
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

    await RisingEdge(dut.CLK_SYS)
    assert dut.DVALID.value == 0

    data_out = list()
    for val in data:
        dut.DATA_IN.value = val
        dut.DO_CALC.value = 1
        t1 = get_sim_time("ns")
        await RisingEdge(dut.CLK_SYS)
        dut.DO_CALC.value = 0
        await RisingEdge(dut.CLK_SYS)

        await RisingEdge(dut.DVALID)
        assert dut.DVALID.value == 1
        t2 = get_sim_time("ns")

        await RisingEdge(dut.CLK_SYS)
        run_time = t2 - t1
        assert run_time in [
            period_clk * (ceil(5 / num_mult) + 5),
            period_clk * (ceil(5 / num_mult) + 3),
            period_clk * (ceil(5 / num_mult) + 2),
        ]
        data_out.append(dut.DATA_OUT.value.to_signed())
        await RisingEdge(dut.CLK_SYS)

    if not data == check:
        # plot_results(data_in=data, data_out=data_out[1:], data_check=check[:-1])
        error = sum([abs(a - b) for a, b in zip(data_out[1:], check[:-1])]) / len(data_out)
        print(error)
        assert error < 0.75


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth, fracwidth", [(8, 7)])
@pytest.mark.parametrize("order", [2])
@pytest.mark.parametrize("num_mult", [1, 2, 3])
def test_filter_biquad_df1(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
    fracwidth: int,
    order: int,
    num_mult: int,
):
    data_in = build_testdata(
        bitwidth=bitwidth, frac=fracwidth, order=order, taps_signal=20, num_repeats=10
    )

    cocotb_test_fixture.write({"data": data_in, "check": data_in})
    cocotb_test_fixture.set_top_module_name("BIQUAD_DF1")
    # cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_package(mac, "verilog/mac.v")
    cocotb_test_fixture.add_srcs_from_package(mult, "verilog/mult_dsp_signed.v")
    cocotb_test_fixture.add_srcs_from_package(windower, "verilog/ring_buffer.v")
    cocotb_test_fixture.run(
        params={
            "BITWIDTH": bitwidth,
            "NUM_MULT": num_mult,
        },
        defines={},
    )


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth, fracwidth", [(8, 4)])
@pytest.mark.parametrize("order", [2])
@pytest.mark.parametrize("num_mult", [1])
def test_filter_biquad_df1_build(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
    fracwidth: int,
    order: int,
    num_mult: int,
):
    dut = Filtering(
        SettingsFilter(
            gain=1.0,
            fs=2e3,
            n_order=order,
            f_filt=[50],
            type="iir",
            f_type="butter",
            b_type="lowpass",
        )
    )
    iir_params = dut.get_coeffs_verilog_string(bitwidth=bitwidth, only_half_fir=False)
    data_in = build_testdata(bitwidth=bitwidth, frac=fracwidth, order=order, taps_signal=4, num_repeats=1)

    load_and_plugin(
        type="biquad_df1",
        id="0",
        params={
            "BITWIDTH": bitwidth,
            "NUM_MULT": num_mult,
            "LENGTH": 5,
            "FILT_COEFFS": iir_params,
        },
        packages=["filter_data"],
        path2save=cocotb_test_fixture.get_artifact_dir() / "verilog",
        add_ringbuffer=False,
        use_dsp_mult=True,
    )

    cocotb_test_fixture.write({"data": data_in, "check": data_in})
    cocotb_test_fixture.set_top_module_name("BIQUAD_DF1_0")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")
    cocotb_test_fixture.run(
        params={},
        defines={},
    )


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth, fracwidth", [(12, 7), (8, 4)])
@pytest.mark.parametrize("order", [2])
@pytest.mark.parametrize("num_mult", [1])
def test_filter_biquad_df1_build_equal(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
    fracwidth: int,
    order: int,
    num_mult: int,
):
    dut = Filtering(
        SettingsFilter(
            gain=1.0,
            fs=2e3,
            n_order=order,
            f_filt=[150],
            type="iir",
            f_type="butter",
            b_type="lowpass",
        )
    )
    fir_params = dut.get_coeffs_verilog_string(bitwidth=bitwidth, only_half_fir=False)
    data_in = build_testdata(
        bitwidth=bitwidth, frac=fracwidth, order=order, taps_signal=40, num_repeats=4
    )

    arith_data = FxpArithmetic(FxpParams(total_bits=bitwidth, frac_bits=fracwidth, signed=True))
    data_checked = dut.filt_quantized(
        xin=np.asarray(data_in) * arith_data._config.minimum_step_as_rational,
        total_bitwidth=bitwidth,
        fraction_width=fracwidth,
        is_signed=True,
    ).tolist()
    data_checked = arith_data.cut_as_integer(data_checked)

    load_and_plugin(
        type="biquad_df1",
        id="1",
        params={
            "BITWIDTH": bitwidth,
            "LENGTH": order,
            "NUM_MULT": num_mult,
            "FILT_COEFFS": fir_params,
        },
        packages=["filter_data"],
        path2save=cocotb_test_fixture.get_artifact_dir() / "verilog",
        add_ringbuffer=True,
        use_dsp_mult=True,
    )

    cocotb_test_fixture.write({"data": data_in, "check": data_checked})
    cocotb_test_fixture.set_top_module_name("BIQUAD_DF1_1")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")
    cocotb_test_fixture.run(
        params={},
        defines={},
    )
