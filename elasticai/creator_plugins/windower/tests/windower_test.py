from random import randint

import cocotb
import numpy as np
import pytest
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer
from elasticai.creator.arithmetic import FxpParams
from elasticai.creator.testing import CocotbTestFixture, eai_testbench

from elasticai.creator_plugins.windower.utils import load_and_plugin
from elasticai.preprocessor.windower import SettingsWindow, WindowSequencer


def build_testdata(bitwidth: int, is_signed: bool, samples: int, repeats: int = 8) -> list[int]:
    params = FxpParams(total_bits=bitwidth, frac_bits=0, signed=is_signed)

    data = [
        randint(a=params.minimum_as_integer, b=params.maximum_as_integer)
        for _ in range(repeats * samples)
    ]
    data.append(params.minimum_as_integer)
    data.append(params.maximum_as_integer)
    return data


@cocotb.test()
@eai_testbench
async def check_transfer_function(
    dut, bitwidth: int, samples: int, num_shift: int, data_in: list[int], check: list[int]
):
    period_clk = 5

    # Initialize signals
    dut.CLK_SYS.value = 0
    dut.RSTN.value = 0
    dut.EN.value = 0
    dut.DO_SHIFT.value = 0
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

    # Set Trigger
    dut.EN.value = 1
    await Timer(1, unit="ps")
    assert dut.DVALID.value == 0

    data_buf_out: list[list[int]] = []

    async def collect_if_valid() -> None:
        await Timer(1, unit="ps")
        if dut.DVALID.value == 1:
            # DATA_BUF is one packed Verilog bit vector.
            packed_window = dut.DATA_BUF.value.to_unsigned()

            # Unpack DATA_BUF into 'samples' values of 'bitwidth' bits each.
            window = [
                (packed_window >> (sample_idx * bitwidth)) & ((1 << bitwidth) - 1)
                for sample_idx in range(samples)
            ]

            # Convert from DUT order (newest first) to reference order (oldest first).
            data_buf_out.append(list(reversed(window)))

    for value in data_in:
        await RisingEdge(dut.CLK_SYS)
        await collect_if_valid()

        dut.DATA_IN.value = value
        dut.DO_SHIFT.value = 1

        await RisingEdge(dut.CLK_SYS)
        await collect_if_valid()

        dut.DO_SHIFT.value = 0

        for _ in range(2):
            await RisingEdge(dut.CLK_SYS)
            await collect_if_valid()

    for _ in range(num_shift + 2):
        await RisingEdge(dut.CLK_SYS)
        await collect_if_valid()

    print("out first window:", data_buf_out[0])
    print("check first window:", check[0])

    assert data_buf_out == check


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [8])
@pytest.mark.parametrize("samples", [32])
@pytest.mark.parametrize("num_shift", [4])
def test_windower(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
    samples: int,
    num_shift: int,
):
    sampling_rate = 100.0
    dut = WindowSequencer(
        SettingsWindow(
            sampling_rate=sampling_rate,
            window_sec=samples / sampling_rate,
            overlap_sec=(samples - num_shift) / sampling_rate,
        )
    )

    data_in = build_testdata(bitwidth=bitwidth, is_signed=False, samples=samples, repeats=8)

    signal = np.pad(
        np.asarray(data_in),
        (samples - num_shift, 0),
        mode="constant",
    )

    data_checked = dut.slide(signal).tolist()

    cocotb_test_fixture.write({"data_in": data_in, "check": data_checked})
    cocotb_test_fixture.set_top_module_name("WINDOWER")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_package(
        "windower",
        "verilog/ring_buffer.v",
    )
    cocotb_test_fixture.add_srcs_from_package(
        "windower",
        "verilog/windower.v",
    )

    cocotb_test_fixture.run(
        params={
            "BITWIDTH": bitwidth,
            "SAMPLES": samples,
            "NUM_SHIFT": num_shift,
        },
        defines={},
    )


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [8])
@pytest.mark.parametrize("samples", [32])
@pytest.mark.parametrize("num_shift", [4])
def test_windower_build(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
    samples: int,
    num_shift: int,
):
    sampling_rate = 100
    dut = WindowSequencer(
        SettingsWindow(
            sampling_rate=sampling_rate,
            window_sec=samples / sampling_rate,
            overlap_sec=(samples - num_shift) / sampling_rate,
        )
    )
    data_in = build_testdata(bitwidth=bitwidth, is_signed=False, samples=samples, repeats=8)

    signal = np.pad(
        np.asarray(data_in),
        (samples - num_shift, 0),
        mode="constant",
    )
    data_checked = dut.slide(signal).tolist()

    build_dir = cocotb_test_fixture.get_artifact_dir() / "verilog"
    load_and_plugin(
        type="windower",
        id="",
        params={"BITWIDTH": bitwidth, "SAMPLES": samples, "NUM_SHIFT": num_shift},
        packages=["windower"],
        path2save=build_dir,
        add_ringbuffer=True,
    )

    cocotb_test_fixture.write({"data_in": data_in, "check": data_checked})
    cocotb_test_fixture.set_top_module_name("WINDOWER")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")
    cocotb_test_fixture.run(
        params={},
        defines={},
    )
