import cocotb
import numpy as np
import pytest
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge
from elasticai.creator.arithmetic import int_converter
from elasticai.creator.testing import CocotbTestFixture, eai_testbench

from elasticai.creator_plugins.adc.utils import load_and_plugin
from elasticai.preprocessor.translation.cocotb_tmp import temporary_directory


@cocotb.test()
@eai_testbench
async def emulator(
    dut,
    bitwidth_in: int,
    bitwidth_out: int,
    is_signed: bool,
    num_samples: int,
    num_repeat: int,
    data_tb: list[int],
    expected: list[int],
):
    period_clk = 5

    dut.CLK_SYS.value = 0
    dut.RSTN.value = 0
    dut.EN.value = 0
    dut.IN_VALID.value = 0
    dut.DATA_IN.value = 0

    # Step #1: Start clock and make reset
    cocotb.start_soon(Clock(dut.CLK_SYS, period_clk, unit="ns").start())
    await ClockCycles(dut.CLK_SYS, 4)
    for idx in range(4):
        await ClockCycles(dut.CLK_SYS, 4)
        dut.RSTN.value = idx % 2
    await ClockCycles(dut.CLK_SYS, 4)

    # Step #2: Make inference with disabled module
    dut.EN.value = 0
    dut.IN_VALID.value = 1
    await ClockCycles(dut.CLK_SYS, 1)
    assert dut.DATA_IN.value == 0
    dut.IN_VALID.value = 0
    await ClockCycles(dut.CLK_SYS, 8)

    # Step #3: Make inference with enabled module
    dut.EN.value = 1
    print(data_tb)
    for data in data_tb:
        await ClockCycles(dut.CLK_SYS, 8)
        dut.DATA_IN.value = data
        dut.IN_VALID.value = 1
        await ClockCycles(dut.CLK_SYS, 1)
        dut.IN_VALID.value = 0
        await RisingEdge(dut.DATA_RDY)

    await ClockCycles(dut.CLK_SYS, 20)
    # assert False


def build_data(
    bitwidth: int, is_signed: bool, num_samples: int, num_repeats: int
) -> tuple[list[int], list[int]]:
    data_tb = list()
    data_ch = list()
    arith = int_converter(total_bits=bitwidth, signed=is_signed)
    for ite in range(num_repeats):
        num = np.random.randint(
            low=arith._config.minimum_as_integer,
            high=arith._config.maximum_as_integer,
            size=(num_samples,),
        ).tolist()
        data_ch.append(num)

        val = ""
        for val0 in num:
            val += arith.integer_to_binary_string_verilog(val0).split("b")[-1]
        data_tb.append(int(val, 2))
    return data_ch, data_tb


@pytest.mark.simulation
@pytest.mark.parametrize(
    "bitwidth_in, bitwidth_out, is_signed, num_samples, num_repeat", [(8, 8, True, 4, 2)]
)
def test_template(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth_in: int,
    bitwidth_out: int,
    is_signed: bool,
    num_samples: int,
    num_repeat: int,
) -> None:

    backup = cocotb_test_fixture.get_artifact_dir()
    with temporary_directory(backup):
        data_ck, data_tb = build_data(
            bitwidth=bitwidth_in, is_signed=is_signed, num_samples=num_samples, num_repeats=num_repeat
        )
        cocotb_test_fixture.write({"data_tb": data_tb, "expected": []})
        cocotb_test_fixture.set_top_module_name("RESCALER_MULT")
        cocotb_test_fixture.clear_srcs()
        cocotb_test_fixture.add_srcs_from_package("adc", "verilog/rescaler_MULT.v")
        cocotb_test_fixture.run(
            params={
                "BITWIDTH_IN": bitwidth_in,
                "IS_SIGNED_IN": is_signed,
                "BITWIDTH_OUT": bitwidth_out,
                "IS_SIGNED_OUT": is_signed,
                "NUM_SAMPLES": num_samples,
            },
            defines={},
        )


@pytest.mark.simulation
@pytest.mark.parametrize(
    "bitwidth_in, bitwidth_out, is_signed, num_samples, num_repeat", [(8, 8, True, 12, 2)]
)
def test_build(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth_in: int,
    bitwidth_out: int,
    is_signed: bool,
    num_samples: int,
    num_repeat: int,
) -> None:

    backup = cocotb_test_fixture.get_artifact_dir()
    with temporary_directory(backup) as tmpdir:
        build_dir = tmpdir / "verilog"

        data_ck, data_tb = build_data(
            bitwidth=bitwidth_in, is_signed=is_signed, num_samples=num_samples, num_repeats=num_repeat
        )

        load_and_plugin(
            type="rescaler_mult",
            id="0",
            params={
                "BITWIDTH_IN": bitwidth_in,
                "BITWIDTH_OUT": bitwidth_out,
                "NUM_SAMPLES": num_samples,
                "IS_SIGNED_IN": int(is_signed),
                "IS_SIGNED_OUT": int(is_signed),
                "SCALE_BIAS": 0,
                "SCALE_MULT": 0,
                "ZERO_POINT": 0,
            },
            packages=["adc"],
            path2save=build_dir,
        )
        assert (build_dir / "rescaler_mult_0.v").exists()

        cocotb_test_fixture.write({"data_tb": data_tb, "expected": []})
        cocotb_test_fixture.set_top_module_name("RESCALER_MULT_0")
        cocotb_test_fixture.clear_srcs()
        cocotb_test_fixture.add_srcs_from_dir(path=tmpdir, glob_pattern="verilog/*.v")
        cocotb_test_fixture.run(
            params={},
            defines={},
        )


@pytest.mark.skip
@pytest.mark.simulation
@pytest.mark.parametrize(
    "bitwidth_in, bitwidth_out, is_signed, num_samples, num_repeat", [(8, 8, True, 12, 2)]
)
def test_codesign(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth_in: int,
    bitwidth_out: int,
    is_signed: bool,
    num_samples: int,
    num_repeat: int,
) -> None:
    pass
