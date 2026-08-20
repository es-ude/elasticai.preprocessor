import cocotb
import numpy as np
import pytest
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles
from elasticai.creator.arithmetic import int_arithmetic
from elasticai.creator.testing import CocotbTestFixture, eai_testbench

from elasticai.creator_plugins.datarate.utils import load_and_plugin
from elasticai.preprocessor.downsampling import DownSampling, SettingsDownSampling, TargetsDownSampling
from elasticai.preprocessor.translation.cocotb_test import temporary_directory


@cocotb.test()
@eai_testbench
async def subsampler_access(
    dut, sig_in: list[int], check: list[int], bitwidth: int, num_dsr: int, index: int, is_signed: bool
):
    period = 10
    dut.CLK_SYS.value = 0
    dut.RSTN.value = 0
    dut.EN.value = 0
    dut.IN_VALID.value = 0
    dut.DATA_IN.value = 0

    cocotb.start_soon(Clock(dut.CLK_SYS, period, unit="ns").start())
    for _ in range(2):
        dut.RSTN.value = 0
        await ClockCycles(dut.CLK_SYS, 4)
        dut.RSTN.value = 1
        await ClockCycles(dut.CLK_SYS, 4)
    dut.EN.value = 1
    await ClockCycles(dut.CLK_SYS, 2)

    data_out = []
    for idx, sample in enumerate(sig_in):
        dut.DATA_IN.value = sample
        dut.IN_VALID.value = 1
        await ClockCycles(dut.CLK_SYS, 2)
        dut.IN_VALID.value = 0

        if dut.DATA_RDY.value:
            assert idx % num_dsr == index
            if is_signed:
                data_out.append(dut.DATA_OUT.value.to_signed())
            else:
                data_out.append(dut.DATA_OUT.value.to_unsigned())
        await ClockCycles(dut.CLK_SYS, 4)

    passed = data_out == check
    if not passed:
        print("Input:", sig_in)
        print("Output:", data_out)
        print("Check:", check)
    assert passed


@pytest.mark.simulation
@pytest.mark.parametrize(
    "bitwidth, is_signed, num_dsr, index",
    [
        (12, True, 4, 0),
        (12, False, 4, 1),
        (8, False, 8, 1),
    ],
)
def test_subsampler(
    cocotb_test_fixture: CocotbTestFixture, bitwidth: int, is_signed: bool, num_dsr: int, index: int
):
    arith = int_arithmetic(total_bits=bitwidth, signed=is_signed)
    sig_in = np.linspace(
        start=arith.minimum_as_integer, stop=arith.maximum_as_integer, num=10 * num_dsr, dtype=int
    ).tolist()
    check = sig_in[index::num_dsr]

    backup = cocotb_test_fixture.get_artifact_dir()
    with temporary_directory(backup):
        cocotb_test_fixture.write({"sig_in": sig_in, "check": check})
        cocotb_test_fixture.set_top_module_name("SUBSAMPLER")
        cocotb_test_fixture.clear_srcs()
        cocotb_test_fixture.add_srcs_from_package("datarate", "verilog/subsampler.v")
        cocotb_test_fixture.run(
            params={
                "BITWIDTH": bitwidth,
                "DEC_RATE": num_dsr,
                "INDEX": index,
            },
            defines={},
        )


@pytest.mark.simulation
@pytest.mark.parametrize(
    "bitwidth, is_signed, num_dsr, index",
    [
        (12, True, 4, 0),
        (12, False, 4, 1),
        (8, False, 8, 1),
    ],
)
def test_subsampler_build(
    cocotb_test_fixture: CocotbTestFixture, bitwidth: int, is_signed: bool, num_dsr: int, index: int
):
    arith = int_arithmetic(total_bits=bitwidth, signed=is_signed)
    sig_in = np.linspace(
        start=arith.minimum_as_integer, stop=arith.maximum_as_integer, num=10 * num_dsr, dtype=int
    ).tolist()
    check = sig_in[index::num_dsr]

    backup = cocotb_test_fixture.get_artifact_dir()
    with temporary_directory(backup) as tmpdir:
        build_dir = tmpdir / "verilog"
        load_and_plugin(
            type="subsampler",
            id="0",
            params={"BITWIDTH": bitwidth, "DEC_RATE": num_dsr, "INDEX": index},
            packages=["datarate"],
            path2save=build_dir,
        )

        cocotb_test_fixture.write({"sig_in": sig_in, "check": check})
        cocotb_test_fixture.set_top_module_name("SUBSAMPLER_0")
        cocotb_test_fixture.clear_srcs()
        cocotb_test_fixture.add_srcs_from_dir(path=build_dir, glob_pattern="*.v")
        cocotb_test_fixture.run(params={}, defines={})


@pytest.mark.simulation
@pytest.mark.parametrize(
    "bitwidth, is_signed, num_dsr, index",
    [
        (12, True, 4, 0),
        (12, False, 4, 0),
        (8, False, 8, 0),
    ],
)
def test_subsampler_build_equal(
    cocotb_test_fixture: CocotbTestFixture, bitwidth: int, is_signed: bool, num_dsr: int, index: int
):
    arith = int_arithmetic(total_bits=bitwidth, signed=is_signed)
    sig_in = np.linspace(
        start=arith.minimum_as_integer, stop=arith.maximum_as_integer, num=10 * num_dsr, dtype=int
    ).tolist()

    dut = DownSampling(
        SettingsDownSampling(
            sampling_rate=1000.0,
            dsr=num_dsr,
        )
    )
    data_checked = dut.do_subsampling(data=np.asarray(sig_in), take_sample=0).tolist()

    backup = cocotb_test_fixture.get_artifact_dir()
    with temporary_directory(backup) as tmpdir:
        build_dir = tmpdir / "verilog"

        dut.create_design(
            target="fpga",
            method=TargetsDownSampling.Subsampling,
            bitwidth=bitwidth,
            signed=is_signed,
            id="1",
            path2save=build_dir,
        )

        cocotb_test_fixture.write({"sig_in": sig_in, "check": data_checked})
        cocotb_test_fixture.set_top_module_name("SUBSAMPLER_1")
        cocotb_test_fixture.clear_srcs()
        cocotb_test_fixture.add_srcs_from_dir(path=build_dir, glob_pattern="*.v")
        cocotb_test_fixture.run(params={}, defines={})
