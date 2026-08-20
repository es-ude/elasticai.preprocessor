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
async def downsampler_mean_access(
    dut, sig_in: list[int], check: list[int], bitwidth: int, is_signed: bool, num_dsr: int
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
            if is_signed:
                data_out.append(dut.DATA_OUT.value.to_signed())
            else:
                data_out.append(dut.DATA_OUT.value.to_unsigned())
        await ClockCycles(dut.CLK_SYS, 4)

    passed = all([meas in [true - 1, true, true + 1] for meas, true in zip(check, data_out[1:])])
    if not passed:
        print("Input:", sig_in)
        print("Output:", data_out[1:])
        print("Check:", check)
    assert passed


@pytest.mark.simulation
@pytest.mark.parametrize(
    "bitwidth, is_signed, num_dsr",
    [
        (12, True, 3),
        (8, False, 3),
    ],
)
def test_downsampler_mean_po2(
    cocotb_test_fixture: CocotbTestFixture, bitwidth: int, is_signed: bool, num_dsr: int
):
    sig_in_po2 = [1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8]
    check_po2 = [2, 6]

    backup = cocotb_test_fixture.get_artifact_dir()
    with temporary_directory(backup):
        cocotb_test_fixture.write({"sig_in": sig_in_po2, "check": check_po2})
        cocotb_test_fixture.set_top_module_name("DOWNSAMPLER_MEAN")
        cocotb_test_fixture.clear_srcs()
        cocotb_test_fixture.add_srcs_from_package("datarate", "verilog/downsampler_mean.v")
        cocotb_test_fixture.run(
            params={
                "BITWIDTH": bitwidth,
                "DEC_RATE": num_dsr,
            },
            defines={},
        )


@pytest.mark.simulation
@pytest.mark.parametrize(
    "bitwidth, is_signed, num_dsr",
    [
        (12, True, 3),
        (8, False, 3),
    ],
)
def test_downsampler_mean(
    cocotb_test_fixture: CocotbTestFixture, bitwidth: int, is_signed: bool, num_dsr: int
):
    sig_in = [1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 9]
    check = [2, 5, 8]

    backup = cocotb_test_fixture.get_artifact_dir()
    with temporary_directory(backup):
        cocotb_test_fixture.write({"sig_in": sig_in, "check": check})
        cocotb_test_fixture.set_top_module_name("DOWNSAMPLER_MEAN")
        cocotb_test_fixture.clear_srcs()
        cocotb_test_fixture.add_srcs_from_package("datarate", "verilog/downsampler_mean.v")
        cocotb_test_fixture.run(
            params={
                "BITWIDTH": bitwidth,
                "DEC_RATE": num_dsr,
            },
            defines={},
        )


@pytest.mark.simulation
@pytest.mark.parametrize(
    "bitwidth, is_signed, num_dsr",
    [
        (12, True, 3),
        (8, False, 3),
    ],
)
def test_downsampler_mean_build(
    cocotb_test_fixture: CocotbTestFixture, bitwidth: int, is_signed: bool, num_dsr: int
):
    sig_in = [1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 9]
    check = [2, 5, 8]

    backup = cocotb_test_fixture.get_artifact_dir()
    with temporary_directory(backup) as tmpdir:
        build_dir = tmpdir / "verilog"
        load_and_plugin(
            type="downsampler_mean",
            id="0",
            params={"BITWIDTH": bitwidth, "DEC_RATE": 3},
            packages=["datarate"],
            path2save=build_dir,
        )

        cocotb_test_fixture.write({"sig_in": sig_in, "check": check})
        cocotb_test_fixture.set_top_module_name("DOWNSAMPLER_MEAN_0")
        cocotb_test_fixture.clear_srcs()
        cocotb_test_fixture.add_srcs_from_dir(path=tmpdir, glob_pattern="verilog/*.v")
        cocotb_test_fixture.run(params={}, defines={})


@pytest.mark.simulation
@pytest.mark.parametrize(
    "bitwidth, is_signed, num_dsr",
    [
        (12, True, 4),
        (12, False, 4),
        (8, False, 6),
        (10, True, 6),
    ],
)
def test_downsampler_mean_build_equal(
    cocotb_test_fixture: CocotbTestFixture, bitwidth: int, is_signed: bool, num_dsr: int
):
    dut = DownSampling(
        SettingsDownSampling(
            sampling_rate=1000.0,
            dsr=num_dsr,
        )
    )

    arith = int_arithmetic(total_bits=bitwidth, signed=is_signed)
    sig_in = np.linspace(
        start=arith.minimum_as_integer, stop=arith.maximum_as_integer, num=10 * num_dsr, dtype=int
    ).tolist()
    sig_in.extend([sig_in[-1] for _ in range(num_dsr)])
    data_checked = (dut.do_simple(uin=np.asarray(sig_in)).astype(int)).tolist()

    backup = cocotb_test_fixture.get_artifact_dir()
    with temporary_directory(backup) as tmpdir:
        build_dir = tmpdir / "verilog"

        dut.create_design(
            target="fpga",
            method=TargetsDownSampling.Simple,
            bitwidth=bitwidth,
            signed=is_signed,
            id="1",
            path2save=build_dir,
        )

        cocotb_test_fixture.write({"sig_in": sig_in, "check": data_checked})
        cocotb_test_fixture.set_top_module_name("DOWNSAMPLER_MEAN_1")
        cocotb_test_fixture.clear_srcs()
        cocotb_test_fixture.add_srcs_from_dir(path=tmpdir, glob_pattern="verilog/*.v")
        cocotb_test_fixture.run(params={}, defines={})
