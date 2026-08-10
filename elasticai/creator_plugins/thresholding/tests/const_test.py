import cocotb
import numpy as np
import pytest
from cocotb.triggers import Timer
from elasticai.creator.testing import CocotbTestFixture, eai_testbench

from elasticai.creator_plugins.thresholding import load_and_plugin
from elasticai.preprocessor.thresholding import SettingsThreshold, Thresholding


@cocotb.test()
@eai_testbench  # add this
async def const_threshold_test(
    dut,
    bitwidth: int,
    is_signed: int,
    const: int,
):
    await Timer(1, units="step")
    assert len(dut.DATA_OUT) == bitwidth
    if is_signed:
        assert dut.DATA_OUT.value.to_signed() == const
    else:
        assert dut.DATA_OUT.value.to_unsigned() == const


@pytest.mark.simulation
@pytest.mark.parametrize(
    "bitwidth, is_signed, const",
    [
        (8, True, 100),
    ],
)
def test_thresholding_const(
    cocotb_test_fixture: CocotbTestFixture, bitwidth: int, is_signed: bool, const: int
):
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_package("thresholding", "verilog/const.v")
    cocotb_test_fixture.set_top_module_name("CONST_THRESHOLD")
    cocotb_test_fixture.run(
        params={
            "BITWIDTH": bitwidth,
        },
        defines={},
    )


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth, is_signed, const", [(8, True, 101), (6, True, -4), (6, False, 60)])
def test_thresholding_const_build(
    cocotb_test_fixture: CocotbTestFixture, bitwidth: int, is_signed: bool, const: int
):

    artifact_dir = cocotb_test_fixture.get_artifact_dir()
    build_dir = artifact_dir / "verilog"

    load_and_plugin(
        type="const",
        id="0",
        params={
            "BITWIDTH": bitwidth,
            "CONST_THR": const,
        },
        packages=["thresholding"],
        path2save=build_dir,
    )

    cocotb_test_fixture.set_top_module_name("CONST_THRESHOLD")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")
    cocotb_test_fixture.run(
        params={},
        defines={},
    )


@pytest.mark.simulation
@pytest.mark.parametrize(
    "bitwidth, is_signed, const",
    [
        (6, True, 10),
    ],
)
def test_thresholding_const_equal(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
    is_signed: bool,
    const: int,
):
    build_dir = cocotb_test_fixture.get_artifact_dir()

    sets = SettingsThreshold(method="const", sampling_rate=100.0, window_sec=1.0, do_quant=False)
    Thresholding(settings=sets).create_design(
        data=np.array([0, 0, 0, 0, 0]),
        id="0",
        target="fpga",
        bitwidth=bitwidth,
        path2save=build_dir / "verilog",
        thr_val=const,
    )

    cocotb_test_fixture.set_top_module_name("CONST_THRESHOLD")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")
    cocotb_test_fixture.run(
        params={},
        defines={},
    )
