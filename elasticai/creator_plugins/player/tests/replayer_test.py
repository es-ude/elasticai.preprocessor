import cocotb
import pytest
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge
from elasticai.creator.testing import CocotbTestFixture, eai_testbench
from elasticai.creator_plugins.bram.utils import translate_path_to_int, write_mem_file

from elasticai.creator_plugins.player.utils import load_and_plugin


@cocotb.test()
@eai_testbench
async def emulator(
    dut,
    bitwidth: int,
    is_signed: bool,
    num_samples: int,
    num_repeat: int,
    check: list[int],
    trgg: list[int],
):
    period_clk = 5
    buffer_data = list()
    buffer_trgg = list()
    trgg_avai = "bram_trgg" in dir(dut)

    dut.CLK_ADC.value = 0
    dut.RSTN.value = 0
    dut.EN.value = 0

    # Start clock and make reset
    cocotb.start_soon(Clock(dut.CLK_ADC, period_clk, unit="ns").start())
    await RisingEdge(dut.CLK_ADC)
    for idx in range(4):
        await RisingEdge(dut.CLK_ADC)
        dut.RSTN.value = idx % 2
    await RisingEdge(dut.CLK_ADC)

    dut.RSTN.value = 1
    for _ in range(2):
        await RisingEdge(dut.CLK_ADC)

    assert dut.DATA_OUT.value == 0
    assert dut.DATA_END.value == 0

    dut.EN.value = 1
    assert dut.NUM_VALUES.value.to_unsigned() == num_samples
    for idx, val in enumerate(check):
        await FallingEdge(dut.CLK_ADC)
        buffer_data.append(dut.DATA_OUT.value.to_unsigned())
        if trgg_avai:
            buffer_trgg.append(dut.DATA_TRGG.value)
        assert dut.cnt_pos.value.to_unsigned() == idx % num_samples

    for _ in range(8):
        await RisingEdge(dut.CLK_ADC)
        dut.EN.value = 0

    if not buffer_data == check:
        print("IND:", len(buffer_data), buffer_data)
        print("REF:", len(check), check)
    assert buffer_data == check

    if trgg_avai:
        assert buffer_trgg == trgg


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth, is_signed, num_samples, num_repeat", [(8, True, 12, 2)])
def test_replayer_only_data(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
    is_signed: bool,
    num_samples: int,
    num_repeat: int,
) -> None:
    data = list()
    for _ in range(num_repeat):
        data.extend([val for val in range(num_samples)])

    build_dir = cocotb_test_fixture.get_artifact_dir()
    path2file = build_dir / "replay_data.mem"
    write_mem_file(path=path2file, data=data, bitwidth=bitwidth)

    cocotb_test_fixture.write({"check": data, "trgg": data})

    cocotb_test_fixture.set_top_module_name("REPLAYER")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_package("player", "verilog/replayer.v")
    cocotb_test_fixture.run(
        params={
            "BITWIDTH": bitwidth,
            "NUM_VALUES": num_samples,
            "PATH2DATA": translate_path_to_int(path2file),
        },
        defines={},
    )


@pytest.mark.parametrize("bitwidth, is_signed, num_samples, num_repeat", [(8, True, 12, 2)])
def test_replayer_data_trgg(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
    is_signed: bool,
    num_samples: int,
    num_repeat: int,
) -> None:
    data = list()
    trgg = list()
    for _ in range(num_repeat):
        data.extend([val for val in range(num_samples)])
        trgg.extend([val % 3 == 0 for val in range(num_samples)])

    build_dir = cocotb_test_fixture.get_artifact_dir()
    path2data = build_dir / "replay_data.mem"
    write_mem_file(path=path2data, data=data, bitwidth=bitwidth)
    path2trgg = build_dir / "replay_trgg.mem"
    write_mem_file(path=path2trgg, data=trgg, bitwidth=1)

    cocotb_test_fixture.write({"check": data, "trgg": trgg})

    cocotb_test_fixture.set_top_module_name("REPLAYER")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_package("player", "verilog/replayer.v")
    cocotb_test_fixture.run(
        params={
            "BITWIDTH": bitwidth,
            "NUM_VALUES": num_samples,
            "PATH2DATA": translate_path_to_int(path2data),
            "PATH2TRGG": translate_path_to_int(path2trgg),
        },
        defines={"ADD_TRIGGER": False},
    )


@pytest.mark.parametrize("bitwidth, is_signed, num_samples, num_repeat", [(8, True, 12, 2)])
def test_replayer_data_trgg_build(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
    is_signed: bool,
    num_samples: int,
    num_repeat: int,
) -> None:
    data = list()
    trgg = list()
    for _ in range(num_repeat):
        data.extend([val for val in range(num_samples)])
        trgg.extend([val % 3 == 0 for val in range(num_samples)])

    build_dir = cocotb_test_fixture.get_artifact_dir() / "verilog"
    path2data = build_dir / "replay_data.mem"
    write_mem_file(path=path2data, data=data, bitwidth=bitwidth)
    path2trgg = build_dir / "replay_trgg.mem"
    write_mem_file(path=path2trgg, data=trgg, bitwidth=1)

    load_and_plugin(
        type="replayer",
        id="0",
        params={
            "BITWIDTH": bitwidth,
            "NUM_VALUES": num_samples,
            "PATH2DATA": translate_path_to_int(path2data),
            "PATH2TRGG": translate_path_to_int(path2trgg),
            "ADD_TRIGGER": True,
        },
        packages=["player"],
        path2save=build_dir,
    )

    cocotb_test_fixture.write({"check": data, "trgg": trgg})
    cocotb_test_fixture.set_top_module_name("REPLAYER_0")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")
    cocotb_test_fixture.run(
        params={},
        defines={},
    )
