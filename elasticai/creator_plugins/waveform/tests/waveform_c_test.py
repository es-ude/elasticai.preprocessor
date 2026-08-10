from pathlib import Path
from shutil import rmtree
from uuid import uuid4

import pytest
from elasticai.equichecker import CompileLoader, get_c_type

from elasticai.creator_plugins.waveform.utils import prepare_waveform
from elasticai.preprocessor import get_path_to_project
from elasticai.preprocessor.waveform import WaveformGenerator


@pytest.fixture(scope="session", autouse=True)
def build_path():
    BUILD_PATH = get_path_to_project("build_test") / "waveform_c"
    if BUILD_PATH.exists():
        rmtree(BUILD_PATH, ignore_errors=True)
    BUILD_PATH.mkdir(parents=True, exist_ok=True)
    yield BUILD_PATH


@pytest.mark.parametrize("bitwidth", [4, 8, 12, 18, 30])
@pytest.mark.parametrize("is_signed", [True, False])
@pytest.mark.parametrize("num_params", [21, 31])
def test_waveform_full_equi(build_path: Path, num_params: int, bitwidth: int, is_signed: bool):
    id_num = "0"
    ctype = get_c_type(bitwidth, is_signed)
    src_path = build_path / f"{num_params}-{bitwidth}-{ctype}-{False}"

    # --- Build design
    data = WaveformGenerator(sampling_rate=1.0)._create_design_c(
        waveform="SINE_FULL",
        num_params=num_params,
        is_signed=is_signed,
        bitwidth=bitwidth,
        id=id_num,
        path2save=src_path,
        do_opt=False,
        path2include="",
    )

    # --- Prepare CFFI
    adapter = src_path / "adapter.h"
    text = (
        f"void rst_waveform_cnt_{id_num}(void);\n"
        + f"uint8_t get_waveform_pos_{id_num}(void);\n"
        + f"uint8_t get_waveform_lgth_{id_num}(_Bool skip_last_point);\n"
        + f"{ctype} get_waveform_value_{id_num}(_Bool skip_last_point);\n"
    )
    adapter.write_text(text)
    loader = CompileLoader(
        headers=adapter.as_posix(),
        sources=[(src_path / f"waveform_lut_{id_num}.c").as_posix()],
        build_dir=(src_path / "build").as_posix(),
        module_name=f"waveform_sine_{uuid4().hex}",
    )
    loader.load()

    # --- Run Test
    offset = 0 if is_signed else 2 ** (bitwidth - 1)
    assert int(loader.get(f"get_waveform_lgth_{id_num}")(False)) == len(data)
    assert int(loader.get(f"get_waveform_lgth_{id_num}")(True)) == len(data) - 1
    for _ in range(3):
        for ite, expected in enumerate(data[:-1]):
            idx = int(loader.get(f"get_waveform_pos_{id_num}")())
            received = int(loader.get(f"get_waveform_value_{id_num}")(True))
            assert (idx, received) == (ite, expected)
    for ite, expected in enumerate(data):
        idx = int(loader.get(f"get_waveform_pos_{id_num}")())
        received = int(loader.get(f"get_waveform_value_{id_num}")(False))
        assert (idx, received) == (ite, expected)
        if idx == len(data) - 1:
            assert received == offset


# @pytest.mark.parametrize("bitwidth", [4, 8, 12, 18, 32])
@pytest.mark.parametrize("bitwidth", [4, 8])
@pytest.mark.parametrize("is_signed", [True, False])
@pytest.mark.parametrize("num_params", [11])
def test_waveform_opt_equi(build_path: Path, num_params: int, bitwidth: int, is_signed: bool):
    id_num = "1"
    ctype = get_c_type(bitwidth, is_signed)
    src_path = build_path / f"{num_params}-{bitwidth}-{ctype}-{True}"

    # --- Build design
    data = WaveformGenerator(sampling_rate=1.0)._create_design_c(
        waveform="SINE_FULL",
        num_params=num_params,
        is_signed=is_signed,
        bitwidth=bitwidth,
        id=id_num,
        path2save=src_path,
        do_opt=True,
        path2include="",
    )
    full_data = prepare_waveform(
        waveform="SINE_FULL",
        num_params=4 * num_params - 3,
        is_signed=is_signed,
        bitwidth=bitwidth,
        do_opt=False,
    )
    full_data.reverse()
    print(data, len(data))
    print(full_data, len(full_data))

    # --- Prepare CFFI
    adapter = src_path / "adapter.h"
    text = (
        f"void rst_waveform_cnt_{id_num}(void);\n"
        + f"uint8_t get_waveform_pos_{id_num}(void);\n"
        + f"uint8_t get_waveform_lgth_{id_num}(_Bool skip_last_point);\n"
        + f"{ctype} get_waveform_value_{id_num}(_Bool skip_last_point);\n"
    )
    adapter.write_text(text)
    loader = CompileLoader(
        headers=adapter.as_posix(),
        sources=[(src_path / f"waveform_lut_{id_num}.c").as_posix()],
        build_dir=(src_path / "build").as_posix(),
        module_name=f"waveform_sine_{uuid4().hex}",
    )
    loader.load()

    # --- Run Test
    offset = 0 if is_signed else 2 ** (bitwidth - 1)
    assert int(loader.get(f"get_waveform_lgth_{id_num}")(False)) == len(data) * 4 - 4
    assert int(loader.get(f"get_waveform_lgth_{id_num}")(True)) == len(data) * 4 - 5
    for _ in range(2):
        for ite, expected in enumerate(full_data[:-1]):
            idx = int(loader.get(f"get_waveform_pos_{id_num}")())
            received = int(loader.get(f"get_waveform_value_{id_num}")(True))
            print(ite, idx, received, expected)
            assert idx in [ite, 0, len(data) * 4 - 4]
            assert received in (expected - 1, expected, expected + 1)
    for ite, expected in enumerate(full_data):
        idx = int(loader.get(f"get_waveform_pos_{id_num}")())
        received = int(loader.get(f"get_waveform_value_{id_num}")(False))
        print(idx)
        assert idx == ite
        assert received in (expected - 1, expected, expected + 1)
        if idx == len(full_data) - 1:
            assert received == offset
