from pathlib import Path
from shutil import which
from uuid import uuid4

import pytest
from elasticai.equichecker import CompileLoader

from elasticai.creator_plugins.player.src.c.replayer import (
    build_replayer,
    build_replayer_with_trigger,
)

pytestmark = pytest.mark.skipif(which("cc") is None, reason="requires a C compiler")

DATA = list(range(12))
TRIGGER = [int(v % 3 == 0) for v in range(12)]  # [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]

INTEGER_CONFIGS = [
    pytest.param(8, True, "signed char", id="int8"),
    pytest.param(16, True, "signed short", id="int16"),
]


def _build_and_load(
    tmp_path: Path,
    bitwidth: int,
    signed: bool,
    c_type: str,
) -> CompileLoader:
    src_dir = tmp_path / "src"
    build_replayer(
        data=DATA,
        bitwidth=bitwidth,
        signed=signed,
        path2save=src_dir,
        replayer_id="0",
        define_path=".",
    )
    adapter = tmp_path / "adapter.h"
    adapter.write_text(
        f"{c_type} replayer_next_0(void);\nint replayer_done_0(void);\nvoid replayer_reset_0(void);\n"
    )
    loader = CompileLoader(
        headers=str(adapter),
        sources=[str(src_dir / "replayer_0.c")],
        build_dir=str(tmp_path / "cffi-build"),
        module_name=f"replayer_{uuid4().hex}",
    )
    loader.load()
    return loader


def _build_and_load_with_trigger(
    tmp_path: Path,
    bitwidth: int,
    signed: bool,
    c_type: str,
) -> CompileLoader:
    src_dir = tmp_path / "src"
    build_replayer_with_trigger(
        data=DATA,
        trigger=TRIGGER,
        bitwidth=bitwidth,
        signed=signed,
        path2save=src_dir,
        replayer_id="0",
        define_path=".",
    )
    adapter = tmp_path / "adapter.h"
    adapter.write_text(
        f"{c_type} replayer_next_0(void);\n"
        "unsigned char replayer_trgg_0(void);\n"
        "int replayer_done_0(void);\n"
        "void replayer_reset_0(void);\n"
    )
    loader = CompileLoader(
        headers=str(adapter),
        sources=[str(src_dir / "replayer_0.c")],
        build_dir=str(tmp_path / "cffi-build"),
        module_name=f"replayer_trgg_{uuid4().hex}",
    )
    loader.load()
    return loader


# data only
@pytest.mark.parametrize("bitwidth,signed,c_type", INTEGER_CONFIGS)
def test_replayer_c_data_values(tmp_path: Path, bitwidth: int, signed: bool, c_type: str) -> None:
    loader = _build_and_load(tmp_path, bitwidth, signed, c_type)
    next_fn = loader.get("replayer_next_0")

    result = [int(next_fn()) for _ in DATA]
    assert result == DATA


@pytest.mark.parametrize("bitwidth,signed,c_type", INTEGER_CONFIGS)
def test_replayer_c_done_flag(tmp_path: Path, bitwidth: int, signed: bool, c_type: str) -> None:
    loader = _build_and_load(tmp_path, bitwidth, signed, c_type)
    next_fn = loader.get("replayer_next_0")
    done_fn = loader.get("replayer_done_0")

    for idx in range(len(DATA)):
        expected_done = 1 if idx == len(DATA) - 1 else 0
        assert int(done_fn()) == expected_done, f"done flag wrong at sample {idx}"
        next_fn()


@pytest.mark.parametrize("bitwidth,signed,c_type", INTEGER_CONFIGS)
def test_replayer_c_wraps_around(tmp_path: Path, bitwidth: int, signed: bool, c_type: str) -> None:
    loader = _build_and_load(tmp_path, bitwidth, signed, c_type)
    next_fn = loader.get("replayer_next_0")

    for _ in DATA:
        next_fn()

    result = [int(next_fn()) for _ in DATA]
    assert result == DATA


@pytest.mark.parametrize("bitwidth,signed,c_type", INTEGER_CONFIGS)
def test_replayer_c_reset(tmp_path: Path, bitwidth: int, signed: bool, c_type: str) -> None:
    loader = _build_and_load(tmp_path, bitwidth, signed, c_type)
    next_fn = loader.get("replayer_next_0")
    done_fn = loader.get("replayer_done_0")
    reset_fn = loader.get("replayer_reset_0")

    for _ in range(6):
        next_fn()

    reset_fn()
    assert int(done_fn()) == 0

    result = [int(next_fn()) for _ in DATA]
    assert result == DATA


# data and trigger
@pytest.mark.parametrize("bitwidth,signed,c_type", INTEGER_CONFIGS)
def test_replayer_c_trgg_data_values(tmp_path: Path, bitwidth: int, signed: bool, c_type: str) -> None:
    loader = _build_and_load_with_trigger(tmp_path, bitwidth, signed, c_type)
    next_fn = loader.get("replayer_next_0")

    result = [int(next_fn()) for _ in DATA]
    assert result == DATA


@pytest.mark.parametrize("bitwidth,signed,c_type", INTEGER_CONFIGS)
def test_replayer_c_trgg_trigger_values(tmp_path: Path, bitwidth: int, signed: bool, c_type: str) -> None:
    loader = _build_and_load_with_trigger(tmp_path, bitwidth, signed, c_type)
    next_fn = loader.get("replayer_next_0")
    trgg_fn = loader.get("replayer_trgg_0")

    result_trgg = []
    for _ in DATA:
        result_trgg.append(int(trgg_fn()))
        next_fn()
    assert result_trgg == TRIGGER


@pytest.mark.parametrize("bitwidth,signed,c_type", INTEGER_CONFIGS)
def test_replayer_c_trgg_done_flag(tmp_path: Path, bitwidth: int, signed: bool, c_type: str) -> None:
    loader = _build_and_load_with_trigger(tmp_path, bitwidth, signed, c_type)
    next_fn = loader.get("replayer_next_0")
    done_fn = loader.get("replayer_done_0")

    for idx in range(len(DATA)):
        expected_done = 1 if idx == len(DATA) - 1 else 0
        assert int(done_fn()) == expected_done, f"done flag wrong at sample {idx}"
        next_fn()


@pytest.mark.parametrize("bitwidth,signed,c_type", INTEGER_CONFIGS)
def test_replayer_c_trgg_reset(tmp_path: Path, bitwidth: int, signed: bool, c_type: str) -> None:
    loader = _build_and_load_with_trigger(tmp_path, bitwidth, signed, c_type)
    next_fn = loader.get("replayer_next_0")
    trgg_fn = loader.get("replayer_trgg_0")
    reset_fn = loader.get("replayer_reset_0")

    for _ in range(6):
        next_fn()

    reset_fn()

    result_data = []
    result_trgg = []
    for _ in DATA:
        result_trgg.append(int(trgg_fn()))
        result_data.append(int(next_fn()))
    assert result_data == DATA
    assert result_trgg == TRIGGER
