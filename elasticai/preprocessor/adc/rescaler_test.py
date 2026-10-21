from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from elasticai.creator.arithmetic import int_arithmetic

from elasticai.preprocessor import BuildPlatformTargets, DataTransient

from .rescaler import Rescaler, RescalerSettingsDefault, RescalerTargets

BIT_CONFIGS = [
    (4, 2, False),
    (4, 2, True),
    (8, 2, False),
    (8, 4, True),
    (12, 4, False),
    (16, 8, True),
    (17, 8, False),
]


def expected_dtype(total_bits: int, is_signed: bool) -> np.dtype:
    bits = next(b for b in (8, 16, 32, 64) if total_bits <= b)
    return np.dtype(f"{'int' if is_signed else 'uint'}{bits}")


def make_rescaler(method=RescalerTargets.FxP, total_bits=8, frac_bits=2, is_signed=False) -> Rescaler:
    sets = deepcopy(RescalerSettingsDefault)
    sets.method, sets.total_bits, sets.frac_bits, sets.is_signed = (
        method,
        total_bits,
        frac_bits,
        is_signed,
    )
    rescaler = Rescaler(settings=sets)
    return rescaler


def rand_int(low: int, high: int, size: int, seed: int = 0, dtype=np.int32) -> np.ndarray:
    return np.random.default_rng(seed).integers(low, high, size=size).astype(dtype)


@pytest.mark.parametrize("method", ["fxp", "FXP", "int_sym", "INT_ASYM"])
def test_method_string_is_converted_to_enum(method: str) -> None:
    sets = deepcopy(RescalerSettingsDefault)
    sets.method = method
    assert Rescaler(settings=sets)._settings.method == RescalerTargets(method.lower())


@pytest.mark.parametrize("method", ["posit", "float", ""])
def test_wrong_method_string(method: str) -> None:
    sets = deepcopy(RescalerSettingsDefault)
    sets.method = method
    with pytest.raises(ValueError):
        Rescaler(settings=sets)


@pytest.mark.parametrize("method", list(RescalerTargets))
@pytest.mark.parametrize("data_type", ["float32", "float64"])
def test_forward_wrong_input_dtype(method: RescalerTargets, data_type: str) -> None:
    with pytest.raises(AttributeError):
        make_rescaler(method).forward(rand_int(-100, 100, 101).astype(data_type))


@pytest.mark.parametrize("data_type", ["int8", "uint8", "int16", "uint16", "int32", "int64"])
def test_forward_accepts_int_dtypes(data_type: str) -> None:
    data = np.arange(0, 100, dtype=data_type)
    assert make_rescaler(RescalerTargets.IntSym).forward(data).shape == data.shape


def test_forward_unsupported_method() -> None:
    rescaler = make_rescaler()
    rescaler._settings.method = "unknown"
    with pytest.raises(AttributeError):
        rescaler.forward(np.arange(10, dtype=np.int16))


@pytest.mark.parametrize("method", list(RescalerTargets))
def test_call_matches_forward_and_keeps_sampling_rate(method: RescalerTargets) -> None:
    data = rand_int(0, 256, 501, dtype=np.uint16)
    rescaler = make_rescaler(method)
    out = rescaler(DataTransient(data=data, sampling_rate=1000.0))
    assert out.sampling_rate == 1000.0
    np.testing.assert_array_equal(out.data, rescaler.forward(data))


@pytest.mark.parametrize("method", list(RescalerTargets))
def test_forward_empty_input(method: RescalerTargets) -> None:
    assert make_rescaler(method).forward(np.array([], dtype=np.int16)).size == 0


def sym_data(is_signed: bool) -> np.ndarray:
    data = rand_int(-1000 if is_signed else 0, 1001, 1001, seed=3)
    data[0] = 1000
    return np.sort(data)


@pytest.mark.parametrize("total_bits, frac_bits, is_signed", BIT_CONFIGS)
def test_int_sym_dtype_range_and_monotonic(total_bits: int, frac_bits: int, is_signed: bool) -> None:
    qmax = int_arithmetic(total_bits=total_bits, signed=is_signed).maximum_as_integer
    out = make_rescaler(RescalerTargets.IntSym, total_bits, frac_bits, is_signed).forward(
        sym_data(is_signed)
    )
    out64 = out.astype(np.int64)
    assert out.dtype == expected_dtype(total_bits, is_signed)
    assert out64.min() >= (-qmax if is_signed else 0)
    assert out64.max() == qmax
    assert np.all(np.diff(out64) >= 0)


def test_int_sym_zero_and_antisymmetry() -> None:
    d = rand_int(0, 1000, 500, seed=4)
    out = (
        make_rescaler(RescalerTargets.IntSym, 8, 2, True)
        .forward(np.concatenate([d, -d, [0]]))
        .astype(np.int64)
    )
    np.testing.assert_array_equal(out[:500], -out[500:1000])
    assert out[-1] == 0


def test_int_sym_all_zero_input() -> None:
    out = make_rescaler(RescalerTargets.IntSym, 8, 2, True).forward(np.zeros(10, dtype=np.int16))
    np.testing.assert_array_equal(out, np.zeros(10))


def test_int_sym_unsigned_clips_negative_values() -> None:
    out = make_rescaler(RescalerTargets.IntSym, 8, 2, False).forward(
        np.array([-100, 0, 50, 100], dtype=np.int16)
    )
    assert out[0] == 0
    assert out.dtype == np.uint8


@pytest.mark.parametrize("is_signed, shift, dtype", [(True, 128, np.int8), (False, 0, np.uint8)])
def test_int_asym_uint8_exact(is_signed: bool, shift: int, dtype: type) -> None:
    data = np.arange(256, dtype=np.uint8)
    out = make_rescaler(RescalerTargets.IntAsym, 8, 2, is_signed).forward(data)
    assert out.dtype == dtype
    np.testing.assert_array_equal(out.astype(np.int64), data.astype(np.int64) - shift)


@pytest.mark.parametrize("total_bits, frac_bits, is_signed", BIT_CONFIGS)
def test_int_asym_full_range_and_monotonic(total_bits: int, frac_bits: int, is_signed: bool) -> None:
    arith = int_arithmetic(total_bits=total_bits, signed=is_signed)
    qmin, qmax = arith.minimum_as_integer, arith.maximum_as_integer
    data = np.sort(rand_int(-300, 700, 1001, seed=6))
    data[0], data[-1] = -300, 699
    out = make_rescaler(RescalerTargets.IntAsym, total_bits, frac_bits, is_signed).forward(data)
    out64 = out.astype(np.int64)
    assert out.dtype == expected_dtype(total_bits, is_signed)
    assert qmin <= out64.min() <= qmin + 1
    assert qmax - 1 <= out64.max() <= qmax
    assert np.all(np.diff(out64) >= 0)


def test_int_asym_constant_input() -> None:
    out = make_rescaler(RescalerTargets.IntAsym, 8, 2, True).forward(np.full(10, 42, dtype=np.int16))
    np.testing.assert_array_equal(out, np.full(10, -128))


def test_int_asym_offset_invariant() -> None:
    data = rand_int(0, 500, 501, seed=8)
    rescaler = make_rescaler(RescalerTargets.IntAsym, 8, 2, True)
    diff = rescaler.forward(data).astype(np.int64) - rescaler.forward(data + 1000).astype(np.int64)
    assert np.max(np.abs(diff)) <= 1


def design_kwargs(path: Path) -> dict:
    return dict(path2save=path, total_bitwidth=8, frac_bitwidth=2, is_signed=False)


@pytest.mark.parametrize("target", ["mem", "pcie"])
def test_create_design_wrong_target(target: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        make_rescaler().create_design(id="0", target=target, **design_kwargs(tmp_path))


@pytest.mark.parametrize("target", [BuildPlatformTargets.MCU, BuildPlatformTargets.Workstation])
def test_create_design_c(target: BuildPlatformTargets, tmp_path: Path) -> None:
    make_rescaler().create_design(id="1", target=target, **design_kwargs(tmp_path))

    files_check = ["rescaler_1.c", "rescaler_1.h", "rescaler_template.h"]
    files_check.sort()
    files_avai = [file.name for file in tmp_path.glob("*.*")]
    files_avai.sort()

    assert len(files_check) == len(files_avai)
    assert files_check == files_avai


@pytest.mark.parametrize("target", [BuildPlatformTargets.FPGA])
def test_create_design_verilog(target: BuildPlatformTargets, tmp_path: Path) -> None:
    make_rescaler().create_design(id="2", target=target, **design_kwargs(tmp_path))

    files_check = ["rescaler_2.v"]
    files_check.sort()
    files_avai = [file.name for file in tmp_path.glob("*.*")]
    files_avai.sort()

    assert len(files_check) == len(files_avai)
    assert files_check == files_avai
