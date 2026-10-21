from copy import deepcopy
from itertools import product
from pathlib import Path

import numpy as np
import pytest
from elasticai.creator.arithmetic import int_arithmetic

from elasticai.preprocessor import BuildPlatformTargets, DataTransient

from .rescaler import Rescaler, RescalerSettings, RescalerSettingsDefault, RescalerTargets

BIT_CONFIGS = [
    (4, 2, False),
    (4, 2, True),
    (8, 2, False),
    (8, 4, True),
    (12, 4, False),
    (16, 8, True),
    (17, 8, False),
]
ALL_METHODS = list(RescalerTargets)
ALL_CASES = [(m, *cfg) for m, cfg in product(ALL_METHODS, BIT_CONFIGS)]
AFFINE_METHODS = [RescalerTargets.FxP, RescalerTargets.IntAsym]


def get_rescaler(
    method: RescalerTargets = RescalerTargets.FxP,
    total_bits: int = 8,
    frac_bits: int = 2,
    is_signed: bool = False,
    scale_safe: float = 1.0,
    update_params: bool = True,
    scale_pow2: bool = False,
) -> Rescaler:
    sets = deepcopy(RescalerSettingsDefault)
    sets.method, sets.total_bits, sets.frac_bits, sets.is_signed, sets.scale_safe, sets.scale_pow2 = (
        method,
        total_bits,
        frac_bits,
        is_signed,
        scale_safe,
        scale_pow2,
    )
    return Rescaler(settings=sets, update_params=update_params)


def limits(method: RescalerTargets, total_bits: int, is_signed: bool) -> tuple[int, int, int, np.dtype]:
    arith = int_arithmetic(total_bits=total_bits, signed=is_signed)
    qmin, qmax = int(arith.minimum_as_integer), int(arith.maximum_as_integer)
    lower = -qmax if (method == RescalerTargets.IntSym and is_signed) else qmin
    bits = next(b for b in (8, 16, 32, 64) if total_bits <= b)
    return qmin, qmax, lower, np.dtype(f"{'int' if is_signed else 'uint'}{bits}")


def to_codes(out: np.ndarray, rescaler: Rescaler) -> np.ndarray:
    return np.round(out.astype(np.float64) / rescaler._settings.lsb).astype(np.int64)


def make_data(
    kind: str = "rand",
    dtype=np.int32,
    n: int = 1000,
    seed: int = 0,
    low: int = 0,
    high: int = 1,
    amplitude: float = 1500.0,
    offset: float = 0.0,
    noise: float = 0.0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "rand":
        return rng.integers(low, high, size=n).astype(dtype)
    t = np.linspace(0, 8 * np.pi, n)
    return np.round(offset + amplitude * np.sin(t) + rng.normal(0.0, noise, n)).astype(dtype)


def test_settings() -> None:
    for name in ["fxp", "FXP", "int_sym", "INT_ASYM"]:
        sets = RescalerSettings(method=name, scale_safe=1.0, total_bits=8, frac_bits=2, is_signed=False)
        assert sets.method == RescalerTargets(name.lower())

        sets = deepcopy(RescalerSettingsDefault)
        sets.method = name
        rescaler = Rescaler(settings=sets)
        assert rescaler._settings.method == RescalerTargets(name.lower())
        assert sets.method == name
        assert rescaler._settings is not sets

    for name in ["posit", "float", ""]:
        sets = deepcopy(RescalerSettingsDefault)
        sets.method = name
        with pytest.raises(ValueError):
            Rescaler(settings=sets)

    assert RescalerSettingsDefault.method == RescalerTargets.FxP
    assert RescalerSettingsDefault.scale_pow2 is False
    for method, lsb in [
        (RescalerTargets.FxP, 0.125),
        (RescalerTargets.IntSym, 1),
        (RescalerTargets.IntAsym, 1),
    ]:
        assert RescalerSettings(method, 1.0, 8, 3, False).lsb == lsb


def test_forward_input_handling() -> None:
    for method in ALL_METHODS:
        for data_type in ["float32", "float64"]:
            with pytest.raises(AttributeError):
                get_rescaler(method).forward(make_data(low=-100, high=100, n=101).astype(data_type))
        for data_type in ["int8", "uint8", "int16", "uint16", "int32", "int64"]:
            data = np.arange(0, 100, dtype=data_type)
            assert get_rescaler(method).forward(data).shape == data.shape
        with pytest.raises(ValueError):
            get_rescaler(method).forward(np.array([], dtype=np.int16))
        fixed = get_rescaler(method, update_params=False)
        fixed.apply_params(1.0, 0.0)
        assert fixed.forward(np.array([], dtype=np.int16)).size == 0


def test_call_matches_forward_and_keeps_sampling_rate() -> None:
    data = make_data(low=0, high=256, n=501, dtype=np.uint16)
    for method in ALL_METHODS:
        rescaler = get_rescaler(method)
        out = rescaler(DataTransient(data=data, sampling_rate=1000.0))
        assert out.sampling_rate == 1000.0
        np.testing.assert_array_equal(out.data, rescaler.forward(data))


def test_output_dtype_and_code_range() -> None:
    data = np.sort(make_data(low=-300, high=700, n=1001, seed=6))
    data[0], data[-1] = -300, 699

    for method, total_bits, frac_bits, is_signed in ALL_CASES:
        case = f"{method}, {total_bits}, {frac_bits}, {is_signed}"
        qmin, qmax, lower, dtype = limits(method, total_bits, is_signed)
        rescaler = get_rescaler(method, total_bits, frac_bits, is_signed)
        out = rescaler.forward(data)
        codes = to_codes(out, rescaler)

        if method == RescalerTargets.FxP:
            assert out.dtype == np.float32, case
            lsb = 2.0**-frac_bits
            np.testing.assert_array_equal(out.astype(np.float64) / lsb, codes, err_msg=case)
            assert out.min() >= qmin * lsb and out.max() <= qmax * lsb, case
        else:
            assert out.dtype == dtype, case

        assert codes.min() >= lower, case
        assert codes.max() <= qmax, case
        assert np.all(np.diff(codes) >= 0), case
        if method == RescalerTargets.IntSym:
            assert codes.max() == qmax, case
        else:
            assert codes.min() <= qmin + 1, case
            assert codes.max() >= qmax - 1, case


def test_fxp_codes_equal_int_asym_codes() -> None:
    data = make_data(low=-300, high=700, n=1001, seed=7)
    for total_bits, frac_bits, is_signed in BIT_CONFIGS:
        fxp = get_rescaler(RescalerTargets.FxP, total_bits, frac_bits, is_signed)
        asym = get_rescaler(RescalerTargets.IntAsym, total_bits, frac_bits, is_signed)
        np.testing.assert_array_equal(
            to_codes(fxp.forward(data), fxp), asym.forward(data).astype(np.int64)
        )


def test_extreme_input_values_span_output_range() -> None:
    qmin, qmax, _, _ = limits(RescalerTargets.FxP, 8, True)
    for data_type in ["int8", "uint8", "int16", "uint16", "int32", "uint32", "int64"]:
        info = np.iinfo(data_type)
        data = np.array([info.min, info.min // 2 + info.max // 2, info.max], dtype=data_type)
        for method in AFFINE_METHODS:
            rescaler = get_rescaler(method, 8, 2, True)
            codes = to_codes(rescaler.forward(data), rescaler)
            assert qmin <= codes[0] <= qmin + 1, f"{method}, {data_type}"
            assert qmax - 1 <= codes[-1] <= qmax, f"{method}, {data_type}"
            assert np.all(np.diff(codes) >= 0), f"{method}, {data_type}"

    for data_type in ["int8", "int16", "int32", "int64"]:
        info = np.iinfo(data_type)
        out = get_rescaler(RescalerTargets.IntSym, 8, 2, True).forward(
            np.array([info.min, 0, info.max], dtype=data_type)
        )
        assert out[0] == -127 and out[1] == 0 and 126 <= out[2] <= 127, data_type


def test_int_sym() -> None:
    for total_bits, frac_bits, is_signed in BIT_CONFIGS:
        _, qmax, _, _ = limits(RescalerTargets.IntSym, total_bits, is_signed)
        rescaler = get_rescaler(RescalerTargets.IntSym, total_bits, frac_bits, is_signed)
        out = rescaler.forward(np.array([0, 1000], dtype=np.int32))
        zero_point = 0 if is_signed else (qmax + 1) // 2
        assert out[0] == zero_point and rescaler.zero_point == zero_point
        assert out[1] == qmax

    d = make_data(low=0, high=1000, n=500, seed=4)
    out = (
        get_rescaler(RescalerTargets.IntSym, 8, 2, True)
        .forward(np.concatenate([d, -d, [0]]))
        .astype(np.int64)
    )
    np.testing.assert_array_equal(out[:500], -out[500:1000])
    assert out[-1] == 0
    out = get_rescaler(RescalerTargets.IntSym, 8, 2, True).forward(
        np.array([-1000, 0, 1000], dtype=np.int16)
    )
    np.testing.assert_array_equal(out, [-127, 0, 127])

    out = get_rescaler(RescalerTargets.IntSym, 8, 2, False).forward(
        np.array([-100, 0, 50, 100], dtype=np.int16)
    )
    assert out.dtype == np.uint8
    assert out[0] == 1 and out[1] == 128 and abs(int(out[2]) - 192) <= 1 and out[3] == 255


def test_int_asym() -> None:
    data = np.arange(256, dtype=np.uint8)
    for is_signed, shift, dtype in [(True, 128, np.int8), (False, 0, np.uint8)]:
        out = get_rescaler(RescalerTargets.IntAsym, 8, 2, is_signed).forward(data)
        assert out.dtype == dtype
        np.testing.assert_array_equal(out.astype(np.int64), data.astype(np.int64) - shift)

    data = make_data(low=0, high=500, n=501, seed=8)
    rescaler = get_rescaler(RescalerTargets.IntAsym, 8, 2, True)
    diff = rescaler.forward(data).astype(np.int64) - rescaler.forward(data + 1000).astype(np.int64)
    assert np.max(np.abs(diff)) <= 1


def test_constant_input() -> None:
    for method, is_signed in product(ALL_METHODS, [True, False]):
        qmin, qmax, lower, _ = limits(method, 8, is_signed)
        for value in [0, 42, -42]:
            rescaler = get_rescaler(method, 8, 2, is_signed)
            out = rescaler.forward(np.full(10, value, dtype=np.int16))
            codes = to_codes(out, rescaler)
            case = f"{method}, {is_signed}, {value}"
            assert np.all(np.isfinite(out)), case
            assert np.all(codes == codes[0]), case
            assert lower <= codes[0] <= qmax, case
            if method in AFFINE_METHODS:
                assert abs(codes[0] - (qmin + qmax) / 2) <= 1, case

    out = get_rescaler(RescalerTargets.IntSym, 8, 2, True).forward(np.zeros(10, dtype=np.int16))
    np.testing.assert_array_equal(out, np.zeros(10))


def test_scale_safe_leaves_headroom() -> None:
    data = np.arange(0, 1001, dtype=np.int32)
    for method in AFFINE_METHODS:
        r_full = get_rescaler(method, 8, 2, False, scale_safe=1.0)
        r_safe = get_rescaler(method, 8, 2, False, scale_safe=2.0)
        full = to_codes(r_full.forward(data), r_full)
        safe = to_codes(r_safe.forward(data), r_safe)
        assert full.min() <= 1 and full.max() >= 254
        assert abs((safe.max() - safe.min()) - (full.max() - full.min()) / 2) <= 2
        assert abs(safe.min() - (255 - safe.max())) <= 2

    out = get_rescaler(RescalerTargets.IntSym, 8, 2, True, scale_safe=2.0).forward(
        np.array([-1000, 0, 1000], dtype=np.int16)
    )
    assert abs(int(out[2]) - 127 / 2) <= 1 and out[1] == 0


def test_calibration_and_fixed_params() -> None:
    data = make_data(low=-500, high=500, n=501, seed=11)
    for method in ALL_METHODS:
        calib = get_rescaler(method, 8, 2, True)
        ref = calib.forward(data)
        scale, zero_point = calib.scaling_factor, calib.zero_point
        calib.forward(data * 10)
        assert calib.scaling_factor > scale, method

        fixed = get_rescaler(method, 8, 2, True, update_params=False)
        fixed.apply_params(scale, zero_point)
        np.testing.assert_array_equal(fixed.forward(data), ref)
        fixed.forward(data * 10)
        assert fixed.scaling_factor == scale and fixed.zero_point == zero_point, method

        rescaler = get_rescaler(method, 8, 2, False, update_params=True)
        rescaler.apply_params(2.0, 10.0)
        codes = to_codes(rescaler.forward(np.array([0, 10, 20], dtype=np.int16)), rescaler)
        np.testing.assert_array_equal(codes, [10, 15, 20])
        assert rescaler.scaling_factor == 2.0 and rescaler.zero_point == 10.0, method


def test_fixed_params_saturate_instead_of_wrap() -> None:
    for method, total_bits, frac_bits, is_signed in ALL_CASES:
        case = f"{method}, {total_bits}, {frac_bits}, {is_signed}"
        _, qmax, lower, _ = limits(method, total_bits, is_signed)
        calib = get_rescaler(method, total_bits, frac_bits, is_signed)
        calib.forward(make_data(low=-100, high=100, n=501))

        fixed = get_rescaler(method, total_bits, frac_bits, is_signed, update_params=False)
        fixed.apply_params(calib.scaling_factor, calib.zero_point)
        codes = to_codes(fixed.forward(np.array([-100000, -100, 0, 100, 100000], dtype=np.int32)), fixed)
        assert codes[0] == lower, case
        assert codes[-1] == qmax, case
        assert np.all(np.diff(codes) >= 0), case

    for method in AFFINE_METHODS:
        calib = get_rescaler(method, 8, 2, False)
        calib.forward(make_data(low=1000, high=2000, n=501, dtype=np.uint16))
        fixed = get_rescaler(method, 8, 2, False, update_params=False)
        fixed.apply_params(calib.scaling_factor, calib.zero_point)
        codes = to_codes(fixed.forward(np.array([0, 500, 1500, 3000, 65535], dtype=np.uint16)), fixed)
        assert codes[0] == 0 and codes[-1] == 255 and np.all(np.diff(codes) >= 0), method


def test_output_is_similar_to_input() -> None:
    x = make_data("sine", np.int16, amplitude=1500.0, noise=50.0, n=2000)
    xf = x.astype(np.float64)
    for method, total_bits, frac_bits, is_signed in ALL_CASES:
        case = f"{method}, {total_bits}, {frac_bits}, {is_signed}"
        rescaler = get_rescaler(method, total_bits, frac_bits, is_signed)
        out = rescaler.forward(x)
        codes = to_codes(out, rescaler)
        cf = codes.astype(np.float64)

        assert np.corrcoef(xf, out.astype(np.float64))[0, 1] >= (0.99 if total_bits <= 4 else 0.999), case

        span = float(cf.max() - cf.min())
        curve_diff = (xf - xf.min()) / (xf.max() - xf.min()) - (cf - cf.min()) / span
        assert np.max(np.abs(curve_diff)) <= 2.0 / span, case

        reconstructed = (codes - rescaler.zero_point) * rescaler.scaling_factor
        assert np.max(np.abs(reconstructed - x)) <= 0.5 * rescaler.scaling_factor * 1.001, case

    x = make_data("sine", np.uint16, amplitude=1800.0, offset=2048.0, noise=20.0, n=2000)
    for method, total_bits, frac_bits, is_signed in [c for c in ALL_CASES if c[0] in AFFINE_METHODS]:
        out = get_rescaler(method, total_bits, frac_bits, is_signed).forward(x)
        corr = np.corrcoef(x.astype(np.float64), out.astype(np.float64))[0, 1]
        assert corr >= (0.99 if total_bits <= 4 else 0.999), f"{method}, {total_bits}, {is_signed}"


def test_scale_pow2_calibration() -> None:
    data = np.sort(make_data(low=-300, high=700, n=1001, seed=6))
    data[0], data[-1] = -300, 699
    for method, total_bits, frac_bits, is_signed in ALL_CASES:
        case = f"{method}, {total_bits}, {frac_bits}, {is_signed}"
        qmin, qmax, lower, _ = limits(method, total_bits, is_signed)
        free = get_rescaler(method, total_bits, frac_bits, is_signed)
        free.forward(data)
        rescaler = get_rescaler(method, total_bits, frac_bits, is_signed, scale_pow2=True)
        codes = to_codes(rescaler.forward(data), rescaler)

        assert np.log2(rescaler.scaling_factor) == rescaler.shift, case
        assert free.scaling_factor <= rescaler.scaling_factor < 2 * free.scaling_factor, case
        assert codes.min() >= lower and codes.max() <= qmax, case
        assert np.all(np.diff(codes) >= 0), case
        if method in AFFINE_METHODS:
            assert (codes.max() - codes.min()) >= (qmax - lower) / 2 - 2, case

        reconstructed = (codes - rescaler.zero_point) * rescaler.scaling_factor
        assert np.max(np.abs(reconstructed - data)) <= 0.5 * rescaler.scaling_factor * 1.001, case

    for method in ALL_METHODS:
        rescaler = get_rescaler(method, 8, 2, True, scale_pow2=True)
        rescaler.forward(np.zeros(10, dtype=np.int16))
        assert rescaler.scaling_factor == 1.0 and rescaler.shift == 0, method


def test_scale_pow2_matches_integer_shift_reference() -> None:
    inputs = [
        make_data(low=-30000, high=30000, n=2000, seed=1, dtype=np.int16),
        make_data(low=0, high=4096, n=2000, seed=2, dtype=np.uint16),
        make_data(low=-5, high=5, n=500, seed=3, dtype=np.int8),
    ]
    for method, total_bits, frac_bits, is_signed in ALL_CASES:
        _, qmax, lower, _ = limits(method, total_bits, is_signed)
        for x in inputs:
            case = f"{method}, {total_bits}, {frac_bits}, {is_signed}, {x.dtype}"
            rescaler = get_rescaler(method, total_bits, frac_bits, is_signed, scale_pow2=True)
            codes = to_codes(rescaler.forward(x), rescaler)

            shift, zero_point, x64 = rescaler.shift, int(rescaler.zero_point), x.astype(np.int64)
            if shift > 0:
                reference = ((x64 + (1 << (shift - 1))) >> shift) + zero_point
            else:
                reference = (x64 << -shift) + zero_point
            np.testing.assert_array_equal(codes, np.clip(reference, lower, qmax), err_msg=case)


def test_scale_pow2_rounds_half_up() -> None:
    for method in ALL_METHODS:
        rescaler = get_rescaler(method, 8, 2, False, update_params=False, scale_pow2=True)
        rescaler.apply_params(4.0, 100.0)
        x = np.array([-6, -2, 2, 6, 10, 1, 3, -1, -3], dtype=np.int16)
        np.testing.assert_array_equal(
            to_codes(rescaler.forward(x), rescaler),
            [99, 100, 101, 102, 103, 100, 101, 100, 99],
            err_msg=str(method),
        )


def test_scale_pow2_apply_params() -> None:
    for method in ALL_METHODS:
        rescaler = get_rescaler(method, 8, 2, False, scale_pow2=True)
        for scaling in [3.0, 0.0, -2.0, 0.75]:
            with pytest.raises(ValueError):
                rescaler.apply_params(scaling, 0.0)
        for scaling, shift in [(0.25, -2), (1.0, 0), (8.0, 3)]:
            rescaler.apply_params(scaling, 5.0)
            assert rescaler.shift == shift and rescaler.zero_point == 5.0

        plain = get_rescaler(method, 8, 2, False)
        plain.apply_params(3.0, 0.0)
        with pytest.raises(ValueError):
            plain.shift


def test_scale_pow2_similarity_and_saturation() -> None:
    x = make_data("sine", np.int16, amplitude=1500.0, noise=50.0, n=2000)
    for method, total_bits, frac_bits, is_signed in ALL_CASES:
        case = f"{method}, {total_bits}, {frac_bits}, {is_signed}"
        _, qmax, lower, _ = limits(method, total_bits, is_signed)
        rescaler = get_rescaler(method, total_bits, frac_bits, is_signed, scale_pow2=True)
        out = rescaler.forward(x)
        assert np.corrcoef(x.astype(np.float64), out.astype(np.float64))[0, 1] >= (
            0.98 if total_bits <= 4 else 0.998
        ), case

        fixed = get_rescaler(
            method, total_bits, frac_bits, is_signed, update_params=False, scale_pow2=True
        )
        fixed.apply_params(rescaler.scaling_factor, rescaler.zero_point)
        codes = to_codes(fixed.forward(np.array([-32768, 0, 32767], dtype=np.int16)), fixed)
        assert codes[0] >= lower and codes[-1] == qmax, case


def test_create_design_wrong_target(tmp_path: Path) -> None:
    for target in ["mem", "pcie"]:
        with pytest.raises(ValueError):
            get_rescaler().create_design(
                id="0",
                target=target,
                path2save=tmp_path,
                total_bitwidth=8,
                frac_bitwidth=2,
                is_signed=False,
            )


@pytest.mark.parametrize(
    "target, scale_pow2, files",
    [
        (BuildPlatformTargets.MCU, False, ["rescaler_1.c", "rescaler_1.h", "rescaler_template.h"]),
        (
            BuildPlatformTargets.Workstation,
            False,
            ["rescaler_1.c", "rescaler_1.h", "rescaler_template.h"],
        ),
        (BuildPlatformTargets.FPGA, False, ["rescaler_mult_1.v"]),
        (BuildPlatformTargets.FPGA, True, ["rescaler_pow2_1.v"]),
    ],
)
def test_create_design(
    target: BuildPlatformTargets, scale_pow2: bool, files: list[str], tmp_path: Path
) -> None:
    get_rescaler(scale_pow2=scale_pow2).create_design(
        id="1", target=target, path2save=tmp_path, total_bitwidth=8, frac_bitwidth=2, is_signed=False
    )
    assert sorted(file.name for file in tmp_path.glob("*.*")) == sorted(files)
