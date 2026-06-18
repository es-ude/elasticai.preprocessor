from copy import deepcopy

import numpy as np
import pytest

from .adc import SettingsResampler, TransientResampler


@pytest.fixture(scope="module", autouse=True)
def adc_sets():
    sets = SettingsResampler(
        total_bits=8, frac_bits=4, is_signed=False, srate_orig=100.0, srate_new=100.0, vpos=0.0, vneg=0.0
    )
    yield sets


@pytest.mark.parametrize("vss, vdd, expected", [(0.0, 1.0, 0.5), (-3.3, +3.3, 0.0)])
def test_adc_settings_vcm(adc_sets: SettingsResampler, vss: float, vdd: float, expected: float) -> None:
    sets = deepcopy(adc_sets)
    sets.vpos = vdd
    sets.vneg = vss
    assert sets.vcm == expected


@pytest.mark.parametrize(
    "bitwidth, vss, vdd, expected", [(4, 0.0, 1.0, 0.0625), (8, -3.3, +3.3, 0.02578125)]
)
def test_adc_settings_lsb(
    adc_sets: SettingsResampler, bitwidth: int, vss: float, vdd: float, expected: float
) -> None:
    sets = deepcopy(adc_sets)
    sets.total_bits = bitwidth
    sets.vpos = vdd
    sets.vneg = vss
    assert sets.lsb == expected


@pytest.mark.parametrize(
    "bitwidth, fracwidth, is_signed, input, expected",
    [
        (8, 0, True, [-512, -126, 0, 64, 128, 130, 300], [-128, -126, 0, 64, 127, 127, 127]),
        (8, 0, False, [-512, -126, 0, 64, 128, 130, 300], [0, 0, 0, 64, 128, 130, 255]),
    ],
)
def test_adc_clamp_integer(
    adc_sets: SettingsResampler,
    bitwidth: int,
    fracwidth: int,
    is_signed: bool,
    input: list,
    expected: list,
) -> None:
    sets = deepcopy(adc_sets)
    sets.total_bits = bitwidth
    sets.frac_bits = fracwidth
    sets.is_signed = is_signed

    data_in = np.asarray(input)
    data_out = TransientResampler(sets)._clamp_digital(data=data_in, use_integer=True).tolist()
    assert data_out == expected


@pytest.mark.parametrize(
    "vpos, vneg, input, expected",
    [
        (2.0, -2.0, [-2.5, -2.0, -0.25, 1.5, 2.0], [-2.0, -2.0, -0.25, 1.5, 2.0]),
        (1.0, -1.0, [-1.0, -0.0, 1.0, 2.0, 3.75, 4.0], [-1.0, 0.0, 1.0, 1.0, 1.0, 1.0]),
    ],
)
def test_adc_clamp_voltage(
    adc_sets: SettingsResampler,
    vpos: float,
    vneg: float,
    input: list,
    expected: list,
) -> None:
    sets = deepcopy(adc_sets)
    sets.vpos = vpos
    sets.vneg = vneg

    data_in = np.asarray(input)
    data_out = TransientResampler(sets)._clamp_analog(data=data_in).tolist()
    assert data_out == expected


@pytest.mark.parametrize(
    "bitwidth, fracwidth, is_signed, input, expected",
    [
        (4, 2, True, [-2.5, -2.0, -0.25, 1.5, 2.0], [-2.0, -2.0, -0.25, 1.5, 1.75]),
        (4, 2, False, [-1.0, -0.0, 1.0, 2.0, 3.75, 4.0], [0.0, 0.0, 1.0, 2.0, 3.75, 3.75]),
        (4, 4, True, [-0.6, -0.5, -0.25, -0.05, 0.45, 0.6], [-0.5, -0.5, -0.25, -0.05, 0.4375, 0.4375]),
        (4, 4, False, [-0.5, -0.5, -0.25, -0.05, 0.4375, 1.4375], [0.0, 0.0, 0.0, 0.0, 0.4375, 0.9375]),
    ],
)
def test_adc_clamp_fxp(
    adc_sets: SettingsResampler,
    bitwidth: int,
    fracwidth: int,
    is_signed: bool,
    input: list,
    expected: list,
) -> None:
    sets = deepcopy(adc_sets)
    sets.total_bits = bitwidth
    sets.frac_bits = fracwidth
    sets.is_signed = is_signed

    data_in = np.asarray(input)
    data_out = TransientResampler(sets)._clamp_digital(data=data_in, use_integer=False).tolist()
    assert data_out == expected


@pytest.mark.parametrize(
    "srate_orig, srate_new, input, expected",
    [
        (1.0, 0.0, [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]),
        (1.0, 1.0, [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]),
        (1.0, 1.0, [1.0], [1.0]),
        (100.0, 50.0, [1.0, 1.0, 1.0, 1.0], [1.0, 1.0]),
        (100.0, 50.0, [1.0, 0.5, 0.5, 0.5], [0.7857241997480537, 0.483205893210929]),
        (100.0, 200.0, [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        (
            100.0,
            200.0,
            [1.0, 1.25, 1.5, 1.25],
            [
                0.9997046615074066,
                1.0411081278218748,
                1.25,
                1.4588918721781252,
                1.5002953384925934,
                1.3808656355522009,
                1.25,
                1.2165946274221189,
            ],
        ),
    ],
)
def test_adc_resampling(
    adc_sets: SettingsResampler, srate_orig: float, srate_new: float, input: list, expected: list
):
    sets = deepcopy(adc_sets)
    sets.srate_new = srate_new
    sets.srate_orig = srate_orig

    data_in = np.asarray(input)
    data_out = TransientResampler(sets)._do_resample(data=data_in).tolist()
    assert data_out == expected


@pytest.mark.parametrize(
    "bitwidth, is_signed, expected",
    [
        (4, False, np.uint8),
        (4, True, np.int8),
        (8, True, np.int8),
        (12, True, np.int16),
        (16, False, np.uint16),
        (20, False, np.uint32),
        (31, True, np.int32),
        (32, True, np.int32),
        (50, False, np.uint64),
        (64, False, np.uint64),
    ],
)
def test_int_dtype_quantize_output(
    adc_sets: SettingsResampler, bitwidth: int, is_signed: bool, expected: np.dtype
) -> None:
    sets = deepcopy(adc_sets)
    sets.total_bits = bitwidth
    sets.is_signed = is_signed

    data_in = np.asarray([1.0, 1.0, 1.0])
    data_out = TransientResampler(sets)._quantize(data=data_in, is_int_output=True)
    assert data_out.dtype == expected


@pytest.mark.parametrize(
    "bitwidth, fracwidth, is_signed, input, expected",
    [
        (6, 2, True, [-1.2343, 0.4434, 0.0032, -10.0, +10.0], [-5, 2, 0, -32, 31]),
        (6, 2, False, [-1.2343, 0.4434, 0.0032, -10.0, +20.0], [0, 2, 0, 0, 63]),
        (8, 4, True, [-1.2343, 0.4434, 0.0032, -10.0, +10.0], [-20, 7, 0, -128, 127]),
        (8, 4, False, [-1.2343, 0.4434, 0.0032, -10.0, +20.0], [0, 7, 0, 0, 255]),
    ],
)
def test_adc_quantize_float(
    adc_sets: SettingsResampler,
    bitwidth: int,
    fracwidth: int,
    is_signed: bool,
    input: list,
    expected: list,
):
    sets = deepcopy(adc_sets)
    sets.total_bits = bitwidth
    sets.frac_bits = fracwidth
    sets.is_signed = is_signed

    data_in = np.asarray(input)
    data_out = TransientResampler(sets)._quantize_digital(
        data=data_in, is_int_input=False, is_int_output=True
    )
    assert data_out.dtype == np.int8 if is_signed else np.uint8
    assert data_out.tolist() == expected


@pytest.mark.parametrize(
    "bitwidth, fracwidth, is_signed, input, expected",
    [
        (6, 2, True, [-5, 2, 0, -40, 32], [-5, 2, 0, -32, 31]),
        (6, 2, False, [0, 2, 0, 0, 63], [0, 2, 0, 0, 63]),
        (8, 4, True, [-20, 7, 0, -140, 227], [-20, 7, 0, -128, 127]),
        (8, 4, False, [0, 7, 0, -1, 355], [0, 7, 0, 0, 255]),
    ],
)
def test_adc_quantize_integer(
    adc_sets: SettingsResampler,
    bitwidth: int,
    fracwidth: int,
    is_signed: bool,
    input: list,
    expected: list,
):
    sets = deepcopy(adc_sets)
    sets.total_bits = bitwidth
    sets.frac_bits = fracwidth
    sets.is_signed = is_signed

    data_in = np.asarray(input)
    data_out = TransientResampler(sets)._quantize_digital(
        data=data_in, is_int_input=True, is_int_output=True
    )
    assert data_out.dtype == np.int8 if is_signed else np.uint8
    assert data_out.tolist() == expected


@pytest.mark.parametrize(
    "bitwidth, fracwidth, signed, vpos, vneg, input, expected",
    [
        (
            6,
            4,
            True,
            0.5,
            -0.5,
            [-0.55, -0.434, -0.12, 0.0, 0.22, 0.45, 0.55],
            [-2.0, -1.75, -0.5, 0.0, 0.875, 1.8125, 2.0],
        ),
        (
            8,
            7,
            False,
            0.5,
            -0.5,
            [-0.55, -0.434, -0.12, 0.0, 0.22, 0.45, 0.55],
            [0.0, 0.1328125, 0.7578125, 1.0, 1.4375, 1.8984375, 2.0],
        ),
        (
            8,
            7,
            True,
            0.5,
            0,
            [-0.55, -0.434, -0.12, 0.0, 0.22, 0.45, 0.55],
            [-1.0, -1.0, -1.0, -1.0, -0.1171875, 0.796875, 1.0],
        ),
    ],
)
def test_adc_rescaling_voltage_fxp(
    adc_sets: SettingsResampler,
    bitwidth: int,
    fracwidth: int,
    signed: bool,
    vpos: float,
    vneg: float,
    input: list,
    expected: list,
):
    sets = deepcopy(adc_sets)
    sets.total_bits = bitwidth
    sets.frac_bits = fracwidth
    sets.is_signed = signed
    sets.vpos = vpos
    sets.vneg = vneg

    data_in = np.asarray(input)
    data_out = TransientResampler(sets)._quantize_voltage(data=data_in, is_int_output=False)
    assert data_out.tolist() == expected


@pytest.mark.parametrize(
    "srate_orig, srate_new, t_range_sec, input, expected",
    [
        (10.0, 10.0, [], [1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0]),
        (10.0, 20.0, [], [1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0]),
        (5.0, 10.0, [0.0, 0.3], [1.0, 1.0, 1.0, 1.0, 1.0], [1.0]),
        (5.0, 10.0, [0.2, 0.8], [1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1]),
    ],
)
def test_adc_cutting_input_data_orig(
    adc_sets: SettingsResampler,
    srate_orig: float,
    srate_new: float,
    t_range_sec: list,
    input: list,
    expected: list,
):
    sets = deepcopy(adc_sets)
    sets.srate_new = srate_new
    sets.srate_orig = srate_orig

    data_in = np.asarray(input)
    data_out = TransientResampler(sets).do_cut_transient(
        data=data_in, t_range_sec=t_range_sec, use_srate_orig=True
    )
    assert data_out.tolist() == expected


@pytest.mark.parametrize(
    "srate_orig, srate_new, t_range_sec, input, expected",
    [
        (10.0, 10.0, [], [1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0]),
        (10.0, 20.0, [], [1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0]),
        (5.0, 10.0, [0.0, 0.3], [1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0]),
        (5.0, 10.0, [0.2, 0.8], [1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0]),
    ],
)
def test_adc_cutting_input_data_new(
    adc_sets: SettingsResampler,
    srate_orig: float,
    srate_new: float,
    t_range_sec: list,
    input: list,
    expected: list,
):
    sets = deepcopy(adc_sets)
    sets.srate_new = srate_new
    sets.srate_orig = srate_orig

    data_in = np.asarray(input)
    data_out = TransientResampler(sets).do_cut_transient(
        data=data_in, t_range_sec=t_range_sec, use_srate_orig=False
    )
    assert data_out.tolist() == expected


@pytest.mark.parametrize(
    "srate_orig, srate_new, t_range_sec, label_id, label_pos, expected",
    [
        (10.0, 10.0, [], [0, 1, 0, 1, 0], [1, 2, 3, 4, 5], [[0, 1, 0, 1, 0], [1, 2, 3, 4, 5]]),
        (10.0, 20.0, [], [0, 1, 0, 1, 0], [1, 2, 3, 4, 5], [[0, 1, 0, 1, 0], [1, 2, 3, 4, 5]]),
        (5.0, 10.0, [0.0, 0.3], [0, 1, 0, 1, 0], [1, 2, 3, 4, 5], [[0], [1]]),
        (5.0, 10.0, [0.3, 0.6], [0, 1, 0, 1, 0], [1, 2, 3, 4, 5], [[1], [2]]),
    ],
)
def test_adc_cutting_input_labels_orig(
    adc_sets: SettingsResampler,
    srate_orig: float,
    srate_new: float,
    t_range_sec: list,
    label_id: list,
    label_pos: list,
    expected: list,
):
    sets = deepcopy(adc_sets)
    sets.srate_new = srate_new
    sets.srate_orig = srate_orig

    label0 = np.asarray(label_id)
    label1 = np.asarray(label_pos)
    data_out = TransientResampler(sets).do_cut_labels(
        label_id=label0, label_pos=label1, t_range_sec=t_range_sec, use_srate_orig=True
    )
    assert data_out[0].tolist() == expected[0]
    assert data_out[1].tolist() == expected[1]


@pytest.mark.parametrize(
    "srate_orig, srate_new, t_range_sec, label_id, label_pos, expected",
    [
        (10.0, 10.0, [], [0, 1, 0, 1, 0], [1, 2, 3, 4, 5], [[0, 1, 0, 1, 0], [1, 2, 3, 4, 5]]),
        (10.0, 20.0, [], [0, 1, 0, 1, 0], [1, 2, 3, 4, 5], [[0, 1, 0, 1, 0], [1, 2, 3, 4, 5]]),
        (5.0, 10.0, [0.0, 0.3], [0, 1, 0, 1, 0], [1, 2, 3, 4, 5], [[0, 1], [1, 2]]),
        (5.0, 10.0, [0.2, 0.4], [0, 1, 0, 1, 0], [1, 2, 3, 4, 5], [[1, 0], [2, 3]]),
    ],
)
def test_adc_cutting_input_labels_new(
    adc_sets: SettingsResampler,
    srate_orig: float,
    srate_new: float,
    t_range_sec: list,
    label_id: list,
    label_pos: list,
    expected: list,
):
    sets = deepcopy(adc_sets)
    sets.srate_new = srate_new
    sets.srate_orig = srate_orig

    label0 = np.asarray(label_id)
    label1 = np.asarray(label_pos)
    data_out = TransientResampler(sets).do_cut_labels(
        label_id=label0, label_pos=label1, t_range_sec=t_range_sec, use_srate_orig=False
    )
    assert data_out[0].tolist() == expected[0]
    assert data_out[1].tolist() == expected[1]


@pytest.mark.parametrize(
    "bitwidth, fracwidth, is_signed, input, expected",
    [
        (6, 2, True, [-5, 2, 0, -40, 32], [-5, 2, 0, -32, 31]),
        (6, 2, False, [0, 2, 0, 0, 63], [0, 2, 0, 0, 63]),
        (8, 4, True, [-20, 7, 0, -140, 227], [-20, 7, 0, -128, 127]),
        (8, 4, False, [0, 7, 0, -1, 355], [0, 7, 0, 0, 255]),
    ],
)
def test_adc_quantize_from_int_to_int(
    adc_sets: SettingsResampler,
    bitwidth: int,
    fracwidth: int,
    is_signed: bool,
    input: list,
    expected: list,
):
    sets = deepcopy(adc_sets)
    sets.total_bits = bitwidth
    sets.frac_bits = fracwidth
    sets.is_signed = is_signed

    data_in = np.asarray(input)
    data_out = TransientResampler(sets).redefine_from_int(data=data_in, is_int_output=True)
    assert data_out.dtype == np.int8 if is_signed else np.uint8
    assert data_out.tolist() == expected


@pytest.mark.parametrize(
    "bitwidth, fracwidth, is_signed, input, expected",
    [
        (6, 2, True, [-5, 2, 0, -40, 32], [-1.25, 0.5, 0.0, -8.0, 7.75]),
        (6, 2, False, [0, 2, 0, 0, 63], [0.0, 0.5, 0.0, 0.0, 15.75]),
    ],
)
def test_adc_quantize_from_int_to_fxp(
    adc_sets: SettingsResampler,
    bitwidth: int,
    fracwidth: int,
    is_signed: bool,
    input: list,
    expected: list,
):
    sets = deepcopy(adc_sets)
    sets.total_bits = bitwidth
    sets.frac_bits = fracwidth
    sets.is_signed = is_signed

    data_in = np.asarray(input)
    data_out = TransientResampler(sets).redefine_from_int(data=data_in, is_int_output=False)
    assert data_out.dtype == np.float32
    assert data_out.tolist() == expected


@pytest.mark.parametrize(
    "bitwidth, fracwidth, is_signed, input, expected",
    [
        (6, 2, True, [-1.2343, 0.4434, 0.0032, -10.0, +10.0], [-5, 2, 0, -32, 31]),
        (6, 2, False, [-1.2343, 0.4434, 0.0032, -10.0, +20.0], [0, 2, 0, 0, 63]),
        (8, 4, True, [-1.2343, 0.4434, 0.0032, -10.0, +10.0], [-20, 7, 0, -128, 127]),
        (8, 4, False, [-1.2343, 0.4434, 0.0032, -10.0, +20.0], [0, 7, 0, 0, 255]),
    ],
)
def test_adc_quantize_from_fxp_to_int(
    adc_sets: SettingsResampler,
    bitwidth: int,
    fracwidth: int,
    is_signed: bool,
    input: list,
    expected: list,
):
    sets = deepcopy(adc_sets)
    sets.total_bits = bitwidth
    sets.frac_bits = fracwidth
    sets.is_signed = is_signed

    data_in = np.asarray(input)
    data_out = TransientResampler(sets).redefine_from_fxp(data=data_in, is_int_output=True)
    assert data_out.dtype == np.int8 if is_signed else np.uint8
    assert data_out.tolist() == expected


@pytest.mark.parametrize(
    "bitwidth, fracwidth, is_signed, input, expected",
    [
        (6, 2, True, [-1.2343, 0.4434, 0.0032, -10.0, +10.0], [-1.25, 0.5, 0.0, -8.0, 7.75]),
        (6, 2, False, [-1.2343, 0.4434, 0.0032, -10.0, +20.0], [0.0, 0.5, 0.0, 0.0, 15.75]),
    ],
)
def test_adc_quantize_from_fxp_to_fxp(
    adc_sets: SettingsResampler,
    bitwidth: int,
    fracwidth: int,
    is_signed: bool,
    input: list,
    expected: list,
):
    sets = deepcopy(adc_sets)
    sets.total_bits = bitwidth
    sets.frac_bits = fracwidth
    sets.is_signed = is_signed

    data_in = np.asarray(input)
    data_out = TransientResampler(sets).redefine_from_fxp(data=data_in, is_int_output=False)
    assert data_out.dtype == np.float32
    assert data_out.tolist() == expected


@pytest.mark.parametrize(
    "vpos, vneg, bitwidth, fracwidth, is_signed, input, expected",
    [
        (
            1.0,
            -1.0,
            6,
            4,
            True,
            [-1.1, -1.0, -0.8, -0.24, 0.0, 0.5, 0.95, 1.1],
            [-2.0, -2.0, -1.625, -0.5, 0.0, 1.0, 1.875, 2.0],
        ),
        (
            1.0,
            -1.0,
            6,
            4,
            False,
            [-1.1, -1.0, -0.8, -0.24, 0.0, 0.5, 0.95, 1.1],
            [0.0, 0.0, 0.375, 1.5, 2.0, 3.0, 3.875, 4.0],
        ),
    ],
)
def test_adc_quantize_from_voltage_to_fxp(
    adc_sets: SettingsResampler,
    vpos: float,
    vneg: float,
    bitwidth: int,
    fracwidth: int,
    is_signed: bool,
    input: list,
    expected: list,
):
    sets = deepcopy(adc_sets)
    sets.total_bits = bitwidth
    sets.frac_bits = fracwidth
    sets.is_signed = is_signed
    sets.vpos = vpos
    sets.vneg = vneg

    data_in = np.asarray(input)
    data_out = TransientResampler(sets).redefine_from_voltage(data=data_in, is_int_output=False)
    assert data_out.dtype == np.float32
    assert data_out.tolist() == expected


@pytest.mark.parametrize(
    "vpos, vneg, bitwidth, fracwidth, is_signed, input, expected",
    [
        (
            1.0,
            -1.0,
            6,
            4,
            True,
            [-1.1, -1.0, -0.8, -0.24, 0.0, 0.5, 0.95, 1.1],
            [-32, -32, -26, -8, 0, 16, 30, 32],
        ),
        (
            1.0,
            -1.0,
            6,
            4,
            False,
            [-1.1, -1.0, -0.8, -0.24, 0.0, 0.5, 0.95, 1.1],
            [0, 0, 6, 24, 32, 48, 62, 64],
        ),
    ],
)
def test_adc_quantize_from_voltage_to_int(
    adc_sets: SettingsResampler,
    vpos: float,
    vneg: float,
    bitwidth: int,
    fracwidth: int,
    is_signed: bool,
    input: list,
    expected: list,
):
    sets = deepcopy(adc_sets)
    sets.total_bits = bitwidth
    sets.frac_bits = fracwidth
    sets.is_signed = is_signed
    sets.vpos = vpos
    sets.vneg = vneg

    data_in = np.asarray(input)
    data_out = TransientResampler(sets).redefine_from_voltage(data=data_in, is_int_output=True)
    assert data_out.dtype == np.int8 if is_signed else np.uint8
    assert data_out.tolist() == expected
