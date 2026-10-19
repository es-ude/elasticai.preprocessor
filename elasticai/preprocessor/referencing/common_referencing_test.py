import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np

from elasticai.preprocessor import DataTransient

from .common_referencing import Referencing, ReferencingSettingsDefault, ReferencingTargets


def make_dut(kernel_size: int = 3) -> Referencing:
    sets = deepcopy(ReferencingSettingsDefault)
    sets.kernel_size = kernel_size
    return Referencing(sets)


def car_2d_expected(signal: np.ndarray) -> np.ndarray:
    rows, cols, _ = signal.shape
    out = np.zeros_like(signal)
    for r in range(rows):
        for c in range(cols):
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if (dr, dc) != (0, 0) and 0 <= rr < rows and 0 <= cc < cols:
                        out[r, c] += signal[rr, cc]
    return out / 8


class CommonReferencingTest(unittest.TestCase):
    num_samples = 1000
    rng = np.random.default_rng(0)
    signal_flat = 2 * (rng.random(num_samples) - 0.5)
    signal_1d = 2 * (rng.random((1, num_samples)) - 0.5)
    signal_2d = 2 * (rng.random((100, num_samples)) - 0.5)
    signal_3d = 2 * (rng.random((3, 2, num_samples)) - 0.5)
    dut = make_dut(3)

    def test_method_string_is_converted_to_enum(self):
        for method in ("car", "CAR"):
            sets = deepcopy(ReferencingSettingsDefault)
            sets.method = method
            self.assertEqual(Referencing(sets)._settings.method, ReferencingTargets.CAR)

    def test_wrong_method_string(self):
        sets = deepcopy(ReferencingSettingsDefault)
        sets.method = "grid"
        with self.assertRaises(ValueError):
            Referencing(sets)

    def test_dummy_build_flat(self):
        rslt = self.dut._build_dummy_active_mapping(self.signal_flat)
        assert rslt.shape == (1,)
        np.testing.assert_array_equal(rslt, np.array([True]))

    def test_dummy_build_1d(self):
        rslt = self.dut._build_dummy_active_mapping(self.signal_1d)
        assert rslt.shape == (1,)
        np.testing.assert_array_equal(rslt, np.array([True] * 1))

    def test_dummy_build_2d(self):
        rslt = self.dut._build_dummy_active_mapping(self.signal_2d)
        assert rslt.shape == (100,)
        np.testing.assert_array_equal(rslt, np.array([True] * 100))

    def test_dummy_build_3d(self):
        rslt = self.dut._build_dummy_active_mapping(self.signal_3d)
        assert rslt.shape == (3, 2)
        np.testing.assert_array_equal(rslt, np.array([[True] * 2] * 3))

    def test_dummy_build_4d(self):
        with self.assertRaises(NotImplementedError):
            self.dut._build_dummy_active_mapping(np.zeros((2, 2, 2, 10)))

    def test_reference_1d_one_channel_direct(self):
        rslt = self.dut._get_reference_map(
            signal=self.signal_flat,
            active=self.dut._build_dummy_active_mapping(self.signal_flat),
        )
        assert rslt.shape == (self.num_samples,)
        np.testing.assert_almost_equal(rslt, self.signal_flat, decimal=6)

    def test_reference_1d_one_channel(self):
        rslt = self.dut._get_reference_map(
            signal=self.signal_1d,
            active=self.dut._build_dummy_active_mapping(self.signal_1d),
        )
        assert rslt.shape == (self.num_samples,)
        np.testing.assert_array_equal(rslt, self.signal_1d.flatten())

    def test_reference_1d_more_channels(self):
        rslt = self.dut._get_reference_map(
            signal=self.signal_2d,
            active=self.dut._build_dummy_active_mapping(self.signal_2d),
        )
        assert rslt.shape == (self.num_samples,)
        np.testing.assert_allclose(rslt, np.mean(self.signal_2d, axis=0), atol=1e-12)

    def test_reference_1d_inactive_channels_count_as_zero(self):
        active = np.ones(100, dtype=bool)
        active[:10] = False
        rslt = self.dut._get_reference_map(signal=self.signal_2d, active=active)
        np.testing.assert_allclose(rslt, self.signal_2d[10:].sum(axis=0) / 100, atol=1e-12)

    def test_reference_2d_kernel_three(self):
        rslt = self.dut._get_reference_map(
            signal=self.signal_3d,
            active=self.dut._build_dummy_active_mapping(self.signal_3d),
        )
        assert rslt.shape == (3, 2, self.num_samples)
        np.testing.assert_allclose(rslt, car_2d_expected(self.signal_3d), atol=1e-12)

    def test_reference_2d_inactive_channel_is_zero(self):
        active = np.ones((3, 2), dtype=bool)
        active[0, 0] = False
        rslt = self.dut._get_reference_map(signal=self.signal_3d, active=active)
        np.testing.assert_array_equal(rslt[0, 0], np.zeros(self.num_samples))
        np.testing.assert_allclose(rslt[active], car_2d_expected(self.signal_3d)[active], atol=1e-12)

    def test_reference_2d_wrong_kernel_size(self):
        active = np.ones((3, 2), dtype=bool)
        for kernel_size in (1, 2, 4):
            with self.subTest(kernel_size=kernel_size):
                with self.assertRaises(ValueError):
                    make_dut(kernel_size)._get_reference_map(signal=self.signal_3d, active=active)

    def test_reference_4d(self):
        with self.assertRaises(NotImplementedError):
            self.dut._get_reference_map(
                signal=np.zeros((2, 2, 2, 10)), active=np.ones((2, 2), dtype=bool)
            )

    def test_forward_1d_one_channel_direct(self):
        rslt = self.dut.forward(self.signal_flat)
        assert rslt.shape == (self.num_samples,)
        np.testing.assert_almost_equal(rslt, np.zeros(self.num_samples), decimal=6)

    def test_forward_1d_one_channel(self):
        rslt = self.dut.forward(self.signal_1d)
        assert rslt.shape == self.signal_1d.shape
        np.testing.assert_array_equal(rslt, np.zeros_like(self.signal_1d))

    def test_forward_1d_more_channels(self):
        rslt = self.dut.forward(self.signal_2d)
        assert rslt.shape == self.signal_2d.shape
        np.testing.assert_allclose(rslt, self.signal_2d - np.mean(self.signal_2d, axis=0), atol=1e-12)
        np.testing.assert_allclose(rslt.mean(axis=0), np.zeros(self.num_samples), atol=1e-12)

    def test_forward_2d(self):
        rslt = self.dut.forward(self.signal_3d)
        assert rslt.shape == self.signal_3d.shape
        np.testing.assert_allclose(rslt, self.signal_3d - car_2d_expected(self.signal_3d), atol=1e-12)

    def test_forward_2d_default_kernel_raises(self):
        with self.assertRaises(ValueError):
            make_dut(1).forward(self.signal_3d)

    def test_call_matches_forward_and_keeps_sampling_rate(self):
        rslt = self.dut(DataTransient(data=self.signal_2d, sampling_rate=1000.0))
        assert rslt.sampling_rate == 1000.0
        np.testing.assert_array_equal(rslt.data, self.dut.forward(self.signal_2d))

    def test_create_design_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.dut.create_design(
                id="0",
                target="mcu",
                path2save=Path("."),
                total_bitwidth=8,
                frac_bitwidth=0,
                is_signed=True,
            )


if __name__ == "__main__":
    unittest.main()
