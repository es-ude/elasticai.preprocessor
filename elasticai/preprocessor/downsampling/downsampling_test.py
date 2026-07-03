from copy import deepcopy
from unittest import TestCase, main

import numpy as np

from .downsampling import (
    DefaultSettingsDownSampling,
    DownSampling,
    SettingsDownSampling,
)


def inp_samp(time: np.ndarray) -> np.ndarray:
    freq = [4, 20]
    z = 0 * time
    for f in freq:
        z += np.sin(2 * np.pi * f * time)
    return z / len(freq)


class TestDownSampling(TestCase):
    def setUp(self):
        self.sets: SettingsDownSampling = deepcopy(DefaultSettingsDownSampling)
        self.sets.sampling_rate = 2e3
        time = np.linspace(0, 1, int(self.sets.sampling_rate) + 1, endpoint=True, dtype=float)
        self.input = 0.75 * inp_samp(time)

    def test_output_sampling_rate(self):
        self.sets.sampling_rate = 10e3
        self.sets.dsr = 4
        results = DownSampling(self.sets).sampling_rate_out
        self.assertEqual(results, 2.5e3)

    def test_do_simple(self):
        results = DownSampling(self.sets).do_simple(self.input)
        self.assertEqual(results.size, self.sets.sampling_rate / self.sets.dsr)

    def test_cic_size(self):
        check = int(1 + (self.input.size - 1) / self.sets.dsr)
        results = DownSampling(self.sets).do_cic(self.input, 5)
        self.assertEqual(results.size, check)

    def test_cic_type(self):  #
        results = DownSampling(self.sets).do_cic(self.input, 5)
        self.assertEqual(type(results), np.ndarray)

    def test_polyphase_one_size(self):
        check = int((self.input.size - 1) / 2)
        results = DownSampling(self.sets)._do_decimation_polyphase_order_one(self.input)
        self.assertEqual(results.size, check)

    def test_polyphase_one_type(self):
        results = DownSampling(self.sets)._do_decimation_polyphase_order_one(self.input)
        self.assertEqual(type(results), np.ndarray)

    def test_polyphase_two_size(self):
        check = int((self.input.size - 1) / 2)
        results = DownSampling(self.sets)._do_decimation_polyphase_order_two(self.input)
        self.assertEqual(results.size, check)

    def test_polyphase_two_type(self):
        results = DownSampling(self.sets)._do_decimation_polyphase_order_two(self.input)
        self.assertEqual(type(results), np.ndarray)

    def test_do_subsampling_without_augmentation_returns_offset_zero_only(self):
        self.sets.dsr = 3
        data = np.array(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            ]
        )

        results = DownSampling(self.sets).do_subsampling(data, augment=False)

        np.testing.assert_array_equal(
            results,
            np.array(
                [
                    [0, 3, 6, 9],
                    [10, 13, 16, 19],
                ]
            ),
        )

    def test_do_subsampling_generates_offset_samples_without_labels(self):
        self.sets.dsr = 3
        data = np.array(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            ]
        )

        results = DownSampling(self.sets).do_subsampling(data, augment=True)

        np.testing.assert_array_equal(
            results,
            np.array(
                [
                    [0, 3, 6, 9],
                    [10, 13, 16, 19],
                    [1, 4, 7, 0],
                    [11, 14, 17, 0],
                    [2, 5, 8, 0],
                    [12, 15, 18, 0],
                ]
            ),
        )

    def test_do_subsampling_preserves_leading_dimensions(self):
        self.sets.dsr = 2
        data = np.array(
            [
                [[0, 1, 2, 3, 4], [10, 11, 12, 13, 14]],
                [[20, 21, 22, 23, 24], [30, 31, 32, 33, 34]],
            ]
        )

        results = DownSampling(self.sets).do_subsampling(data, augment=True)

        np.testing.assert_array_equal(
            results,
            np.array(
                [
                    [[0, 2, 4], [10, 12, 14]],
                    [[20, 22, 24], [30, 32, 34]],
                    [[1, 3, 0], [11, 13, 0]],
                    [[21, 23, 0], [31, 33, 0]],
                ]
            ),
        )

    def test_do_subsampling_factor_one_returns_input_unchanged(self):
        self.sets.dsr = 1
        data = np.array([[1, 2, 3], [4, 5, 6]])

        results = DownSampling(self.sets).do_subsampling(data, augment=False)

        np.testing.assert_array_equal(results, data)

    def test_do_subsampling_rejects_invalid_downsampling_ratio(self):
        self.sets.dsr = 0
        data = np.array([[1, 2, 3]])

        with self.assertRaisesRegex(ValueError, "dsr must be >= 1"):
            DownSampling(self.sets).do_subsampling(data)


if __name__ == "__main__":
    main()
