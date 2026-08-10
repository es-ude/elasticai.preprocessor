from copy import deepcopy
from unittest import TestCase, main

import numpy as np

from .thresholding import DefaultSettingsThreshold, SettingsThreshold, Thresholding


class SettingsThresholdingTest(TestCase):
    set0: SettingsThreshold = deepcopy(DefaultSettingsThreshold)

    def test_window_steps(self):
        self.set0.sampling_rate = 1e3
        self.set0.window_sec = 0.1
        self.assertEqual(self.set0.window_steps, 100)


class ThresholdingTest(TestCase):
    def setUp(self):
        self.set0: SettingsThreshold = deepcopy(DefaultSettingsThreshold)
        t_end = 1.0
        time = np.linspace(start=0, stop=t_end, num=int(t_end * self.set0.sampling_rate), endpoint=True)
        self.signal_in = np.sin(2 * np.pi * time * 10.0)

    def test_indices_event_detection(self):
        stimuli = np.array([3, 4, 5, 18, 19, 20, 33, 34, 35, 37])
        chck = np.array([3, 18, 33, 37])
        rslt = Thresholding(settings=self.set0)._get_values_non_incremented_change(stimuli)
        self.assertEqual(set(rslt), set(chck))

    def test_getting_overview(self):
        dut = Thresholding(settings=self.set0)
        rslt = dut._get_overview()
        assert len(rslt) == 8
        self.assertTrue("const" in rslt)

    def test_getting_position_constant_positive_normal(self):
        self.set0.method = "const"
        rslt = Thresholding(settings=self.set0).get_threshold_position(xin=self.signal_in, thr_val=0.5)
        chck = np.array([9, 109, 209, 309, 408, 508, 608, 708, 808, 908])
        np.testing.assert_array_almost_equal(rslt, chck)

    def test_getting_position_constant_negative_normal(self):
        self.set0.method = "const"
        rslt = Thresholding(settings=self.set0).get_threshold_position(xin=self.signal_in, thr_val=-0.5)
        chck = np.array([59, 159, 259, 358, 458, 558, 658, 758, 858, 958])
        np.testing.assert_array_almost_equal(rslt, chck)

    def test_getting_position_constant_positive_pretime(self):
        self.set0.method = "const"
        rslt = Thresholding(settings=self.set0).get_threshold_position(
            xin=self.signal_in, pre_time=0.05, thr_val=-0.5
        )
        chck = np.array([9, 109, 209, 308, 408, 508, 608, 708, 808, 908])
        np.testing.assert_array_almost_equal(rslt, chck)

    def test_constant(self):
        self.set0.method = "const"
        dut = Thresholding(settings=self.set0)
        rslt = dut.get_threshold(self.signal_in, thr_val=0.5)

        assert rslt.size == self.signal_in.size
        self.assertEqual(np.mean(rslt), 0.5)

    def test_abs_mean(self):
        self.set0.method = "abs_mean"
        dut = Thresholding(settings=self.set0)
        rslt = dut.get_threshold(self.signal_in)

        assert rslt.size == self.signal_in.size
        chck = 1 / np.sqrt(2)
        self.assertLess(np.abs(np.mean(rslt) - chck), 6e-4)

    def test_median_absolute_derivation(self):
        self.set0.method = "mad"
        dut = Thresholding(settings=self.set0)
        rslt = dut.get_threshold(self.signal_in)

        assert rslt.size == self.signal_in.size
        chck = np.zeros_like(rslt) + 1.048301
        np.testing.assert_almost_equal(rslt, chck, decimal=3)

    def test_moving_average(self):
        self.set0.method = "mavg"
        self.set0.window_sec = 0.2
        dut = Thresholding(settings=self.set0)
        rslt = dut.get_threshold(self.signal_in)

        assert rslt.size == self.signal_in.size
        chck = np.zeros_like(rslt)
        np.testing.assert_almost_equal(rslt[300:], chck[300:], decimal=2)

    def test_moving_absolute_average(self):
        self.set0.method = "mavg_abs"
        self.set0.window_sec = 0.2
        dut = Thresholding(settings=self.set0)
        rslt = dut.get_threshold(self.signal_in)

        assert rslt.size == self.signal_in.size
        chck = np.zeros_like(rslt) + 0.637
        np.testing.assert_almost_equal(rslt[300:], chck[300:], decimal=3)

    def test_root_mean_squared_normal(self):
        self.set0.method = "rms_norm"
        dut = Thresholding(settings=self.set0)
        rslt = dut.get_threshold(self.signal_in)

        assert rslt.size == self.signal_in.size
        chck = np.zeros_like(rslt) + 0.70709
        np.testing.assert_almost_equal(rslt, chck, decimal=3)

    def test_root_mean_squared_blackrock(self):
        self.set0.method = "rms_black"
        dut = Thresholding(settings=self.set0)
        rslt = dut.get_threshold(self.signal_in)

        assert rslt.size == self.signal_in.size
        chck = np.zeros_like(rslt) + 3.181901
        np.testing.assert_almost_equal(rslt, chck, decimal=2)

    def test_welford(self):
        self.set0.method = "welford"
        dut = Thresholding(settings=self.set0)
        rslt = dut.get_threshold(self.signal_in)

        assert rslt.size == self.signal_in.size
        chck = np.zeros_like(rslt) + 0.707
        np.testing.assert_almost_equal(rslt[5000:], chck[5000:], decimal=1)


if __name__ == "__main__":
    main()
