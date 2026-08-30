from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main

import numpy as np
import pytest

from elasticai.preprocessor import get_path_to_project
from elasticai.preprocessor.translation.cocotb_tmp import temporary_directory

from .thresholding import (
    DefaultSettingsThreshold,
    SettingsThreshold,
    TargetsThreshold,
    Thresholding,
)

INTEGER_CONFIGS = [
    pytest.param(8, np.int8, "signed char", id="int8"),
    pytest.param(32, np.int32, "signed int", id="int32"),
]

THRESHOLDING_CONFIGS = [
    pytest.param(1000.0, 10e-3, TargetsThreshold.Const, "thresholding_constant", id="method_constant"),
    pytest.param(1000.0, 10e-3, TargetsThreshold.Welford, "thresholding_welford", id="method_welford"),
    pytest.param(1000.0, 10e-3, TargetsThreshold.MovingAverage, "thresholding_mavg", id="method_mavg"),
    pytest.param(512.0, 0.015625, TargetsThreshold.MovingAverage, "thresholding_mavg_pow2", id="method_mavg_pow2"),  # window_steps = int(0.015625 * 512) = 8 = 2^3
    pytest.param(1000.0, 10e-3, TargetsThreshold.RmsMove, "thresholding_mavg_abs", id="method_mavg_abs"),
    pytest.param(512.0, 0.015625, TargetsThreshold.RmsMove, "thresholding_mavg_pow2_abs", id="method_mavg_pow2_abs"),  # window_steps = 8 = 2^3
]

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
        self.assertTrue(TargetsThreshold.Const in rslt)

    def test_getting_position_constant_positive_normal(self):
        self.set0.method = TargetsThreshold.Const
        rslt = Thresholding(settings=self.set0).get_threshold_position(xin=self.signal_in, thr_val=0.5)
        chck = np.array([9, 109, 209, 309, 408, 508, 608, 708, 808, 908])
        np.testing.assert_array_almost_equal(rslt, chck)

    def test_getting_position_constant_negative_normal(self):
        self.set0.method = TargetsThreshold.Const
        rslt = Thresholding(settings=self.set0).get_threshold_position(xin=self.signal_in, thr_val=-0.5)
        chck = np.array([59, 159, 259, 358, 458, 558, 658, 758, 858, 958])
        np.testing.assert_array_almost_equal(rslt, chck)

    def test_getting_position_constant_positive_pretime(self):
        self.set0.method = TargetsThreshold.Const
        rslt = Thresholding(settings=self.set0).get_threshold_position(
            xin=self.signal_in, pre_time=0.05, thr_val=-0.5
        )
        chck = np.array([9, 109, 209, 308, 408, 508, 608, 708, 808, 908])
        np.testing.assert_array_almost_equal(rslt, chck)

    def test_constant(self):
        self.set0.method = TargetsThreshold.Const
        dut = Thresholding(settings=self.set0)
        rslt = dut.get_threshold(self.signal_in, thr_val=0.5)

        assert rslt.size == self.signal_in.size
        self.assertEqual(np.mean(rslt), 0.5)

    def test_abs_mean(self):
        self.set0.method = TargetsThreshold.AbsoluteMean
        dut = Thresholding(settings=self.set0)
        rslt = dut.get_threshold(self.signal_in)

        assert rslt.size == self.signal_in.size
        chck = 1 / np.sqrt(2)
        self.assertLess(np.abs(np.mean(rslt) - chck), 6e-4)

    def test_median_absolute_derivation(self):
        self.set0.method = TargetsThreshold.MedianAbsoluteDeviation
        dut = Thresholding(settings=self.set0)
        rslt = dut.get_threshold(self.signal_in)

        assert rslt.size == self.signal_in.size
        chck = np.zeros_like(rslt) + 1.048301
        np.testing.assert_almost_equal(rslt, chck, decimal=3)

    def test_moving_average(self):
        self.set0.method = TargetsThreshold.MovingAverage
        self.set0.window_sec = 0.2
        dut = Thresholding(settings=self.set0)
        rslt = dut.get_threshold(self.signal_in)

        assert rslt.size == self.signal_in.size
        chck = np.zeros_like(rslt)
        np.testing.assert_almost_equal(rslt[300:], chck[300:], decimal=2)

    def test_moving_absolute_average(self):
        self.set0.method = TargetsThreshold.RmsMove
        self.set0.window_sec = 0.2
        dut = Thresholding(settings=self.set0)
        rslt = dut.get_threshold(self.signal_in)

        assert rslt.size == self.signal_in.size
        chck = np.zeros_like(rslt) + 0.637
        np.testing.assert_almost_equal(rslt[300:], chck[300:], decimal=3)

    def test_root_mean_squared_normal(self):
        self.set0.method = TargetsThreshold.RmsNorm
        dut = Thresholding(settings=self.set0)
        rslt = dut.get_threshold(self.signal_in)

        assert rslt.size == self.signal_in.size
        chck = np.zeros_like(rslt) + 0.70709
        np.testing.assert_almost_equal(rslt, chck, decimal=3)

    def test_root_mean_squared_blackrock(self):
        self.set0.method = TargetsThreshold.RmsBlackrock
        dut = Thresholding(settings=self.set0)
        rslt = dut.get_threshold(self.signal_in)

        assert rslt.size == self.signal_in.size
        chck = np.zeros_like(rslt) + 3.181901
        np.testing.assert_almost_equal(rslt, chck, decimal=2)

    def test_welford(self):
        self.set0.method = TargetsThreshold.Welford
        dut = Thresholding(settings=self.set0)
        rslt = dut.get_threshold(self.signal_in)

        assert rslt.size == self.signal_in.size
        chck = np.zeros_like(rslt) + 0.707
        np.testing.assert_almost_equal(rslt[500:], chck[500:], decimal=1)

    def test_create_design_fpga_const(self):
        self.set0.method = TargetsThreshold.Const
        with TemporaryDirectory() as tmpdir:
            path2temp = Path(tmpdir)

            Thresholding(settings=self.set0).create_design(
                data=self.signal_in, id="0", target="fpga", bitwidth=12, path2save=path2temp, thr_val=2
            )
            files_available = [
                "const_0.v",
            ]
            for file in path2temp.glob("*.v"):
                assert file.exists()
                assert file.name in files_available

    def test_create_design_fpga_mavg_norm(self):
        self.set0.method = TargetsThreshold.MovingAverage
        self.set0.window_sec = 0.1
        self.set0.sampling_rate = 1e3

        with TemporaryDirectory() as tmpdir:
            path2temp = Path(tmpdir)

            Thresholding(settings=self.set0).create_design(
                data=self.signal_in, id="0", target="fpga", bitwidth=12, path2save=path2temp
            )
            files_available = [
                "mov_avg_norm_0.v",
            ]
            for file in path2temp.glob("*.v"):
                assert file.exists()
                assert file.name in files_available

    def test_create_design_fpga_mavg_shift(self):
        self.set0.method = TargetsThreshold.MovingAverage
        self.set0.window_sec = 0.1
        self.set0.sampling_rate = 1.28e3

        with TemporaryDirectory() as tmpdir:
            path2temp = Path(tmpdir)

            Thresholding(settings=self.set0).create_design(
                data=self.signal_in, id="0", target="fpga", bitwidth=12, path2save=path2temp
            )
            files_available = [
                "mov_avg_pow2_0.v",
            ]
            for file in path2temp.glob("*.v"):
                assert file.exists()
                assert file.name in files_available

    def test_create_design_fpga_mavg_abs_norm(self):
        self.set0.method = TargetsThreshold.RmsMove
        self.set0.window_sec = 0.1
        self.set0.sampling_rate = 1e3

        with TemporaryDirectory() as tmpdir:
            path2temp = Path(tmpdir)

            Thresholding(settings=self.set0).create_design(
                data=self.signal_in, id="0", target="fpga", bitwidth=12, path2save=path2temp
            )
            files_available = [
                "mov_avg_abs_norm_0.v",
            ]
            for file in path2temp.glob("*.v"):
                assert file.exists()
                assert file.name in files_available

    def test_create_design_fpga_mavg_abs_shit(self):
        self.set0.method = TargetsThreshold.RmsMove
        self.set0.window_sec = 0.1
        self.set0.sampling_rate = 1.28e3

        with TemporaryDirectory() as tmpdir:
            path2temp = Path(tmpdir)

            Thresholding(settings=self.set0).create_design(
                data=self.signal_in, id="0", target="fpga", bitwidth=12, path2save=path2temp
            )
            files_available = [
                "mov_avg_abs_pow2_0.v",
            ]
            for file in path2temp.glob("*.v"):
                assert file.exists()
                assert file.name in files_available

class TestCreateDesing:
    @pytest.mark.parametrize("target", ["mcu", "pc"])
    @pytest.mark.parametrize("bitwidth,numpy_dtype,c_type", INTEGER_CONFIGS)
    @pytest.mark.parametrize("sampling_rate,window_sec,method,c_name", THRESHOLDING_CONFIGS)
    def test_create_desing_generates_thresholding_c_files(
        self,
        target: str,
        sampling_rate: float,
        window_sec: float,
        method: TargetsThreshold,
        c_name: str,
        bitwidth: int,
        numpy_dtype,
        c_type: str,
    ) -> None:
        thresholder = Thresholding(
            SettingsThreshold(
                method=method,
                sampling_rate=sampling_rate,
                window_sec=window_sec,
                do_quant=True,
            )
        )
        backup = get_path_to_project("build_test") / f"{method}"
        with temporary_directory(backup) as tmpdir:
            thresholder.create_design(
                const_threshold=10,
                id="0",
                target=target,
                bitwidth=bitwidth,
                signed=True,
                path2save=tmpdir,
            )
            assert (tmpdir / f"{c_name}_0.c").exists()
            assert (tmpdir / f"{c_name}_0.h").exists()
            assert (tmpdir / f"{c_name}_template.h").exists()


if __name__ == "__main__":
    main()
