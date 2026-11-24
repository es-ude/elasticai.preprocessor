from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from elasticai.preprocessor import get_path_to_project
from elasticai.preprocessor.thresholding import (
    ThresholdTargets,
)
from elasticai.preprocessor.transformation.transformation import transformation_window_method
from elasticai.preprocessor.translation.cocotb_tmp import temporary_directory

from .window import WindowerTargets, WindowSequencer, WindowSettings

WIN_METHOD_CONFIGS = [
    pytest.param(
        WindowerTargets.Sliding,
        "windower_sliding_0.c",
        "windower_sliding_0.h",
        "windower_sliding_template.h",
        id="Sliding",
    ),
    pytest.param(
        WindowerTargets.Event,
        "windower_event_pipeline_0.c",
        "windower_event_pipeline_0.h",
        "preprocessing_normal_template.h",
        id="Event",
    ),
    pytest.param(
        WindowerTargets.Sequence,
        "windower_sequence_0.c",
        "windower_sequence_0.h",
        "windower_sequence_template.h",
        id="Sequence",
    ),
]


class TestWindowMethod(TestCase):
    time = np.linspace(start=0, stop=100e-3, num=2000, endpoint=False, dtype=float)
    vsig = np.sin(2 * np.pi * 100.0 * time) + 0.25 * np.sin(2 * np.pi * 1000.0 * time)
    fs = float(1 / np.diff(time).min())

    def test_transformation_window_method_false(self):
        with pytest.raises(ValueError):
            transformation_window_method(window_size=self.vsig.size, method="Hammingd")

    def test_transformation_window_method_ones(self):
        window = transformation_window_method(window_size=self.vsig.size, method="ones")
        np.testing.assert_array_equal(window, np.ones_like(self.vsig))


class TestWindowSettingsSequencer(TestCase):
    sets = WindowSettings(
        method_window=WindowerTargets.Sequence, sampling_rate=10e3, window_sec=10e-3, overlap_sec=0.1e-3
    )

    def test_settings_length(self):
        self.assertEqual(self.sets.window_length, 100)

    def test_settings_overlap(self):
        self.assertEqual(self.sets.overlap_length, 1)


def stimuli_strategy(num_samples: int):
    """Random stimulus in the range [-1, 1] with a fixed number of samples"""
    return arrays(
        dtype=np.float64,
        shape=num_samples,
        elements=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )


def trials_with_stimuli(window_length: int, max_trials: int = 9):
    """Random number of full windows (1..max_trials) together with a matching random stimulus"""
    return st.integers(min_value=1, max_value=max_trials).flatmap(
        lambda num_trials: st.tuples(st.just(num_trials), stimuli_strategy(num_trials * window_length))
    )


class TestWindowSequencer(TestCase):
    sets = WindowSettings(
        method_window=WindowerTargets.Sequence, sampling_rate=10e3, window_sec=10e-3, overlap_sec=0.1e-3
    )

    @given(data=trials_with_stimuli(window_length=sets.window_length))
    def test_window_sequence_match_full(self, data):
        num_trials, stimuli = data
        set0 = deepcopy(self.sets)
        rslt = WindowSequencer(set0).forward(stimuli)

        assert rslt.shape == (int(num_trials), set0.window_length)
        for idx, sequence in enumerate(rslt):
            start_point = idx * set0.window_length
            chck = stimuli[start_point : start_point + set0.window_length]
            np.testing.assert_array_equal(sequence, chck)

    @given(
        stimuli=st.integers(
            min_value=sets.window_length // 2, max_value=10 * sets.window_length - 1
        ).flatmap(stimuli_strategy)
    )
    def test_window_sequence_match_not_full(self, stimuli):
        set0: WindowSettings = deepcopy(self.sets)
        set0.method_window = WindowerTargets.Sequence
        num_trials = stimuli.size // set0.window_length
        rslt = WindowSequencer(set0).forward(stimuli)

        assert rslt.shape == (num_trials, set0.window_length)
        for idx, sequence in enumerate(rslt):
            start_point = idx * set0.window_length
            chck = stimuli[start_point : start_point + set0.window_length]
            np.testing.assert_array_equal(sequence, chck)

    @given(data=trials_with_stimuli(window_length=sets.window_length))
    def test_window_slide_max_overlapping(self, data):
        num_trials, stimuli = data
        set0: WindowSettings = deepcopy(self.sets)
        set0.method_window = WindowerTargets.Sliding
        set0.overlap_sec = set0.window_sec - 1 / set0.sampling_rate
        num_samples = num_trials * set0.window_length

        delta_steps = set0.window_length - set0.overlap_length
        padded_stimuli = np.pad(
            stimuli,
            (set0.window_length - delta_steps, 0),
            mode="constant",
            constant_values=0,
        )

        rslt = WindowSequencer(set0).forward(stimuli)

        assert rslt.shape == (
            num_samples,
            set0.window_length,
        )
        for idx, sequence in enumerate(rslt):
            start_point = idx
            chck = padded_stimuli[start_point : start_point + set0.window_length]
            np.testing.assert_array_equal(sequence, chck)

    @given(data=trials_with_stimuli(window_length=sets.window_length))
    def test_window_slide_quarter_overlapping(self, data):
        num_trials, stimuli = data
        set0: WindowSettings = deepcopy(self.sets)
        set0.method_window = WindowerTargets.Sliding
        set0.overlap_sec = 0.25 * set0.window_sec

        delta_steps = set0.window_length - set0.overlap_length
        padded_stimuli = np.pad(
            stimuli,
            (set0.window_length - delta_steps, 0),
            mode="constant",
            constant_values=0,
        )

        rslt = WindowSequencer(set0).forward(stimuli)

        expected_num_windows = len(range(0, len(padded_stimuli) - set0.window_length + 1, delta_steps))

        assert rslt.shape == (
            expected_num_windows,
            set0.window_length,
        )

        for idx, sequence in enumerate(rslt):
            start_point = int(idx * self.sets.window_length * 0.75)
            chck = padded_stimuli[start_point : start_point + set0.window_length]
            np.testing.assert_array_equal(sequence, chck)

    @given(data=trials_with_stimuli(window_length=sets.window_length))
    def test_window_slide_half_overlapping(self, data):
        num_trials, stimuli = data
        set0: WindowSettings = deepcopy(self.sets)
        set0.method_window = WindowerTargets.Sliding
        set0.overlap_sec = 0.5 * set0.window_sec

        delta_steps = set0.window_length - set0.overlap_length
        padded_stimuli = np.pad(
            stimuli,
            (set0.window_length - delta_steps, 0),
            mode="constant",
            constant_values=0,
        )

        rslt = WindowSequencer(set0).forward(stimuli)

        expected_num_windows = len(range(0, len(padded_stimuli) - set0.window_length + 1, delta_steps))

        assert rslt.shape == (
            expected_num_windows,
            set0.window_length,
        )

        for idx, sequence in enumerate(rslt):
            start_point = int(idx * self.sets.window_length * 0.5)
            chck = padded_stimuli[start_point : start_point + set0.window_length]
            np.testing.assert_array_equal(sequence, chck)

    @given(data=trials_with_stimuli(window_length=sets.window_length))
    def test_window_slide_sequence(self, data):
        num_trials, stimuli = data
        set0: WindowSettings = deepcopy(self.sets)
        set0.method_window = WindowerTargets.Sliding
        set0.overlap_sec = 0.0
        rslt = WindowSequencer(set0).forward(stimuli)

        assert rslt.shape == (num_trials, set0.window_length)
        for idx, sequence in enumerate(rslt):
            start_point = idx * set0.window_length
            chck = stimuli[start_point : start_point + set0.window_length]
            np.testing.assert_array_equal(sequence, chck)

    def test_window_event_detection_without_padding_normal(self):
        set0: WindowSettings = deepcopy(self.sets)
        set0.method_window = WindowerTargets.Event
        set0.window_sec = 0.25
        num_trials = 5
        stimuli = np.sin(2 * np.pi * np.arange(start=0, stop=num_trials, step=1 / set0.sampling_rate))
        sequence = WindowSequencer(set0)._do_event_detection(signal=stimuli, thr=0.25, pre_time=0.01)
        self.assertEqual(sequence.shape, (num_trials, set0.window_length))
        chck0 = [np.sum(frame == frame[0]) for frame in sequence]
        self.assertTrue(all(x == chck0[0] for x in chck0))

    def test_window_event_detection_without_padding_absolute(self):
        set0: WindowSettings = deepcopy(self.sets)
        set0.window_sec = 0.25
        num_trials = 5
        stimuli = np.sin(2 * np.pi * np.arange(start=0, stop=num_trials, step=1 / set0.sampling_rate))
        sequence = WindowSequencer(set0)._do_event_detection(signal=stimuli, thr=0.25, pre_time=0.01)
        self.assertEqual(sequence.shape, (num_trials, set0.window_length))
        chck0 = [np.sum(frame == frame[0]) for frame in sequence]
        self.assertTrue(all(x == chck0[0] for x in chck0))

    def test_window_event_detection_with_prepadding(self):
        set0: WindowSettings = deepcopy(self.sets)
        set0.window_sec = 0.25
        num_trials = 5
        stimuli = np.sin(2 * np.pi * np.arange(start=0, stop=num_trials, step=1 / set0.sampling_rate))
        sequence = WindowSequencer(set0)._do_event_detection(signal=stimuli, thr=0.25, pre_time=0.05)
        chck0 = np.sum(sequence[0, :] == sequence[0, 0])
        chck1 = np.sum(sequence[1, :] == sequence[1, 0])
        self.assertGreater(chck0, chck1)

    def test_window_event_detection_with_postpadding(self):
        set0: WindowSettings = deepcopy(self.sets)
        set0.window_sec = 0.25
        num_trials = 5
        stimuli = np.sin(2 * np.pi * np.arange(start=0, stop=num_trials, step=1 / set0.sampling_rate))
        stimuli[int(set0.window_length / 2)] = 1.5
        stimuli[-int(set0.window_length / 2)] = 1.5
        sequence = WindowSequencer(set0)._do_event_detection(signal=stimuli, thr=1.2, pre_time=0.05)
        chck0 = np.sum(sequence[-2, :] == sequence[-2, -1])
        chck1 = np.sum(sequence[-1, :] == sequence[-1, -1])
        self.assertGreater(chck1, chck0)

    def test_window_event_detection_high_threshold(self):
        set0: WindowSettings = deepcopy(self.sets)
        set0.window_sec = 0.25
        num_trials = 5
        stimuli = np.sin(2 * np.pi * np.arange(start=0, stop=num_trials, step=1 / set0.sampling_rate))
        sequence = WindowSequencer(set0)._do_event_detection(signal=stimuli, thr=100.0, pre_time=0.01)
        assert sequence == np.asarray([0])

    def test_create_verilog_windower(self):
        set0 = deepcopy(self.sets)
        set0.sampling_rate = 100
        set0.window_sec = 0.32
        set0.overlap_sec = 0.28

        backup = get_path_to_project() / "build_test" / "build_windower_verilog"
        with temporary_directory(backup) as tmpdir:
            WindowSequencer(set0).create_design(target="fpga", total_bitwidth=8, id="0", path2save=tmpdir)
            files_available = [
                "windower_0.v",
                "shift_register.v",
            ]

            for filename in files_available:
                file = tmpdir / filename
                assert file.exists()

    def test_create_design_c_mcu(self):
        set0: WindowSettings = deepcopy(self.sets)
        set0.method_window = WindowerTargets.Sliding
        set0.sampling_rate = 100
        set0.window_sec = 0.32
        set0.overlap_sec = 0.28

        with TemporaryDirectory() as directory:
            path2save = Path(directory)
            WindowSequencer(set0).create_design(
                target="mcu", total_bitwidth=8, id="0", path2save=path2save
            )
            assert (path2save / "windower_sliding_0.h").exists()

    def test_create_design_c_pc(self):
        set0: WindowSettings = deepcopy(self.sets)
        set0.method_window = WindowerTargets.Sliding
        set0.sampling_rate = 100
        set0.window_sec = 0.32
        set0.overlap_sec = 0.28

        with TemporaryDirectory() as directory:
            path2save = Path(directory)
            WindowSequencer(set0).create_design(
                target="pc", total_bitwidth=8, id="0", path2save=path2save
            )
            assert (path2save / "windower_sliding_0.h").exists()


class TestCreateDesign:
    @pytest.mark.parametrize("target", ["mcu", "pc"])
    @pytest.mark.parametrize("window_method,exp_c,exp_h,exp_tmpl", WIN_METHOD_CONFIGS)
    def test_create_design_generates_windower_c_files(
        self,
        target: str,
        window_method: WindowerTargets,
        exp_c: str,
        exp_h: str,
        exp_tmpl: str,
    ) -> None:
        windower = WindowSequencer(
            WindowSettings(
                sampling_rate=2e3,
                window_sec=0.1,
                overlap_sec=0.0,
                method_window=window_method,
                method_thr=ThresholdTargets.Constant,
                threshold=10.0,
                pre_time=0.0025,
            )
        )

        backup = get_path_to_project("build_test") / "windower"
        with temporary_directory(backup) as tmpdir:
            windower.create_design(
                target=target,
                total_bitwidth=8,
                id="0",
                path2save=tmpdir,
                is_signed=True,
            )
            assert (tmpdir / exp_c).exists()
            assert (tmpdir / exp_h).exists()
            assert (tmpdir / exp_tmpl).exists()


if __name__ == "__main__":
    main()
