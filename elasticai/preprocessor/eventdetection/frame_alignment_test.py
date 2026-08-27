import unittest
from copy import deepcopy

import numpy as np

from elasticai.preprocessor.waveform import WaveformGenerator

from .frame_alignment import (
    DefaultSettingsFrameAlignment,
    FrameAligner,
    SettingsFrameAlignment,
)


def _build_spike_waveform(sampling_rate: float) -> np.ndarray:
    return (
        WaveformGenerator(sampling_rate=sampling_rate)
        .generate_waveform(
            time_points=[0.0],
            time_duration=[1.6e-3],
            waveform_select=["EAP"],
            polarity_cathodic=[False],
        )
        .signal
    )


def _build_spike_signal(
    scale_pp_range: list[float],
    scale_noise: float,
    pos_spike: list[float],
    sampling_rate: float,
    do_noise: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    t_end_sim = float(pos_spike[-1] + 3.2e-3)
    time = np.linspace(start=0.0, stop=t_end_sim, num=int(t_end_sim * sampling_rate), endpoint=True)
    spike_signal = np.zeros_like(time)
    spike_pos = np.zeros_like(pos_spike, dtype=np.int32)
    spike_template = _build_spike_waveform(sampling_rate)
    for idx, pos in enumerate(pos_spike):
        pos_start = int(pos * sampling_rate)
        pos_end = pos_start + spike_template.size
        scale_pp = (scale_pp_range[1] - scale_pp_range[0]) * np.random.rand(1) + scale_pp_range[0]

        spike_signal[pos_start:pos_end] = scale_pp * spike_template
        spike_pos[idx] = pos_start + int(spike_template.size / 2)

    spike_signal += (
        np.zeros_like(spike_signal)
        if not do_noise
        else 0.5 * scale_noise * np.random.randn(spike_signal.size)
    )
    return spike_signal, spike_pos


def _build_sorted_timestamps(count: int, min_gap: float = 0.002, max_gap: float = 0.01) -> list:
    return WaveformGenerator(sampling_rate=0.0).build_random_timestamps(
        count=count,
        min_gap=min_gap,
        max_gap=max_gap,
    )


class TestBuildEAP(unittest.TestCase):
    def test_build_spike_waveform(self):
        signal = _build_spike_waveform(sampling_rate=20e3)
        assert signal.size == 32
        self.assertEqual(signal.min(), -1.0)

    def test_build_spike_signal_specific(self):
        signal = _build_spike_signal(
            scale_pp_range=[80e-6, 120e-6],
            scale_noise=10e-6,
            pos_spike=[0.01, 0.21, 0.3, 0.34, 0.42, 0.44, 0.46, 0.48, 0.8, 1.0],
            sampling_rate=20e3,
        )
        self.assertEqual(signal[1].size, 10)

    def test_build_spike_signal_random(self):
        signal = _build_spike_signal(
            scale_pp_range=[80e-6, 120e-6],
            scale_noise=10e-6,
            pos_spike=_build_sorted_timestamps(
                count=100,
                min_gap=0.005,
                max_gap=0.1,
            ),
            sampling_rate=20e3,
        )
        self.assertEqual(signal[1].size, 100)

    def test_build_sorted_timestamps(self):
        pos = _build_sorted_timestamps(count=10, min_gap=2e-3, max_gap=20e-3)
        self.assertEqual(len(pos), 10)


class TestSettingsFrameGenerator(unittest.TestCase):
    def setUp(self):
        self.set0: SettingsFrameAlignment = deepcopy(DefaultSettingsFrameAlignment)

    def test_integer_offset_length(self):
        self.set0.sampling_rate = 20e3
        input = [1.6e-3, 2e-3, 2.4e-3]
        chck = [int(val * self.set0.sampling_rate) for val in input]

        for test_val, true_val in zip(input, chck):
            self.set0.offset_sec = test_val
            rslt = self.set0.length_offset_int
            self.assertEqual(rslt, true_val)

    def test_integer_align_position(self):
        self.set0.sampling_rate = 20e3
        input = [1.6e-3, 2e-3, 2.4e-3]
        chck = [int(val * self.set0.sampling_rate) for val in input]

        for test_val, true_val in zip(input, chck):
            self.set0.align_sec = test_val
            rslt = self.set0.length_align_position
            self.assertEqual(rslt, true_val)


class TestFrameGenerator(unittest.TestCase):
    def setUp(self):
        self.set0: SettingsFrameAlignment = deepcopy(DefaultSettingsFrameAlignment)
        self.set0.window_sec = 1.6e-3
        self.frames_eap = np.array(
            [_build_spike_waveform(sampling_rate=self.set0.sampling_rate) for _ in range(10)]
        )

    def test_methods_overview(self):
        rslt = FrameAligner(self.set0)._get_methods()
        self.assertEqual(len(rslt), 7)
        self.assertTrue("none" in rslt)
        self.assertTrue("max" in rslt)

    def test_get_align_frame_none(self):
        self.set0.type = "none"
        rslt = FrameAligner(self.set0).get_aligned_position(frame_in=self.frames_eap[0])
        assert rslt == [self.set0.length_offset_int] * 1

    def test_get_align_frames_none(self):
        self.set0.type = "none"
        rslt = FrameAligner(self.set0).get_aligned_position(frame_in=self.frames_eap)
        assert rslt == [self.set0.length_offset_int] * self.frames_eap.shape[0]

    def test_get_align_frames_max(self):
        self.set0.type = "max"
        rslt = FrameAligner(self.set0).get_aligned_position(frame_in=self.frames_eap)
        assert rslt == [17 - self.set0.length_offset_int] * self.frames_eap.shape[0]

    def test_get_align_frames_min(self):
        self.set0.type = "min"
        rslt = FrameAligner(self.set0).get_aligned_position(frame_in=self.frames_eap)
        assert rslt == [9 - self.set0.length_offset_int] * self.frames_eap.shape[0]

    def test_get_align_frames_ntp(self):
        self.set0.type = "ntp"
        rslt = FrameAligner(self.set0).get_aligned_position(frame_in=self.frames_eap)
        assert rslt == [7 - self.set0.length_offset_int] * self.frames_eap.shape[0]

    def test_get_align_frames_ptp(self):
        self.set0.type = "ptp"
        rslt = FrameAligner(self.set0).get_aligned_position(frame_in=self.frames_eap)
        assert rslt == [12 - self.set0.length_offset_int] * self.frames_eap.shape[0]

    def test_get_align_frames_absmax(self):
        self.set0.type = "absmax"
        rslt = FrameAligner(self.set0).get_aligned_position(frame_in=self.frames_eap)
        assert rslt == [9 - self.set0.length_offset_int] * self.frames_eap.shape[0]

    def test_get_align_frames_absmin(self):
        self.set0.type = "absmin"
        rslt = FrameAligner(self.set0).get_aligned_position(frame_in=self.frames_eap)
        assert rslt == [31 - self.set0.length_offset_int] * self.frames_eap.shape[0]
