from dataclasses import dataclass
from logging import Logger, getLogger

import numpy as np

from elasticai.preprocessor.eventdetection import (
    EventDetection,
    EventPreprocessor,
    FrameAligner,
    SettingsEventDetection,
    SettingsEventPreprocessor,
    SettingsFrameAlignment,
    TargetsEventDetection,
    TargetsEventPreprocessors,
    TargetsFrameAlignment,
)
from elasticai.preprocessor.thresholding import (
    SettingsThreshold,
    TargetsThreshold,
    Thresholding,
)


@dataclass
class FrameWaveform:
    waveform: np.ndarray
    xpos: np.ndarray
    label: np.ndarray
    sampling_rate: float

    @property
    def length(self) -> int:
        return self.waveform.shape[1]

    @property
    def num_samples(self) -> int:
        return self.xpos.size

    @property
    def is_data_labeled(self) -> bool:
        return np.unique(self.label).size > 0 and 255 not in self.label.tolist()


@dataclass
class SettingsSDA:
    """Configuration class for defining the Spike Detection Algorithm (SDA)
    Attributes:
        mode_sda:       Applied spike detection algorithm (SDA) on transient signal [normal, absolute, Non-Linear Energy Operator (NEO) or Teager-Kaiser-Operator (dx_sda = 1 or kNEO with dx_sda > 1),
                        Multiresolution Teager Energy Operator (MTEO), absolute difference operator (ADO),
                        enhanced energy-derivation operator (eED),
                        amplitude slope operator (ASO, k for window size, and f_hp as additional float arg),
                        spike band-power estimation [Nason et al., 2020] (SBP, using f_bp with two values as additional arg)
        mode_thr:       String with used method for thresholding ['const': constant given value,
                        'abs_mean': absolute mean value, 'mad': median absolute derivation, 'mavg', moving average,
                        'mavg_abs': absolute mean absolute value, 'rms_norm': Root-Mean-Squared,
                        'rms_move': Moving RMS, 'rms_black': RMS method used in Blackrock Neurotechnology Systems,
                        'welford': Welford Online Algorithm for STD Calculation]
        mode_align:     Aligning mode of the detected spike frames [none, max, min,
                        ptp (Positive turning point), ntp (Negative turning point), abs-max (Absolute maximum)]
        sampling_rate:  Sampling rate [Hz]
        dx_sda:         Position difference for extracting SDA method. Configuration with length(x) == 1: with dX = 1 --> NEO, dX > 1 --> k-NEO
        t_frame_length: Floating value with total window length [s]
        t_frame_start:  Floating value with time point for aligned position [s]
        dt_offset:      Time offset for the first larger spike window [neg, pos]
        f_filt:         List with floating of the filter frequencies [Hz]
    """

    mode_sda: TargetsEventPreprocessors
    mode_thr: TargetsThreshold
    mode_align: TargetsFrameAlignment
    dx_sda: list
    sampling_rate: float
    t_frame_length: float
    t_frame_start: float
    dt_offset: float
    f_filt: list[float]

    @property
    def get_integer_offset(self) -> int:
        """Getting the integer offset for negative offset in building the spike window"""
        return round(self.dt_offset * self.sampling_rate)

    @property
    def get_integer_spike_frame(self) -> int:
        """Getting the integer for total length of a spike window"""
        return round(self.t_frame_length * self.sampling_rate)

    @property
    def get_integer_spike_start(self) -> int:
        """Getting the integer for starting the aligned method on each spike window"""
        return round(self.t_frame_start * self.sampling_rate)

    @property
    def get_integer_spike_total(self) -> int:
        """Getting the integer for total length of a spike window"""
        return self.get_integer_spike_frame + 2 * self.get_integer_offset


DefaultSettingsSDA = SettingsSDA(
    sampling_rate=20e3,
    dx_sda=[1],
    mode_sda=TargetsEventPreprocessors("eed"),
    mode_thr=TargetsThreshold("const"),
    mode_align=TargetsFrameAlignment("min"),
    t_frame_length=1.6e-3,
    t_frame_start=0.4e-3,
    dt_offset=0.1e-3,
    f_filt=[100.0],
)


class SpikeDetection:
    _settings: SettingsSDA
    _threshold: Thresholding
    _events: EventDetection
    _event_pre: EventPreprocessor
    _aligner: FrameAligner
    _logger: Logger

    def __init__(self, settings: SettingsSDA) -> None:
        """Class SpikeDetection for extracting Spike Waveforms from neural transient input
        :param settings:    Class SettingsSDA for configuring the accelerator
        :return:            None
        """
        self._logger: Logger = getLogger(__name__)
        self._settings = settings

        self._threshold = Thresholding(
            settings=SettingsThreshold(
                method=self._settings.mode_thr.value
                if isinstance(self._settings.mode_thr, TargetsThreshold)
                else self._settings.mode_thr,
                sampling_rate=self._settings.sampling_rate,
                window_sec=self._settings.t_frame_length,
                do_quant=False,
            )
        )
        self._event_pre = EventPreprocessor(
            settings=SettingsEventPreprocessor(
                type=self._settings.mode_sda,
                sampling_rate=self._settings.sampling_rate,
                window_size=self._settings.dx_sda,
                f_filt=self._settings.f_filt,
            )
        )
        self._events = EventDetection(
            settings=SettingsEventDetection(
                type=TargetsEventDetection("normal"), out_invert=False, window_size=1
            )
        )
        self._aligner = FrameAligner(
            settings=SettingsFrameAlignment(
                type=self._settings.mode_align,
                sampling_rate=self._settings.sampling_rate,
                align_sec=self._settings.t_frame_start,
                offset_sec=self._settings.dt_offset,
            )
        )

    def __frame_extraction(
        self, xraw: np.ndarray, xpos: np.ndarray | list, xoffset: int = 0
    ) -> FrameWaveform:
        def _in_bounds(start: int, end: int, size: int) -> bool:
            return start >= 0 and end <= size

        offset = self._settings.get_integer_offset
        spike_total = self._settings.get_integer_spike_total
        spike_frame = self._settings.get_integer_spike_frame

        alig_frames = list()
        alig_xpos = list()
        for pos in xpos:
            # Cutting larger frame from transient stream
            x_neg0: int = pos - offset + xoffset
            x_pos0: int = x_neg0 + spike_total
            if not _in_bounds(x_neg0, x_pos0, xraw.size):
                continue
            frame0 = xraw[x_neg0:x_pos0]

            # Cutting aligned frame from transient stream
            aligned_pos = self._aligner.get_aligned_position(frame0)[0]
            x_neg1: int = x_neg0 + aligned_pos
            x_pos1: int = x_neg1 + spike_frame
            if not _in_bounds(x_pos1, x_pos1, xraw.size):
                continue
            frame1 = xraw[x_neg1:x_pos1]

            alig_frames.append(frame1)
            alig_xpos.append(x_neg1)
        return FrameWaveform(
            waveform=np.array(alig_frames),
            xpos=np.array(alig_xpos),
            label=np.full(len(alig_xpos), 255, dtype=np.uint8),
            sampling_rate=self._settings.sampling_rate,
        )

    def get_frames(self, xraw: np.ndarray, **kwargs) -> FrameWaveform:
        """Function for extracting the spike waveforms from transient input
        :param xraw:    Numpy array with transient input
        :return:        Class FrameWaveform with waveforms, labels and position
        """
        sda = self._event_pre.get_preprocessed(xraw=xraw)
        thr = self._threshold.get_threshold(xin=sda, gain=1.0, **kwargs)
        xpos = self._events.get_events_position(xin=sda, threshold=thr)
        return self.__frame_extraction(xraw=xraw, xpos=xpos, xoffset=0)

    def get_frames_from_positions(
        self, xraw: np.ndarray, xpos: np.ndarray, xoffset: int
    ) -> FrameWaveform:
        """Function for extracting the spike waveforms from transient input and given position
        :param xraw:    Numpy array with transient input
        :param xpos:    Numpy array with positions where spike waveforms are available (ground truth)
        :param xoffset: Integer for shifting the xpos values
        :return:        Class FrameWaveform with waveforms, labels and position
        """
        return self.__frame_extraction(xraw=xraw, xpos=xpos, xoffset=xoffset)
