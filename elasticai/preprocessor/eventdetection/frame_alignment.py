from dataclasses import dataclass
from enum import Enum
from logging import Logger, getLogger

import numpy as np


class TargetsFrameAlignment(Enum):
    Normal = "none"
    Max = "max"
    Min = "min"
    PositiveTurning = "ptp"
    NegativeTurning = "ntp"
    AbsMax = "absmax"
    AbsMin = "absmin"


@dataclass
class SettingsFrameAlignment:
    """Class with settings for the FrameGenerator to configure his properties
    Attributes:
        type:           Aligning mode of the detected spike frames [none, max, min,
                        ptp (Positive turning point), ntp (Negative turning point), abs-max (Absolute maximum)]
        sampling_rate:  Sampling rate of the transient signal [Hz]
        align_sec:      Starting position for aligning the frame waveform [s]
        offset_sec:     Offset for aligning the frame waveform [s]
    """

    type: TargetsFrameAlignment
    sampling_rate: float
    offset_sec: float
    align_sec: float

    @property
    def length_align_position(self) -> int:
        return int(self.align_sec * self.sampling_rate)

    @property
    def length_offset_int(self) -> int:
        return int(self.offset_sec * self.sampling_rate)


DefaultSettingsFrameAlignment = SettingsFrameAlignment(
    type="max",
    sampling_rate=20e3,
    align_sec=0.4e-3,
    offset_sec=0.4e-3,
)


class FrameAligner:
    def __init__(self, settings: SettingsFrameAlignment) -> None:
        """Class for aligning frame waveforms after event detection
        :param settings: Class SettingsFrameAlignment for defining the properties
        """
        self._logger: Logger = getLogger(__name__)
        self._settings = settings
        if isinstance(settings.type, str):
            self._settings.type = TargetsFrameAlignment(settings.type)

    def _frame_align_none(self, frame_in: np.ndarray) -> int:
        return self._settings.length_offset_int

    def _frame_align_max(self, frame_in: np.ndarray) -> int:
        x_start = np.argmax(frame_in, axis=0)
        return int(x_start - self._settings.length_align_position)

    def _frame_align_min(self, frame_in: np.ndarray) -> int:
        x_start = np.argmin(frame_in, axis=0)
        return int(x_start - self._settings.length_align_position)

    def _frame_align_ptp(self, frame_in: np.ndarray) -> int:
        frame_diff = np.diff(frame_in)
        x_start = 1 + np.argmax(frame_diff, axis=0)
        return int(x_start - self._settings.length_align_position)

    def _frame_align_ntp(self, frame_in: np.ndarray) -> int:
        frame_diff = np.diff(frame_in)
        x_start = 1 + np.argmin(frame_diff, axis=0)
        return int(x_start - self._settings.length_align_position)

    def _frame_align_absmax(self, frame_in: np.ndarray) -> int:
        frames_abs = np.abs(frame_in)
        x_max = np.argmax(frames_abs, axis=0)
        return int(x_max - self._settings.length_align_position)

    def _frame_align_absmin(self, frame_in: np.ndarray) -> int:
        frames_abs = np.abs(frame_in)
        x_max = np.argmin(frames_abs, axis=0)
        return int(x_max - self._settings.length_align_position)

    def _get_methods(self) -> list:
        split_key = "_frame_align_"
        return [method.split(split_key)[-1] for method in dir(self) if split_key in method]

    def get_aligned_position(self, frame_in: np.ndarray) -> list[int]:
        """Extracting aligning position of spike frames
        :param frame_in:    Numpy array with detected spike frames
        :return:            List with integer of starting positions
        """
        method = f"_frame_align_{self._settings.type.value.lower()}"
        if method in self._get_methods():
            raise ValueError(
                f"Frame Aligning Method '{self._settings.type.value.lower()}' is not in {self._get_methods()}. Please change!"
            )

        num_trials = frame_in.shape[0] if len(frame_in.shape) > 1 else 1
        frames_out = list()
        if num_trials == 1:
            frames_out.append(getattr(self, method)(frame_in))
        else:
            for i in range(num_trials):
                frames_out.append(getattr(self, method)(frame_in[i,]))
        return frames_out

    def create_design(self) -> None:
        raise NotImplementedError
