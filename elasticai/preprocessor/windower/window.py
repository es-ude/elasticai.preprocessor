from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from elasticai.preprocessor._basics import DataTransient, DataWaveform, PreprocessingModule
from elasticai.preprocessor.eventdetection.preprocessors import (
    EventPreprocessor,
    EventPreprocessorSettings,
    EventPreprocessorsTargets,
)
from elasticai.preprocessor.thresholding import Thresholding, ThresholdSettings, ThresholdTargets


class WindowerTargets(Enum):
    Sequence = "sequence"
    Sliding = "sliding"
    Event = "event"


@dataclass
class WindowSettings:
    """Class for defining the properties for applying a window on transient signals
    Attributes:
        method_window:  Windowing method (Sequence, Sliding, Event)
        method_thr:     Threshold method for event detection (only used for Event windower)
        method_input:   Input preprocessing method (Normal, Absolute, NEO, ...)
        sampling_rate:  Floating value with sampling rate of the transient signal [Hz]
        window_sec:     Floating value with the size of the window [s]
        overlap_sec:    Floating value with overlapping the sequences [s]
        pre_time:       Pre-event time in window before event is detected [s] (only for Event windower)
        threshold:      Absolute threshold value for event detection (only for Event windower)
    """

    sampling_rate: float
    window_sec: float
    overlap_sec: float
    method_window: WindowerTargets
    method_thr: ThresholdTargets = ThresholdTargets.Constant
    method_input: EventPreprocessorsTargets = EventPreprocessorsTargets.Normal
    pre_time: float = 0.0
    threshold: float = 10.0

    @property
    def window_length(self) -> int:
        """Returning an integer with total number of samples for building the window sequence"""
        assert self.window_sec > 0, "Window length must be greater than zero"
        return int(abs(self.window_sec * self.sampling_rate))

    @property
    def overlap_length(self) -> int:
        """Returning an integer with total number of samples for overlapping"""
        assert self.overlap_sec < self.window_sec, "Overlapping size should be smaller than window size"
        return int(abs(self.overlap_sec * self.sampling_rate))


DefaultWindowSettings = WindowSettings(
    method_window=WindowerTargets.Sequence,
    sampling_rate=1000.0,
    window_sec=0.1,
    overlap_sec=0.0,
)


class WindowSequencer(PreprocessingModule[DataTransient, DataWaveform]):
    _settings: WindowSettings

    def __init__(self, settings: WindowSettings) -> None:
        """Class for applying a window on transient signals
        :param settings:    Class SettingsWindow with definitions for the window
        :return:            None
        """
        super().__init__()
        self._settings = settings
        if isinstance(settings.method_window, str):
            self._settings.method_window = WindowerTargets(settings.method_window.lower())
        if isinstance(settings.method_thr, str):
            self._settings.method_thr = ThresholdTargets(settings.method_thr.lower())

        self._settings_thr = ThresholdSettings(
            method=self._settings.method_thr,
            sampling_rate=self._settings.sampling_rate,
            window_sec=self._settings.window_sec / 2,
            thr_val=settings.threshold,
            do_quant=False,
        )
        self._preprocessor = EventPreprocessor(
            EventPreprocessorSettings(
                method=settings.method_input,
                sampling_rate=settings.sampling_rate,
                window_size=[1],
                f_filt=[150.0],
            )
        )

    def _do_sequencing(self, signal: np.ndarray) -> np.ndarray:
        num_sequences = int(signal.shape[0] / self._settings.window_length)
        array_length = num_sequences * self._settings.window_length
        return signal[0:array_length].reshape((num_sequences, self._settings.window_length), copy=True)

    def _do_sliding(self, signal: np.ndarray) -> np.ndarray:
        delta_steps = self._settings.window_length - self._settings.overlap_length

        num_pre_padding = self._settings.window_length - delta_steps

        signal = np.pad(
            signal,
            (num_pre_padding, 0),
            mode="constant",
            constant_values=0,
        )

        return sliding_window_view(
            x=signal, axis=0, window_shape=self._settings.window_length, writeable=True
        )[::delta_steps]

    def _do_event_detection(
        self,
        signal: np.ndarray,
        thr: float,
        pre_time: float,
        signal_for_detection: np.ndarray | None = None,
    ) -> np.ndarray:
        signal_detect = signal_for_detection if signal_for_detection is not None else signal
        self._settings_thr.thr_val = thr
        xpos_event = Thresholding(settings=self._settings_thr).get_threshold_position(
            xin=signal_detect, pre_time=pre_time
        )

        if not xpos_event.tolist():
            return np.zeros((1, 1))
        else:
            sequence_window = np.zeros((len(xpos_event), self._settings.window_length))
            for ite, idx in enumerate(xpos_event):
                start_xpos = max(0, idx)
                num_pre_padding = max(0, -idx)
                stop_xpos = (
                    start_xpos + self._settings.window_length
                    if start_xpos + self._settings.window_length < signal.size
                    else -1
                )
                num_post_padding = (
                    0
                    if start_xpos + self._settings.window_length < signal.size
                    else abs(signal.size - start_xpos)
                )

                cutted_signal = signal[start_xpos + num_pre_padding : stop_xpos]
                if num_pre_padding:
                    pre_padding = (
                        np.zeros((self._settings.window_length - cutted_signal.size,)) + cutted_signal[0]
                    )
                    cutted_signal = np.concatenate((pre_padding, cutted_signal))

                if num_post_padding:
                    post_padding = (
                        np.zeros((self._settings.window_length - cutted_signal.size,)) + cutted_signal[-1]
                    )
                    cutted_signal = np.concatenate((cutted_signal, post_padding))

                sequence_window[ite, :] = cutted_signal
            return sequence_window

    def forward(self, x: np.ndarray) -> np.ndarray:
        match self._settings.method_window:
            case WindowerTargets.Sequence:
                return self._do_sequencing(x)
            case WindowerTargets.Sliding:
                return self._do_sliding(x)
            case WindowerTargets.Event:
                signal_preprocessed = self._preprocessor.forward(x)
                return self._do_event_detection(
                    signal=x,
                    thr=self._settings.threshold,
                    pre_time=self._settings.pre_time,
                    signal_for_detection=signal_preprocessed,
                )
            case _:
                raise AttributeError

    def __call__(self, x: DataTransient) -> DataWaveform:
        self._settings.sampling_rate = x.sampling_rate
        return DataWaveform(
            waveform=self.forward(x=x.data),
            xpos=np.asarray([0]),
            label=np.asarray([255]),
            sampling_rate=self._settings.sampling_rate,
        )

    def create_design(
        self,
        id: str,
        target: str,
        path2save: Path,
        total_bitwidth: int,
        frac_bitwidth: int = 0,
        is_signed: bool = True,
    ) -> None:
        supported_targets = ["mcu", "pc", "fpga"]
        target = target.lower()
        if target not in supported_targets:
            raise ValueError(f"Target {target} is not supported: only {supported_targets}")

        if target in ["mcu", "pc"]:
            self._create_design_c(
                id=id,
                bitwidth=total_bitwidth,
                signed=is_signed,
                path2save=path2save,
            )
        else:
            self._create_design_verilog(
                id=id,
                bitwidth=total_bitwidth,
                signed=is_signed,
                path2save=path2save,
            )

    def _create_design_verilog(self, id: str, bitwidth: int, signed: bool, path2save: Path) -> None:
        import elasticai.creator_plugins.windower as hw_windower

        window_length = self._settings.window_length
        overlap_length = self._settings.overlap_length
        num_shift = window_length - overlap_length

        params = {
            "type": "windower",
            "id": id,
            "params": {
                "BITWIDTH": bitwidth,
                "SAMPLES": window_length,
                "NUM_SHIFT": num_shift,
            },
            "add_ringbuffer": False,
        }

        hw_windower.load_and_plugin(
            packages=["windower"],
            path2save=path2save,
            **params,
        )

    def _create_design_c(
        self,
        id: str,
        bitwidth: int,
        signed: bool,
        path2save: Path,
    ) -> None:
        from elasticai.creator_plugins.windower.src import c_compile

        match self._settings.method_window:
            case WindowerTargets.Sequence:
                c_compile.build_windower_sequence(
                    settings=self._settings,
                    bitwidth=bitwidth,
                    signed=signed,
                    path2save=path2save,
                    windower_id=id,
                    define_path=".",
                )
            case WindowerTargets.Sliding:
                c_compile.build_windower_sliding(
                    settings=self._settings,
                    bitwidth=bitwidth,
                    signed=signed,
                    path2save=path2save,
                    windower_id=id,
                    define_path=".",
                )
            case WindowerTargets.Event:
                c_compile.build_windower_event_pipeline(
                    settings=self._settings,
                    bitwidth=bitwidth,
                    signed=signed,
                    path2save=path2save,
                    windower_id=id,
                    define_path=".",
                )
            case _:
                raise NotImplementedError(
                    f"window_method '{self._settings.method_window.value}' does not have a C implementation yet."
                )
