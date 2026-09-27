from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal.windows import gaussian

import elasticai.creator_plugins.windower as hw_windower
from elasticai.preprocessor._check_funcs import check_key_elements
from elasticai.preprocessor.eventdetection import (
    EventPreprocessor,
    SettingsEventPreprocessor,
    TargetsEventPreprocessors,
)
from elasticai.preprocessor.thresholding import SettingsThreshold, TargetsThreshold, Thresholding

if TYPE_CHECKING:
    from elasticai.creator_plugins.windower.src.c_compile.pipeline_settings import (
        SettingsPipelineDownsampling,
        SettingsPipelineFilter,
    )


def transformation_window_method(window_size: int, method: str = "hamming") -> np.ndarray:
    """Generating window for smoothing input of signal transformation method.
    :param window_size:     Integer number with size of the window
    :param method:          Selection of window method ['': Ones, 'hamming', 'hanning', 'gaussian', 'bartlett', 'blackman']
    :return:                Numpy array with window
    """
    methods_avai = {
        "":         np.ones(window_size),
        "hamming":  np.hamming(window_size),
        "gaussian": gaussian(window_size, int(0.16 * window_size), sym=True),
        "hanning":  np.hanning(window_size),
        "bartlett": np.bartlett(window_size),
        "blackman": np.blackman(window_size),
    }
    methods_check = [method.lower() for method in methods_avai.keys()]
    if not check_key_elements(method.lower(), methods_check):
        raise ValueError(f"Wrong method ({methods_check})")
    return methods_avai[[key for key in methods_check if key == method.lower()][0]]


class TargetsWindower(Enum):
    Sequence = "sequence"
    Sliding  = "sliding"
    Event    = "event"


@dataclass
class SettingsWindow:
    """Class for defining the properties for applying a window on transient signals
    Attributes:
        method_window:      TargetsWindower [Sequence, Sliding, Event]
        method_thr:         TargetsThreshold — threshold method for event detection
        method_input:       TargetsEventPreprocessors — signal transformation before thresholding
                            (e.g. Normal, Absolute, NEO, MTEO, ADO, ASO, EED, SBP)
        sampling_rate:      Floating value with sampling rate of the transient signal [Hz]
        window_sec:         Floating value with the size of the window [s]
        overlap_sec:        Floating value with overlapping the sequences [s]
        pre_time:           Pre-event time included in each returned window [s]
        threshold:          Threshold value (used directly for Constant, as starting point for others)
        window_size:        Shift distances k for the event preprocessor (e.g. [1] for NEO, [1,2,3] for MTEO)
        f_filt:             Filter frequencies for event preprocessor methods that need them (ASO, SBP) [Hz]
        method_transform:   Window shape applied to preprocessed signal before thresholding
                            ['': Ones (no smoothing), 'hamming', 'hanning', 'gaussian', 'bartlett', 'blackman']
    """

    method_window:    TargetsWindower
    method_thr:       TargetsThreshold
    method_input:     TargetsEventPreprocessors
    sampling_rate:    float
    window_sec:       float
    overlap_sec:      float
    pre_time:         float
    threshold:        float
    window_size:      list[int]       = field(default_factory=lambda: [1])
    f_filt:           list[float]     = field(default_factory=lambda: [100.0])
    method_transform: str             = ""

    @property
    def window_length(self) -> int:
        """Total number of samples per window"""
        assert self.window_sec > 0, "Window length must be greater than zero"
        return int(abs(self.window_sec * self.sampling_rate))

    @property
    def overlap_length(self) -> int:
        """Number of samples that consecutive windows overlap"""
        assert self.overlap_sec < self.window_sec, "Overlapping size should be smaller than window size"
        return int(abs(self.overlap_sec * self.sampling_rate))


DefaultSettingsWindow = SettingsWindow(
    method_window=TargetsWindower.Event,
    method_thr=TargetsThreshold.Constant,
    method_input=TargetsEventPreprocessors.NEO,
    sampling_rate=2e3,
    window_sec=0.1,
    overlap_sec=0.0,
    pre_time=0.01,
    threshold=10.0,
)


class WindowSequencer:
    _settings: SettingsWindow
    _event_pre: EventPreprocessor
    _thresholding: Thresholding
    _transform_window: np.ndarray
    _transform_window_norm: np.ndarray

    def __init__(self, settings: SettingsWindow) -> None:
        """Class for applying a window on transient signals
        :param settings:    Class SettingsWindow with all configuration
        :return:            None
        """
        self._settings = settings

        # EventPreprocessor for signal transformation before event detection.
        # Its output is used only for thresholding — the returned windows always
        # contain the raw (untransformed) signal.
        self._event_pre = EventPreprocessor(
            settings=SettingsEventPreprocessor(
                type=settings.method_input,
                sampling_rate=settings.sampling_rate,
                window_size=settings.window_size,
                f_filt=settings.f_filt,
            )
        )

        # Window function for smoothing the preprocessed signal before thresholding.
        # Convolved with the preprocessed signal to form a smooth energy envelope.
        # With method_transform="" this is all-ones → rectangular moving average.
        self._transform_window = transformation_window_method(
            window_size=settings.window_length,
            method=settings.method_transform,
        )
        self._transform_window_norm = self._transform_window / self._transform_window.sum()

        self._thresholding = Thresholding(
            settings=SettingsThreshold(
                method=settings.method_thr,
                sampling_rate=settings.sampling_rate,
                window_sec=settings.window_sec / 2,
                thr_val=settings.threshold,
                do_quant=False,
            )
        )

    def get_windowed(self, signal: np.ndarray) -> np.ndarray:
        """Main interface: dispatch windowing based on method_window setting.
        :param signal:  Numpy array with raw input signal, shape=(N,)
        :return:        Numpy array of windows, shape=(M, window_length)
        """
        match self._settings.method_window:
            case TargetsWindower.Sequence | TargetsWindower.Sliding:
                return self.slide(signal)
            case TargetsWindower.Event:
                return self.window_event_detected(
                    signal=signal,
                    thr=self._settings.threshold,
                    pre_time=self._settings.pre_time,
                )
            case _:
                raise ValueError(f"Unknown method_window: {self._settings.method_window}")

    def sequence(self, signal: np.ndarray) -> np.ndarray:
        """Cut signal into non-overlapping consecutive windows.
        Equivalent to slide() with overlap=0; kept as named convenience method.
        :param signal:  Numpy array with input signal, shape=(N,)
        :return:        Numpy array of windows, shape=(M, window_length)
        """
        original_overlap = self._settings.overlap_sec
        self._settings.overlap_sec = 0.0
        result = self.slide(signal)
        self._settings.overlap_sec = original_overlap
        return result

    def slide(self, signal: np.ndarray) -> np.ndarray:
        """Build overlapping sliding windows over signal.
        :param signal:  Numpy array with input signal, shape=(N,)
        :return:        Numpy array of windows, shape=(M, window_length)
        """
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

    def window_event_detected(self, signal: np.ndarray, thr: float, pre_time: float) -> np.ndarray:
        """Detect events and return windows cut from the raw signal.

        Pipeline:
          1. Apply EventPreprocessor (method_input) to raw signal         → x_pre
          2. Smooth x_pre with transformation_window_method via convolution → x_detect
          3. Threshold x_detect to find event positions
          4. Cut windows from the ORIGINAL raw signal at those positions

        The transformation is only used for detection — the returned array
        always contains unmodified raw signal samples.

        :param signal:      Numpy array with raw input signal, shape=(N,)
        :param thr:         Threshold value (overrides settings.threshold for this call)
        :param pre_time:    Pre-event time [s] included before each event in the window
        :return:            Numpy array of raw windows, shape=(M, window_length),
                            or shape=(1, 1) if no events found
        """
        # Step 1: preprocess signal for detection (NOT returned)
        x_pre = self._event_pre.get_preprocessed(xraw=signal)

        # Step 2: optionally smooth with transformation_window_method (NOT returned).
        # Only applied when a non-trivial window method is configured;
        # with method_transform="" the preprocessed signal is used directly.
        if self._settings.method_transform:
            x_detect = np.convolve(x_pre, self._transform_window_norm, mode="same")
        else:
            x_detect = x_pre

        # Step 3: threshold
        self._thresholding._settings.thr_val = thr
        xpos_event = self._thresholding.get_threshold_position(xin=x_detect, pre_time=pre_time)

        if not xpos_event.tolist():
            return np.zeros((1, 1))

        # Step 4: cut windows from raw signal
        sequence_window = np.zeros((len(xpos_event), self._settings.window_length))
        num_samples_pre = int(pre_time * self._settings.sampling_rate)

        for ite, idx in enumerate(xpos_event):
            start_xpos = idx - num_samples_pre if idx - num_samples_pre > 0 else idx
            num_pre_padding = 0 if idx - num_samples_pre > 0 else abs(idx - num_samples_pre)
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
                pre_padding = np.zeros((self._settings.window_length - cutted_signal.size,)) + cutted_signal[0]
                cutted_signal = np.concatenate((pre_padding, cutted_signal))

            if num_post_padding:
                post_padding = np.zeros((self._settings.window_length - cutted_signal.size,)) + cutted_signal[-1]
                cutted_signal = np.concatenate((cutted_signal, post_padding))

            sequence_window[ite, :] = cutted_signal

        return sequence_window

    def create_design(
        self,
        target: str,
        bitwidth: int,
        id: str,
        path2save: Path,
        signed: bool = True,
        threshold: int | None = None,
        pre_samples: int | None = None,
    ) -> None:
        """Create a target-specific windower design.
        :param target:      String with target name ["mcu", "pc", "fpga"]
        :param bitwidth:    Integer with total bitwidth
        :param id:          String with unique identifier of device (appended to the name)
        :param path2save:   Path to save the hardware files
        :param signed:      Whether generated C designs use a signed integer data type
        :param threshold:   Override threshold for C design (defaults to settings.threshold)
        :param pre_samples: Override pre-event samples for C design (defaults from settings.pre_time)
        :return:            None
        """
        supported_targets = ["mcu", "pc", "fpga"]
        target = target.lower()
        if target not in supported_targets:
            raise ValueError(f"Target {target} is not supported: only {supported_targets}")

        thr_int = threshold if threshold is not None else int(self._settings.threshold)
        pre_int = pre_samples if pre_samples is not None else int(
            self._settings.pre_time * self._settings.sampling_rate
        )

        if target in ["mcu", "pc"]:
            self._create_design_c(
                id=id,
                bitwidth=bitwidth,
                signed=signed,
                path2save=path2save,
                threshold=thr_int,
                pre_samples=pre_int,
            )
        else:
            self._create_design_verilog(
                id=id,
                bitwidth=bitwidth,
                signed=signed,
                path2save=path2save,
            )

    def create_pipeline_design(
        self,
        filter_settings: SettingsPipelineFilter,
        downsampling_settings: SettingsPipelineDownsampling,
        bitwidth: int,
        id: str,
        path2save: Path,
        signed: bool = True,
        define_path: str = "src",
    ) -> None:
        """Generate C files for the complete streaming pipeline (filter → decimation → windower).

        Uses the window settings from this WindowSequencer instance (window_length,
        overlap → num_shift) together with the supplied filter and decimation
        configuration to call build_pipeline_from_settings().

        :param filter_settings:      SettingsPipelineFilter — filter stage configuration.
        :param downsampling_settings: SettingsPipelineDownsampling — decimation stage config.
        :param bitwidth:             Bit width of each sample (2..32).
        :param id:                   Unique ID appended to generated function names.
        :param path2save:            Directory where generated files are written.
        :param signed:               Whether the C data type is signed.
        :param define_path:          Include prefix in the generated #include lines.

        Example::

            from elasticai.creator_plugins.windower.src.c_compile import (
                SettingsPipelineFilter, SettingsPipelineDownsampling,
                TargetsFilterC, TargetsDownsamplingC,
            )
            sequencer.create_pipeline_design(
                filter_settings=SettingsPipelineFilter(
                    method=TargetsFilterC.IIR, iir_a=coeffs.a, iir_b=coeffs.b,
                ),
                downsampling_settings=SettingsPipelineDownsampling(
                    method=TargetsDownsamplingC.PolyOne, ratio=4,
                ),
                bitwidth=32, id="0", path2save=Path("build/"),
            )
        """
        from elasticai.creator_plugins.windower.src.c_compile import (
            SettingsPipeline,
            build_pipeline_from_settings,
        )
        build_pipeline_from_settings(
            settings=SettingsPipeline(
                filter=filter_settings,
                downsampling=downsampling_settings,
                window=self._settings,
                bitwidth=bitwidth,
                signed=signed,
            ),
            path2save=path2save,
            pipeline_id=id,
            define_path=define_path,
        )

    def _create_design_verilog(self, id: str, bitwidth: int, signed: bool, path2save: Path) -> None:
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
            "add_ringbuffer": True,
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
        threshold: int = 0,
        pre_samples: int = 0,
    ) -> None:
        from elasticai.creator_plugins.windower.src import c_compile
        match self._settings.method_window:
            case TargetsWindower.Sequence:
                c_compile.build_windower_sequence(
                    settings=self._settings,
                    bitwidth=bitwidth,
                    signed=signed,
                    path2save=path2save,
                    num_shift=0,
                    windower_id=id,
                    define_path=".",
                )
            case TargetsWindower.Sliding:
                c_compile.build_windower_sliding(
                    settings=self._settings,
                    bitwidth=bitwidth,
                    signed=signed,
                    path2save=path2save,
                    windower_id=id,
                    define_path=".",
                )
            case TargetsWindower.Event:
                c_compile.build_windower_event(
                    settings=self._settings,
                    threshold=threshold,
                    pre_padding=pre_samples,
                    bitwidth=bitwidth,
                    signed=signed,
                    path2save=path2save,
                    windower_id=id,
                    define_path=".",
                )
            case _:
                raise NotImplementedError(
                    f"method_window '{self._settings.method_window}' has no C implementation yet."
                )
