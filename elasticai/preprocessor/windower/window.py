from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal.windows import gaussian

import elasticai.creator_plugins.windower as hw_windower
from elasticai.preprocessor._check_funcs import check_key_elements
from elasticai.preprocessor.thresholding import SettingsThreshold, Thresholding


def transformation_window_method(window_size: int, method: str = "hamming") -> np.ndarray:
    """Generating window for smoothing input of signal transformation method.
    :param window_size:     Integer number with size of the window
    :param method:          Selection of window method ['': Ones, 'hamming', 'hanning', 'gaussian', 'bartlett', 'blackman']
    :return:                Numpy array with window
    """
    methods_avai = {
        "": np.ones(window_size),
        "hamming": np.hamming(window_size),
        "gaussian": gaussian(window_size, int(0.16 * window_size), sym=True),
        "hanning": np.hanning(window_size),
        "bartlett": np.bartlett(window_size),
        "blackman": np.blackman(window_size),
    }
    methods_check = [method.lower() for method in methods_avai.keys()]
    assert check_key_elements(method.lower(), methods_check), f"Wrong method ({methods_check})"
    return methods_avai[[key for key in methods_check if key == method.lower()][0]]


@dataclass
class SettingsWindow:
    """Class for defining the properties for applying a window sequenzer on transient signals
    Attributes:
        sampling_rate:  Floating value with sampling rate of the transient signal [Hz]
        window_sec:     Floating value with the size of the window [s]
        overlap_sec:    Floating value with overlapping the sequences [s]
    """

    sampling_rate: float
    window_sec: float
    overlap_sec: float

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


DefaultSettingsWindow = SettingsWindow(sampling_rate=2e3, window_sec=0.1, overlap_sec=0.0)


class WindowSequencer:
    _settings: SettingsWindow
    _window_normalization: np.ndarray

    def __init__(self, settings: SettingsWindow) -> None:
        """Class for applying a window sequenzer on transient signals
        :param settings:    Class SettingsWindow with definitions for the window sequenzer
        :return:            None
        """
        self._settings = settings
        self._settings_thr = SettingsThreshold(
            method="const",
            module_type="",
            sampling_rate=self._settings.sampling_rate,
            gain=1.0,
            window_sec=self._settings.window_length / 2,
        )
        self._window_normalization = transformation_window_method(
            window_size=self._settings.window_length, method=""
        )

    def sequence(self, signal: np.ndarray) -> np.ndarray:
        """Building a sequence-to-sequence output array from signal input
        :param signal:  Numpy array with input signal to build the sequence with shape=(N, )
        :return:        Numpy array of sequence signals with shape=(M, window length)
        """
        num_sequences = int(signal.shape[0] / self._settings.window_length)
        array_length = num_sequences * self._settings.window_length
        return signal[0:array_length].reshape((num_sequences, self._settings.window_length), copy=True)

    def slide(self, signal: np.ndarray) -> np.ndarray:
        """Building a sliding window sequencer on signal input
        :param signal:  Numpy array with input signal to build the sequence with shape=(N, )
        :return:        Numpy array of sequence signals with shape=(M, window length)
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

    def window_event_detected(
        self, signal: np.ndarray, thr: float, pre_time: float, do_abs: bool = False
    ) -> np.ndarray:
        """Building a window sequencer based on an event-detection (absolute input)
        :param signal:      Numpy array with input signal to build the sequence with shape=(N, )
        :param thr:         Floating value with absolute threshold value
        :param pre_time:    Floating value with pre-time in the window before event is detected
        :param do_abs:      Boolean for applying absolute signal to threshold calculation
        :return:            Numpy array of sequence signals with shape=(M, window length)
        """
        if thr < 0.0:
            raise ValueError("Threshold must be positive")

        xpos_event = Thresholding(settings=self._settings_thr).get_threshold_position(
            xin=signal, pre_time=pre_time, do_abs=do_abs, thr_val=thr
        )

        if not xpos_event.tolist():
            return np.zeros((1, 1))
        else:
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

    def create_design(
        self,
        target: str,
        bitwidth: int,
        id: str,
        path2save: Path,
        signed: bool = True,
    ) -> None:
        """Create a target-specific windower design.
        :param target:      String with target name ["mcu", "pc", "fpga"]
        :param bitwidth:    Integer with total bitwidth
        :param id:          String with unique identifier of device (appended to the name)
        :param path2save:   Path to save the hardware files
        :param signed:      Whether generated C designs use a signed integer data type
        :return:            None
        """
        supported_targets = ["mcu", "pc", "fpga"]
        target = target.lower()
        if target not in supported_targets:
            raise ValueError(f"Target {target} is not supported: only {supported_targets}")

        if target in ["mcu", "pc"]:
            self._create_design_c(
                id=id,
                bitwidth=bitwidth,
                signed=signed,
                path2save=path2save,
            )
        else:
            self._create_design_verilog(
                id=id,
                bitwidth=bitwidth,
                signed=signed,
                path2save=path2save,
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

    def _create_design_c(self, id: str, bitwidth: int, signed: bool, path2save: Path) -> None:
        raise NotImplementedError
