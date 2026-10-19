from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from elasticai.preprocessor._basics import DataSpectral, DataTransient, PreprocessingModule

from .window_scaler import WindowingTargets, transformation_window_method


class TransformationTargets(Enum):
    FFT = "fft"
    imagFFT = "fft_imag"


@dataclass
class TransformationSettings:
    """Class for defining the properties for applying a window on transient signals
    Attributes:
        method:         Selected method for transformation (FFT, FFT with imag)
        method_window:  Selected method for windowing (ones, hamming, hanning, ...)
        sampling_rate:  Floating value with sampling rate of the transient signal [Hz]
    """

    method: str | TransformationTargets
    method_window: str | WindowingTargets
    sampling_rate: float


TransformationSettingsDefault = TransformationSettings(
    method=TransformationTargets.FFT, method_window=WindowingTargets.Hamming, sampling_rate=1000.0
)


class Transformation(PreprocessingModule[DataTransient, DataSpectral]):
    _settings: TransformationSettings

    def __init__(self, settings: TransformationSettings) -> None:
        """Class for transforming the transient input to spectral output or inverse process
        :param settings:    Settings for defining the properties of the Transformation Module
        :return:            None
        """
        super().__init__()
        self._settings = settings
        if isinstance(settings.method, str):
            self._settings.method = TransformationTargets(settings.method.lower())
        if isinstance(settings.method_window, str):
            self._settings.method_window = WindowingTargets(settings.method_window.lower())

    def _do_fft(self, x: np.ndarray) -> DataSpectral:
        window = transformation_window_method(window_size=x.size, method=self._settings.method_window)
        fft_in = window * x

        N = x.size // 2
        fft_out = 2 / N * np.abs(np.fft.fft(fft_in))
        fft_out[0] = fft_out[0] / 2
        freq = self._settings.sampling_rate * np.fft.fftfreq(fft_out.size)

        xsel = np.where(freq >= 0.0)
        fft_out = fft_out[xsel]
        freq = freq[xsel]
        return DataSpectral(data=fft_out, freq=freq, sampling_rate=self._settings.sampling_rate)

    def _do_fft_with_imag(self, x: np.ndarray) -> DataSpectral:
        window = transformation_window_method(window_size=x.size, method=self._settings.method_window)
        fft_in = window * x

        fft_out = np.fft.rfft(fft_in)
        freq = np.fft.rfftfreq(fft_in.size, d=1 / self._settings.sampling_rate)
        return DataSpectral(data=fft_out, freq=freq, sampling_rate=self._settings.sampling_rate)

    def _forward_full(self, x: np.ndarray) -> DataSpectral:
        match self._settings.method:
            case TransformationTargets.FFT:
                return self._do_fft(x=x)
            case TransformationTargets.imagFFT:
                return self._do_fft_with_imag(x=x)
            case _:
                raise AttributeError

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self._forward_full(x=x).data

    def __call__(self, x: DataTransient) -> DataSpectral:
        self._settings.sampling_rate = x.sampling_rate
        return self._forward_full(x=x.data)

    def create_design(
        self,
        id: str,
        target: str,
        path2save: Path,
        total_bitwidth: int,
        frac_bitwidth: int = 0,
        is_signed: bool = True,
    ) -> None:
        raise NotImplementedError
