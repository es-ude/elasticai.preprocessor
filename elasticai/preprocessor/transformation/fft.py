import numpy as np

from .transformation import Transformation, TransformationSettings, TransformationTargets
from .window_scaler import WindowingTargets


def do_fft(
    y: np.ndarray, sampling_rate: float, method_window: str | WindowingTargets = WindowingTargets.Hamming
) -> tuple[np.ndarray, np.ndarray]:
    """Performing the Discrete Fast Fourier Transformation.
    :param y:               Transient input signal
    :param sampling_rate:   Sampling rate [Hz]
    :param method_window:   Selected window ['ones', 'Hamming', 'hanning', 'guassian', 'bartlett', 'blackman']
    :return:                Tuple with (1) freq - Frequency and (2) Y - Discrete output
    """
    data = Transformation(
        settings=TransformationSettings(
            method=TransformationTargets.FFT, method_window=method_window, sampling_rate=sampling_rate
        )
    )._do_fft(x=y)
    return data.freq, data.data


def do_fft_withimag(
    y: np.ndarray, sampling_rate: float, method_window: str | WindowingTargets = WindowingTargets.Ones
) -> tuple[np.ndarray, np.ndarray]:
    """Performing the Discrete Fast Fourier Transformation with imaginary part.
    :param y:               Transient input signal
    :param sampling_rate:   Sampling rate [Hz]
    :param method_window:   Selected window ['ones', 'Hamming', 'hanning', 'guassian', 'bartlett', 'blackman']
    :return:                Tuple with (1) freq - Frequency and (2) Y - Discrete output
    """
    data = Transformation(
        settings=TransformationSettings(
            method=TransformationTargets.imagFFT, method_window=method_window, sampling_rate=sampling_rate
        )
    )._do_fft_with_imag(x=y)
    return data.freq, data.data


def do_fft_inverse(y: np.ndarray, len_original: int) -> np.ndarray:
    """Perform inverse real FFT.
    :param y:               Fourier domain signal
    :param len_original:    Length of original time domain signal
    :return:                Time domain signal
    """
    if "complex" not in y.dtype.name:
        raise ValueError("Only complex numpy arrays numbers are allowed")
    return np.fft.irfft(y, n=len_original)
