from enum import Enum

import numpy as np
from scipy.signal.windows import gaussian


class WindowingTargets(Enum):
    Ones = "ones"
    Hamming = "hamming"
    Gaussian = "gaussian"
    Hanning = "hanning"
    Bartlett = "bartlett"
    Blackman = "blackman"


def transformation_window_method(
    window_size: int, method: WindowingTargets | str = WindowingTargets.Hamming
) -> np.ndarray:
    """Generating window for smoothing input of signal transformation method.
    :param window_size: Integer number with size of the window
    :param method:      Selection of window method [
                            'ones',
                            'hamming',
                            'hanning',
                            'gaussian',
                            'bartlett',
                            'blackman']
    :return:            Numpy array with window
    """
    if isinstance(method, str):
        method_used = WindowingTargets(method.lower())
    else:
        method_used = method

    match method_used:
        case WindowingTargets.Ones:
            return np.ones(window_size)
        case WindowingTargets.Hamming:
            return np.hamming(window_size)
        case WindowingTargets.Gaussian:
            return gaussian(window_size, int(0.16 * window_size), sym=True)
        case WindowingTargets.Hanning:
            return np.hanning(window_size)
        case WindowingTargets.Bartlett:
            return np.bartlett(window_size)
        case WindowingTargets.Blackman:
            return np.blackman(window_size)
        case _:
            raise AttributeError
