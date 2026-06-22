from unittest import TestCase, main

import numpy as np
from scipy.signal import find_peaks

from elasticai.preprocessor._check_funcs import is_close

from .transformation import do_fft, do_fft_inverse, do_fft_withimag


class TestTransformation(TestCase):
    time = np.linspace(start=0, stop=100e-3, num=2000, endpoint=False, dtype=float)
    vsig = np.sin(2 * np.pi * 100.0 * time) + 0.25 * np.sin(2 * np.pi * 1000.0 * time)
    fs = float(1 / np.diff(time).min())

    def test_fft_spec(self):
        freq, spec = do_fft(y=self.vsig, fs=self.fs, method_window="hamming")
        chck = freq.size == self.time.size / 2 and is_close(freq[-1], self.fs / 2, 50.0)
        self.assertTrue(chck)

    def test_fft_value(self):
        freq, spec = do_fft(y=self.vsig, fs=self.fs, method_window="hamming")
        x = find_peaks(x=spec, wlen=2)
        chck_amp = all(
            [is_close(spec[pred_frq], true_amp, 0.1) for pred_frq, true_amp in zip(x[0], [1.0, 0.25])]
        )
        chck_frq = all(
            [is_close(freq[pred_frq], true_frq, 0.1) for pred_frq, true_frq in zip(x[0], [100.0, 1000.0])]
        )
        self.assertTrue(chck_amp)
        self.assertTrue(chck_frq)

    def test_fft_imag(self):
        freq, spec = do_fft_withimag(y=self.vsig, fs=self.fs, method_window="hamming")
        self.assertTrue(freq.size == self.time.size / 2 + 1)
        self.assertTrue(is_close(freq[-1], self.fs / 2, 50.0))

    def test_ifft_spec(self):
        freq, spec = do_fft_withimag(y=self.vsig, fs=self.fs, method_window="")
        sig = do_fft_inverse(y=spec, len_original=self.time.size)
        errors = np.sum(np.abs(sig - self.vsig)) / sig.size
        self.assertLessEqual(errors, 1e-12)

    def test_ifft_spec_non_complex(self):
        freq, spec = do_fft(y=self.vsig, fs=self.fs, method_window="")
        try:
            do_fft_inverse(y=spec, len_original=self.time.size)
        except ValueError:
            self.assertTrue(True)
        else:
            self.assertTrue(False)


if __name__ == "__main__":
    main()
