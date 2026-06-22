from copy import deepcopy
from pathlib import Path
from shutil import which
from subprocess import run
from tempfile import TemporaryDirectory
from unittest import TestCase, main, skipUnless

import numpy as np
from scipy.signal import find_peaks

from elasticai.preprocessor import get_path_to_project
from elasticai.preprocessor.transformation import do_fft

from .filtering import Filtering, SettingsFilter

test_settings = SettingsFilter(
    gain=1,
    fs=1e3,
    n_order=2,
    f_filt=[250],
    type="iir",
    f_type="butter",
    b_type="lowpass",
)


def extract_peaks(signal: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    freq, trans = do_fft(y=signal, fs=fs)
    peakx, _ = find_peaks(x=trans, height=0.05)
    return freq[peakx], trans[peakx], peakx


class TestDigitalFilters(TestCase):
    time = np.linspace(
        start=0.0,
        stop=5.0,
        num=int(5.0 * test_settings.fs),
        endpoint=False,
        dtype=float,
    )
    freq = [10.0, 20.0, 50.0, 100.0, 200.0]

    def assert_c_compiles(self, project_dir: Path, source: Path) -> None:
        run(
            [
                "cc",
                "-std=c11",
                f"-I{project_dir}",
                "-c",
                str(source),
                "-o",
                str(source.with_suffix(".o")),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_signal_generation(self):
        signal = np.sum([np.sin(2 * np.pi * f0 * self.time) for f0 in self.freq], axis=0)
        f0, y0, _ = extract_peaks(signal, test_settings.fs)
        assert self.freq == f0.tolist()
        np.testing.assert_almost_equal(
            y0, [1.07953976, 1.07954007, 1.07954007, 1.07954005, 1.07954001], decimal=3
        )

    def test_lowpass_iir_first_order(self):
        signal = np.sum([np.sin(2 * np.pi * f0 * self.time) for f0 in self.freq], axis=0)
        sets = deepcopy(test_settings)
        sets.type = "iir"
        sets.n_order = 1
        sets.b_type = "lowpass"
        sets.f_filt = [50.0]
        result = Filtering(sets).filt(signal)

        freq0, peak0, pos = extract_peaks(signal, sets.fs)
        freq1, peak1 = do_fft(result, sets.fs)
        assert freq0.tolist() == freq1[pos].tolist()
        gain = np.array(peak1[pos]) / np.array(peak0)
        np.testing.assert_almost_equal(gain, [0.98, 0.93, 0.71, 0.44, 0.21], decimal=1)

    def test_lowpass_iir_first_order_quantized(self):
        signal = np.sum([np.sin(2 * np.pi * f0 * self.time) for f0 in self.freq], axis=0)
        sets = deepcopy(test_settings)
        sets.type = "iir"
        sets.n_order = 1
        sets.b_type = "lowpass"
        sets.f_filt = [50.0]
        result = Filtering(sets).filt_quantized(
            signal, total_bitwidth=10, fraction_width=4, is_signed=True
        )

        freq0, peak0, pos = extract_peaks(signal, sets.fs)
        freq1, peak1 = do_fft(result, sets.fs)
        assert freq0.tolist() == freq1[pos].tolist()
        gain = np.array(peak1[pos]) / np.array(peak0)
        np.testing.assert_almost_equal(gain, [0.98, 0.93, 0.71, 0.44, 0.21], decimal=1)

    def test_lowpass_iir_second_order(self):
        signal = np.sum([np.sin(2 * np.pi * f0 * self.time) for f0 in self.freq], axis=0)
        sets = deepcopy(test_settings)
        sets.type = "iir"
        sets.n_order = 2
        sets.b_type = "lowpass"
        sets.f_filt = [50.0]
        result = Filtering(sets).filt(signal)

        freq0, peak0, pos = extract_peaks(signal, sets.fs)
        freq1, peak1 = do_fft(result, sets.fs)
        assert freq0.tolist() == freq1[pos].tolist()
        gain = np.array(peak1[pos]) / np.array(peak0)
        np.testing.assert_almost_equal(gain, [0.99, 0.98, 0.71, 0.23, 0.05], decimal=1)

    def test_highpass_iir_first_order(self):
        signal = np.sum([np.sin(2 * np.pi * f0 * self.time) for f0 in self.freq], axis=0)
        sets = deepcopy(test_settings)
        sets.type = "iir"
        sets.n_order = 1
        sets.b_type = "highpass"
        sets.f_filt = [50.0]
        result = Filtering(sets).filt(signal)

        freq0, peak0, pos = extract_peaks(signal, sets.fs)
        freq1, peak1 = do_fft(result, sets.fs)
        assert freq0.tolist() == freq1[pos].tolist()
        gain = np.array(peak1[pos]) / np.array(peak0)
        np.testing.assert_almost_equal(gain, [0.19, 0.36, 0.71, 0.9, 0.98], decimal=1)

    def test_bandpass_iir_first_order(self):
        signal = np.sum([np.sin(2 * np.pi * f0 * self.time) for f0 in self.freq], axis=0)
        sets = deepcopy(test_settings)
        sets.type = "iir"
        sets.n_order = 1
        sets.b_type = "bandpass"
        sets.f_filt = [50.0, 100.0]
        result = Filtering(sets).filt(signal)

        freq0, peak0, pos = extract_peaks(signal, sets.fs)
        freq1, peak1 = do_fft(result, sets.fs)
        assert freq0.tolist() == freq1[pos].tolist()
        gain = np.array(peak1[pos]) / np.array(peak0)
        np.testing.assert_almost_equal(gain, [0.1, 0.21, 0.71, 0.71, 0.25], decimal=1)

    def test_bandstop_iir_first_order(self):
        signal = np.sum([np.sin(2 * np.pi * f0 * self.time) for f0 in self.freq], axis=0)
        sets = deepcopy(test_settings)
        sets.type = "iir"
        sets.n_order = 1
        sets.b_type = "bandstop"
        sets.f_filt = [50.0, 100.0]
        result = Filtering(sets).filt(signal)

        freq0, peak0, pos = extract_peaks(signal, sets.fs)
        freq1, peak1 = do_fft(result, sets.fs)
        assert freq0.tolist() == freq1[pos].tolist()
        gain = np.array(peak1[pos]) / np.array(peak0)
        np.testing.assert_almost_equal(gain, [0.99, 0.98, 0.71, 0.71, 0.97], decimal=1)

    def test_notch_iir_first_order(self):
        signal = np.sum([np.sin(2 * np.pi * f0 * self.time) for f0 in self.freq], axis=0)
        sets = deepcopy(test_settings)
        sets.type = "iir"
        sets.n_order = 1
        sets.b_type = "notch"
        sets.f_filt = [50.0]
        result = Filtering(sets).filt(signal)

        freq0, peak0, pos = extract_peaks(signal, sets.fs)
        freq1, peak1 = do_fft(result, sets.fs)
        assert freq0.tolist() == freq1[pos].tolist()
        gain = np.array(peak1[pos]) / np.array(peak0)
        np.testing.assert_almost_equal(gain, [0.98, 0.9, 0.001, 0.84, 0.97], decimal=1)

    def test_allpass_iir_first_order(self):
        signal = np.sum([np.sin(2 * np.pi * f0 * self.time) for f0 in self.freq], axis=0)
        sets = deepcopy(test_settings)
        sets.type = "iir"
        sets.n_order = 1
        sets.b_type = "allpass"
        sets.f_filt = [50.0]
        result = Filtering(sets).filt(signal)

        freq0, peak0, pos = extract_peaks(signal, sets.fs)
        freq1, peak1 = do_fft(result, sets.fs)
        assert freq0.tolist() == freq1[pos].tolist()
        gain = np.array(peak1[pos]) / np.array(peak0)
        np.testing.assert_almost_equal(gain, [0.99, 0.99, 0.99, 0.99, 0.99], decimal=1)

    def test_lowpass_fir_taps21(self):
        signal = np.sum([np.sin(2 * np.pi * f0 * self.time) for f0 in self.freq], axis=0)
        sets = deepcopy(test_settings)
        sets.type = "fir"
        sets.n_order = 21
        sets.b_type = "lowpass"
        sets.f_filt = [50.0]
        result = Filtering(sets).filt(signal)

        freq0, peak0, pos = extract_peaks(signal, sets.fs)
        freq1, peak1 = do_fft(result, sets.fs)
        assert freq0.tolist() == freq1[pos].tolist()
        gain = np.array(peak1[pos]) / np.array(peak0)
        np.testing.assert_almost_equal(gain, [0.98, 0.92, 0.59, 0.09, 0.001], decimal=2)

    def test_lowpass_fir_taps51(self):
        signal = np.sum([np.sin(2 * np.pi * f0 * self.time) for f0 in self.freq], axis=0)
        sets = deepcopy(test_settings)
        sets.type = "fir"
        sets.n_order = 51
        sets.b_type = "lowpass"
        sets.f_filt = [50.0]
        result = Filtering(sets).filt(signal)

        freq0, peak0, pos = extract_peaks(signal, sets.fs)
        freq1, peak1 = do_fft(result, sets.fs)
        assert freq0.tolist() == freq1[pos].tolist()
        gain = np.array(peak1[pos]) / np.array(peak0)
        np.testing.assert_almost_equal(gain, [0.99, 0.99, 0.5, 0.001, 0.001], decimal=2)

    def test_highpass_fir_taps51(self):
        signal = np.sum([np.sin(2 * np.pi * f0 * self.time) for f0 in self.freq], axis=0)
        sets = deepcopy(test_settings)
        sets.type = "fir"
        sets.n_order = 51
        sets.b_type = "highpass"
        sets.f_filt = [20.0]
        result = Filtering(sets).filt(signal)

        freq0, peak0, pos = extract_peaks(signal, sets.fs)
        freq1, peak1 = do_fft(result, sets.fs)
        assert freq0.tolist() == freq1[pos].tolist()
        gain = np.array(peak1[pos]) / np.array(peak0)
        np.testing.assert_almost_equal(gain, [0.26, 0.5, 0.99, 0.99, 0.99], decimal=2)

    def test_bandpass_fir_taps51(self):
        signal = np.sum([np.sin(2 * np.pi * f0 * self.time) for f0 in self.freq], axis=0)
        sets = deepcopy(test_settings)
        sets.type = "fir"
        sets.n_order = 51
        sets.b_type = "bandpass"
        sets.f_filt = [20.0, 100.0]
        result = Filtering(sets).filt(signal)

        freq0, peak0, pos = extract_peaks(signal, sets.fs)
        freq1, peak1 = do_fft(result, sets.fs)
        assert freq0.tolist() == freq1[pos].tolist()
        gain = np.array(peak1[pos]) / np.array(peak0)
        np.testing.assert_almost_equal(gain, [0.25, 0.49, 0.99, 0.49, 0.001], decimal=2)

    def test_bandstop_fir_taps51(self):
        signal = np.sum([np.sin(2 * np.pi * f0 * self.time) for f0 in self.freq], axis=0)
        sets = deepcopy(test_settings)
        sets.type = "fir"
        sets.n_order = 51
        sets.b_type = "bandstop"
        sets.f_filt = [20.0, 100.0]
        result = Filtering(sets).filt(signal)

        freq0, peak0, pos = extract_peaks(signal, sets.fs)
        freq1, peak1 = do_fft(result, sets.fs)
        assert freq0.tolist() == freq1[pos].tolist()
        gain = np.array(peak1[pos]) / np.array(peak0)
        np.testing.assert_almost_equal(gain, [0.88, 0.59, 0.01, 0.59, 1.18], decimal=2)

    def test_notch_fir_taps51(self):
        signal = np.sum([np.sin(2 * np.pi * f0 * self.time) for f0 in self.freq], axis=0)
        sets = deepcopy(test_settings)
        sets.type = "fir"
        sets.n_order = 501
        sets.b_type = "notch"
        sets.f_filt = [50.0, 1.0]
        result = Filtering(sets).filt(signal)

        freq0, peak0, pos = extract_peaks(signal, sets.fs)
        freq1, peak1 = do_fft(result, sets.fs)
        assert freq0.tolist() == freq1[pos].tolist()
        gain = np.array(peak1[pos]) / np.array(peak0)
        np.testing.assert_almost_equal(gain, [0.99, 0.99, 0.73, 0.99, 0.99], decimal=2)

    def test_allpass_fir_taps51(self):
        signal = np.sum([np.sin(2 * np.pi * f0 * self.time) for f0 in self.freq], axis=0)
        sets = deepcopy(test_settings)
        sets.type = "fir"
        sets.n_order = 201
        sets.b_type = "allpass"
        sets.f_filt = [0.0]
        result = Filtering(sets).filt(signal)
        np.testing.assert_almost_equal(signal[: -sets.n_order], result[sets.n_order :], decimal=5)

    def test_compare_normal_vs_quantized_fir_lowpass(self):
        sets = deepcopy(test_settings)
        sets.type = "fir"
        sets.n_order = 51
        sets.b_type = "lowpass"
        sets.f_filt = [80.0]
        sets.fs = 200.0
        bitwidth = 10

        signal = np.sin(2 * np.pi * self.freq[0] * self.time)
        dut = Filtering(sets)

        out_norm = dut.filt(xin=signal)
        out_quant = dut.filt_quantized(
            xin=signal,
            total_bitwidth=bitwidth,
            fraction_width=bitwidth - 2,
            is_signed=True,
        )

        error = out_norm - out_quant
        assert abs(error.min()) < 2 ** -(bitwidth - 2)
        assert float(error.max()) <= 2 * 2 ** -(bitwidth - 2)

    def test_compare_normal_vs_quantized_fir_highpass(self):
        sets = deepcopy(test_settings)
        sets.type = "fir"
        sets.n_order = 51
        sets.b_type = "highpass"
        sets.f_filt = [80.0]
        sets.fs = 200.0
        bitwidth = 10

        signal = np.sin(2 * np.pi * self.freq[0] * self.time)
        dut = Filtering(sets)

        out_norm = dut.filt(xin=signal)
        out_quant = dut.filt_quantized(
            xin=signal,
            total_bitwidth=bitwidth,
            fraction_width=bitwidth - 2,
            is_signed=True,
        )

        error = out_norm - out_quant
        assert abs(error.min()) < 2 ** -(bitwidth - 2)
        assert float(error.max()) <= 2 * 2 ** -(bitwidth - 2)

    def test_compare_normal_vs_quantized_iir_lowpass(self):
        sets = deepcopy(test_settings)
        sets.type = "iir"
        sets.n_order = 2
        sets.b_type = "lowpass"
        sets.f_filt = [80.0]
        sets.fs = 200.0
        bitwidth = 10

        signal = np.sin(2 * np.pi * self.freq[0] * self.time)
        dut = Filtering(sets)

        out_norm = dut.filt(xin=signal)
        out_quant = dut.filt_quantized(
            xin=signal,
            total_bitwidth=bitwidth,
            fraction_width=bitwidth - 2,
            is_signed=True,
        )

        error = out_norm - out_quant
        assert abs(error.min()) < 4 * 2 ** -(bitwidth - 2)
        assert error.max() <= 6 * 2 ** -(bitwidth - 2)

    def test_compare_normal_vs_quantized_iir_highpass(self):
        sets = deepcopy(test_settings)
        sets.type = "iir"
        sets.n_order = 2
        sets.b_type = "highpass"
        sets.f_filt = [80.0]
        sets.fs = 200.0
        bitwidth = 10

        signal = np.sin(2 * np.pi * self.freq[0] * self.time)
        dut = Filtering(sets)

        out_norm = dut.filt(xin=signal)
        out_quant = dut.filt_quantized(
            xin=signal,
            total_bitwidth=bitwidth,
            fraction_width=bitwidth - 2,
            is_signed=True,
        )

        error = out_norm - out_quant
        assert abs(error.min()) < 4 * 2 ** -(bitwidth - 2)
        assert error.max() <= 4 * 2 ** -(bitwidth - 2)

    def test_coeffs_fir_lowpass(self):
        sets = deepcopy(test_settings)
        sets.type = "fir"
        sets.n_order = 11
        result = Filtering(sets).get_coeffs()

        assert result.a == 1.0
        assert len(result.b) == 11
        assert result.b == [
            0.0050603171248448505,
            -3.2506154907344878e-18,
            -0.04194287943134475,
            1.3210434491046798e-17,
            0.2884848263026376,
            0.4967954720077247,
            0.2884848263026376,
            1.3210434491046798e-17,
            -0.04194287943134475,
            -3.2506154907344878e-18,
            0.0050603171248448505,
        ]

    def test_coeffs_iir_lowpass(self):
        sets = deepcopy(test_settings)
        sets.type = "iir"
        sets.n_order = 2
        result = Filtering(sets).get_coeffs()

        check_a = [1.0, -1.8403695052228406e-16, 0.17157287525380993]
        check_b = [0.2928932188134525, 0.585786437626905, 0.2928932188134525]

        np.testing.assert_almost_equal(result.a, check_a, decimal=6)
        assert len(result.a) == 3
        np.testing.assert_almost_equal(result.b, check_b, decimal=6)
        assert len(result.b) == 3

    def test_coeffs_quantized_fir_8bit(self):
        sets = deepcopy(test_settings)
        sets.type = "fir"
        sets.n_order = 11
        result = Filtering(sets).get_coeffs_quantized(8)

        assert len(result[0].a) == 1
        assert result[0].a == [1.0]
        assert result[1].a == [0.0]
        assert len(result[0].b) == sets.n_order
        assert result[0].b == [
            0.0,
            0.0,
            -0.0390625,
            0.0,
            0.28125,
            0.4921875,
            0.28125,
            0.0,
            -0.0390625,
            0.0,
            0.0,
        ]
        assert result[1].b == [
            0.0050603171248448505,
            -3.2506154907344878e-18,
            -0.002880379431344747,
            1.3210434491046798e-17,
            0.007234826302637609,
            0.0046079720077247255,
            0.007234826302637609,
            1.3210434491046798e-17,
            -0.002880379431344747,
            -3.2506154907344878e-18,
            0.0050603171248448505,
        ]

    def test_coeffs_quantized_iir_8bit(self):
        sets = deepcopy(test_settings)
        sets.type = "iir"
        sets.n_order = 2
        result = Filtering(sets).get_coeffs_quantized(8)

        assert len(result[0].a) == 3
        np.testing.assert_almost_equal(result[0].a, [1.0, 0.0, 0.15625], decimal=6)
        np.testing.assert_almost_equal(
            result[1].a, [0.0, -1.8403695052228406e-16, 0.015322875253809931], decimal=6
        )
        assert len(result[0].b) == 3
        np.testing.assert_almost_equal(result[0].b, [0.28125, 0.578125, 0.28125], decimal=6)
        np.testing.assert_almost_equal(
            result[1].b,
            [
                0.011643218813452483,
                0.0076614376269049655,
                0.011643218813452483,
            ],
            decimal=6,
        )

    def test_verilog_string_iir(self):
        sets = deepcopy(test_settings)
        sets.type = "iir"
        sets.n_order = 2
        result = Filtering(sets).get_coeffs_verilog_string(8, True)
        assert result == "{8'h12, 8'h25, 8'h12, 8'h00, 8'hF6}"

    def test_verilog_string_fir_8bit_half(self):
        sets = deepcopy(test_settings)
        sets.type = "fir"
        sets.n_order = 11
        result = Filtering(sets).get_coeffs_verilog_string(8, True)
        assert result == "{8'h00, 8'h00, 8'hFB, 8'h00, 8'h24, 8'h3F}"

    def test_verilog_string_fir_8bit_full(self):
        sets = deepcopy(test_settings)
        sets.type = "fir"
        sets.n_order = 11
        result = Filtering(sets).get_coeffs_verilog_string(8, False)
        assert result == "{8'h00, 8'h00, 8'hFB, 8'h00, 8'h24, 8'h3F, 8'h24, 8'h00, 8'hFB, 8'h00, 8'h00}"

    def test_create_verilog_filter_biquad(self):
        sets = deepcopy(test_settings)
        sets.type = "iir"
        sets.b_type = "lowpass"
        sets.n_order = 2

        path2save = get_path_to_project("build_files") / "design_iir"
        path2save.mkdir(parents=True, exist_ok=True)

        Filtering(sets).create_design("fpga", 8, "0", path2save)
        files_available = ["biquad_df1_0.v", "mac.v", "mult_dsp_signed.v"]
        for file in path2save.glob("*.v"):
            assert file.exists()
            assert file.name in files_available

    def test_create_verilog_filter_fir_full(self):
        sets = deepcopy(test_settings)
        sets.type = "fir"
        sets.b_type = "lowpass"
        sets.n_order = 20

        path2save = get_path_to_project("build_files") / "design_fir_full"
        path2save.mkdir(parents=True, exist_ok=True)

        Filtering(sets).create_design("fpga", 8, "0", path2save)
        files_available = [
            "fir_full_0.v",
            "mac.v",
            "mult_dsp_signed.v",
            "ring_buffer.v",
        ]
        for file in path2save.glob("*.v"):
            assert file.exists()
            assert file.name in files_available

    def test_create_verilog_filter_fir_half(self):
        sets = deepcopy(test_settings)
        sets.type = "fir"
        sets.b_type = "lowpass"
        sets.n_order = 21

        path2save = get_path_to_project("build_files") / "design_fir_half"
        path2save.mkdir(parents=True, exist_ok=True)

        Filtering(sets).create_design("fpga", 8, "0", path2save)
        files_available = [
            "fir_half_0.v",
            "mac.v",
            "mult_dsp_signed.v",
            "ring_buffer.v",
        ]
        for file in path2save.glob("*.v"):
            assert file.exists()
            assert file.name in files_available

    def test_create_verilog_filter_fir_delay(self):
        sets = deepcopy(test_settings)
        sets.type = "fir"
        sets.b_type = "allpass"
        sets.n_order = 12

        path2save = get_path_to_project("build_files") / "design_fir_delay"
        path2save.mkdir(parents=True, exist_ok=True)

        Filtering(sets).create_design("fpga", 8, "0", path2save)
        files_available = ["fir_delay_0.v", "ring_buffer.v"]
        for file in path2save.glob("*.v"):
            assert file.exists()
            assert file.name in files_available

    def test_create_verilog_filter_fir_simple_lowpass(self):
        sets = deepcopy(test_settings)
        sets.type = "fir"
        sets.b_type = "simple_low"
        sets.n_order = 1

        path2save = get_path_to_project("build_files") / "design_fir_low"
        path2save.mkdir(parents=True, exist_ok=True)

        Filtering(sets).create_design("fpga", 8, "0", path2save)
        files_available = ["fir_low_0.v"]
        for file in path2save.glob("*.v"):
            assert file.exists()
            assert file.name in files_available

    def test_create_c_filter_fir_full(self):
        sets = deepcopy(test_settings)
        sets.type = "fir"
        sets.b_type = "lowpass"
        sets.n_order = 20

        with TemporaryDirectory() as directory:
            path2save = Path(directory)
            Filtering(sets).create_design("mcu", 8, "0", path2save, signed=False)

            assert {file.name for file in path2save.iterdir()} == {
                "filter_fir_template.h",
                "filter_fir_low0.h",
                "filter_fir_low0.c",
            }
            assert "uint8_t" in (path2save / "filter_fir_low0.h").read_text()
            assert "template (full)" in (path2save / "filter_fir_low0.c").read_text()

    def test_create_c_filter_fir_optimized(self):
        sets = deepcopy(test_settings)
        sets.type = "fir"
        sets.b_type = "lowpass"
        sets.n_order = 21

        with TemporaryDirectory() as directory:
            path2save = Path(directory)
            Filtering(sets).create_design("mcu", 8, "0", path2save)

            assert "template (opt)" in (path2save / "filter_fir_low0.c").read_text()

    def test_create_c_filter_fir_allpass(self):
        sets = deepcopy(test_settings)
        sets.type = "fir"
        sets.b_type = "allpass"
        sets.n_order = 12
        sets.f_filt = [50.0]

        with TemporaryDirectory() as directory:
            path2save = Path(directory)
            Filtering(sets).create_design("mcu", 8, "0", path2save)

            assert {file.name for file in path2save.iterdir()} == {
                "filter_fir_all_template.h",
                "filter_fir_all0.h",
                "filter_fir_all0.c",
            }

    def test_create_c_filter_iir(self):
        sets = deepcopy(test_settings)
        sets.type = "iir"
        sets.b_type = "lowpass"
        sets.n_order = 2

        with TemporaryDirectory() as directory:
            path2save = Path(directory)
            Filtering(sets).create_design("pc", 16, "0", path2save)

            assert {file.name for file in path2save.iterdir()} == {
                "filter_iir_template.h",
                "filter_iir_low0.h",
                "filter_iir_low0.c",
            }

    @skipUnless(which("cc"), "requires a C compiler")
    def test_create_c_filter_fir_compiles(self):
        sets = deepcopy(test_settings)
        sets.type = "fir"
        sets.b_type = "lowpass"
        sets.n_order = 21

        with TemporaryDirectory() as directory:
            project_dir = Path(directory)
            path2save = project_dir / "src"
            Filtering(sets).create_design("mcu", 8, "0", path2save)

            self.assert_c_compiles(project_dir, path2save / "filter_fir_low0.c")

    @skipUnless(which("cc"), "requires a C compiler")
    def test_create_c_filter_fir_allpass_compiles(self):
        sets = deepcopy(test_settings)
        sets.type = "fir"
        sets.b_type = "allpass"
        sets.n_order = 12
        sets.f_filt = [50.0]

        with TemporaryDirectory() as directory:
            project_dir = Path(directory)
            path2save = project_dir / "src"
            Filtering(sets).create_design("mcu", 8, "0", path2save)

            self.assert_c_compiles(project_dir, path2save / "filter_fir_all0.c")


if __name__ == "__main__":
    main()
