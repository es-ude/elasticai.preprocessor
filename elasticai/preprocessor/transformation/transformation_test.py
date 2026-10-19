from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from elasticai.preprocessor import DataSpectral, DataTransient

from .transformation import (
    Transformation,
    TransformationSettingsDefault,
    TransformationTargets,
)
from .window_scaler import WindowingTargets

FS = 1000.0
N = 1000
F_SIG = 100.0


def make_transformation(
    method: TransformationTargets | str = TransformationTargets.FFT,
) -> Transformation:
    sets = deepcopy(TransformationSettingsDefault)
    sets.method = method
    sets.sampling_rate = FS
    return Transformation(settings=sets)


def make_sine(fs: float = FS, n: int = N, freq: float = F_SIG) -> np.ndarray:
    return np.sin(2 * np.pi * freq * np.arange(n) / fs)


@pytest.mark.parametrize("method", ["fft", "FFT", "fft_imag", "FFT_IMAG"])
def test_method_string_is_converted_to_enum(method: str) -> None:
    assert make_transformation(method)._settings.method == TransformationTargets(method.lower())


@pytest.mark.parametrize("method", ["dft", "wavelet", ""])
def test_wrong_method_string(method: str) -> None:
    with pytest.raises(ValueError):
        make_transformation(method)


@pytest.mark.parametrize(
    "window", [WindowingTargets.Hamming.value, WindowingTargets.Hamming.value.upper()]
)
def test_window_string_is_converted_to_enum(window: str) -> None:
    sets = deepcopy(TransformationSettingsDefault)
    sets.method_window = window
    assert Transformation(settings=sets)._settings.method_window == WindowingTargets.Hamming


def test_forward_unsupported_method() -> None:
    transformation = make_transformation()
    transformation._settings.method = "unknown"
    with pytest.raises(AttributeError):
        transformation.forward(make_sine())


def test_fft_output_shape_and_freq() -> None:
    out = make_transformation(TransformationTargets.FFT)(
        DataTransient(data=make_sine(), sampling_rate=FS)
    )
    assert isinstance(out, DataSpectral)
    assert out.data.shape == out.freq.shape == (N // 2,)
    assert out.freq[0] == 0.0
    np.testing.assert_allclose(out.freq, FS * np.fft.fftfreq(N)[: N // 2])
    assert np.all(out.data >= 0)


def test_fft_imag_output_shape_and_freq() -> None:
    out = make_transformation(TransformationTargets.imagFFT)(
        DataTransient(data=make_sine(), sampling_rate=FS)
    )
    assert out.data.shape == out.freq.shape == (N // 2 + 1,)
    assert np.iscomplexobj(out.data)
    np.testing.assert_allclose(out.freq, np.fft.rfftfreq(N, d=1 / FS))


@pytest.mark.parametrize("method", list(TransformationTargets))
def test_peak_at_signal_frequency(method: TransformationTargets) -> None:
    out = make_transformation(method)(DataTransient(data=make_sine(), sampling_rate=FS))
    assert out.freq[np.argmax(np.abs(out.data))] == F_SIG


@pytest.mark.parametrize("method", list(TransformationTargets))
def test_forward_matches_call(method: TransformationTargets) -> None:
    data = make_sine()
    transformation = make_transformation(method)
    np.testing.assert_array_equal(
        transformation.forward(data),
        transformation(DataTransient(data=data, sampling_rate=FS)).data,
    )


@pytest.mark.parametrize("method", list(TransformationTargets))
def test_call_uses_sampling_rate_of_input(method: TransformationTargets) -> None:
    fs_new = 2000.0
    transformation = make_transformation(method)
    out = transformation(DataTransient(data=make_sine(fs=fs_new), sampling_rate=fs_new))
    assert transformation._settings.sampling_rate == fs_new
    assert out.sampling_rate == fs_new
    assert out.freq[np.argmax(np.abs(out.data))] == F_SIG


def test_create_design_not_implemented(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError):
        make_transformation().create_design(id="0", target="mcu", path2save=tmp_path, total_bitwidth=8)
