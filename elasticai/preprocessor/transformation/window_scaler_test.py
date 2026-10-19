import numpy as np
import pytest
from scipy.signal.windows import gaussian

from .window_scaler import WindowingTargets
from .window_scaler import transformation_window_method as twm

REFERENCE = {
    WindowingTargets.Ones: np.ones,
    WindowingTargets.Hamming: np.hamming,
    WindowingTargets.Hanning: np.hanning,
    WindowingTargets.Bartlett: np.bartlett,
    WindowingTargets.Blackman: np.blackman,
    WindowingTargets.Gaussian: lambda n: gaussian(n, int(0.16 * n), sym=True),
}
METHODS = list(WindowingTargets)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("size", [16, 31, 101])  # >= 7, sonst std=0 bei Gaussian
def test_matches_reference(method, size):
    np.testing.assert_allclose(twm(size, method), REFERENCE[method](size))


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("size", [16, 31])
def test_shape_symmetry_range(method, size):
    w = twm(size, method)
    assert w.shape == (size,)
    assert np.all(np.isfinite(w))
    np.testing.assert_allclose(w, w[::-1], atol=1e-12)
    assert -1e-12 <= w.min() and w.max() <= 1 + 1e-12


@pytest.mark.parametrize("method", METHODS)
def test_size_one(method):
    np.testing.assert_allclose(twm(1, method), [1.0])


@pytest.mark.parametrize(
    "method", [WindowingTargets.Hanning, WindowingTargets.Bartlett, WindowingTargets.Blackman]
)
def test_tapering_windows_start_and_end_at_zero(method):
    w = twm(32, method)
    assert w[0] == pytest.approx(0, abs=1e-12)
    assert w[-1] == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize("method", [m for m in METHODS if m != WindowingTargets.Ones])
def test_peak_in_middle_for_odd_size(method):
    w = twm(31, method)
    assert np.argmax(w) == 15
    assert w[15] == pytest.approx(1.0)


def test_default_is_hamming():
    np.testing.assert_allclose(twm(16), np.hamming(16))


@pytest.mark.parametrize("method", METHODS)
def test_string_input_case_insensitive(method):
    expected = twm(20, method)
    np.testing.assert_allclose(twm(20, method.value), expected)
    np.testing.assert_allclose(twm(20, method.value.upper()), expected)


@pytest.mark.parametrize("name", ["unknown", "", "hann"])
def test_invalid_string_raises(name):
    with pytest.raises(ValueError):
        twm(16, name)
