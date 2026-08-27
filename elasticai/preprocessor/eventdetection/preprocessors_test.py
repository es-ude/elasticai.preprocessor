from copy import deepcopy

import numpy as np
import pytest

from .preprocessors import DefaultSettingsEventPreprocessor, EventPreprocessor, SettingsEventPreprocessor


@pytest.fixture()
def settings() -> SettingsEventPreprocessor:
    sets = deepcopy(DefaultSettingsEventPreprocessor)
    return sets


@pytest.fixture()
def signal() -> np.ndarray:
    return np.sin(np.linspace(start=0, stop=4 * np.pi, num=51))


def test_methods_overview(settings: SettingsEventPreprocessor):
    rslt = EventPreprocessor(settings=settings)._get_methods()
    assert len(rslt) == 8

    for method in ["normal", "absolute", "neo"]:
        assert method in rslt


def test_methods_none(settings: SettingsEventPreprocessor):
    settings.type = "none"
    try:
        EventPreprocessor(settings=settings)
    except:
        assert True
    else:
        assert False


def test_sda_normal(settings: SettingsEventPreprocessor, signal: np.ndarray):
    settings.type = "normal"
    rslt = EventPreprocessor(settings=settings).get_preprocessed(xraw=signal)
    check = signal

    assert rslt.size == signal.size
    np.testing.assert_array_equal(rslt, check)


def test_sda_absolute(settings: SettingsEventPreprocessor, signal: np.ndarray):
    settings.type = "absolute"
    rslt = EventPreprocessor(settings=settings).get_preprocessed(xraw=signal)
    check = np.abs(signal)

    assert rslt.size == signal.size
    np.testing.assert_array_equal(rslt, check)


def test_sda_neo_ones(settings: SettingsEventPreprocessor, signal: np.ndarray):
    settings.type = "neo"
    settings.window_size = [1]
    rslt = EventPreprocessor(settings=settings).get_preprocessed(xraw=signal)
    check = np.zeros_like(signal) + 0.06184665997806704

    assert rslt.size == signal.size
    np.testing.assert_array_almost_equal(rslt, check, decimal=6)


def test_sda_neo_two(settings: SettingsEventPreprocessor, signal: np.ndarray):
    settings.type = "neo"
    settings.window_size = [2]
    rslt = EventPreprocessor(settings=settings).get_preprocessed(xraw=signal)
    check = np.zeros_like(signal) + 0.23208660251050067

    assert rslt.size == signal.size
    np.testing.assert_array_almost_equal(rslt, check, decimal=6)


def test_sda_mteo_two(settings: SettingsEventPreprocessor, signal: np.ndarray):
    settings.type = "mteo"
    settings.window_size = [1, 2, 3]
    rslt = EventPreprocessor(settings=settings).get_preprocessed(xraw=signal)
    check = np.zeros_like(signal) + 0.4686047402353426

    assert rslt.size == signal.size
    np.testing.assert_array_almost_equal(rslt, check, decimal=6)


def test_sda_ado_ones(settings: SettingsEventPreprocessor, signal: np.ndarray):
    settings.type = "ado"
    settings.window_size = [1]
    rslt = EventPreprocessor(settings=settings).get_preprocessed(xraw=signal)
    check = np.abs(rslt)

    assert rslt.size == signal.size
    np.testing.assert_array_equal(rslt, check)
    assert rslt.min() > 0.0
    assert rslt.max() < 0.255


def test_sda_ado_threes(settings: SettingsEventPreprocessor, signal: np.ndarray):
    settings.type = "ado"
    settings.window_size = [3]
    rslt = EventPreprocessor(settings=settings).get_preprocessed(xraw=signal)
    check = np.abs(rslt)

    assert rslt.size == signal.size
    np.testing.assert_array_equal(rslt, check)
    assert rslt.min() > 0.0
    assert rslt.max() < 0.75


def test_sda_eed(settings: SettingsEventPreprocessor, signal: np.ndarray):
    settings.type = "eed"
    settings.f_filt = [150.0]
    rslt = EventPreprocessor(settings=settings).get_preprocessed(xraw=signal)
    check = np.abs(rslt)

    assert rslt.size == signal.size
    np.testing.assert_array_equal(rslt, check)
    assert rslt.min() >= 0.0
    assert rslt.max() < 0.92


def test_sda_spb_none(settings: SettingsEventPreprocessor, signal: np.ndarray):
    settings.type = "spb"
    try:
        EventPreprocessor(settings=settings).get_preprocessed(xraw=signal)
    except:
        assert True
    else:
        assert False


def test_sda_spb(settings: SettingsEventPreprocessor, signal: np.ndarray):
    settings.type = "spb"
    settings.f_filt = [100.0, 1000.0]
    rslt = EventPreprocessor(settings=settings).get_preprocessed(xraw=signal)
    check = np.abs(rslt)

    assert rslt.size == signal.size
    np.testing.assert_array_equal(rslt, check)
    assert rslt.min() >= 0.0
    assert rslt.max() < 1.11
