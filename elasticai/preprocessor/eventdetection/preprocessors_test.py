from copy import deepcopy

import numpy as np
import pytest

from elasticai.preprocessor import get_path_to_project
from elasticai.preprocessor.translation.cocotb_tmp import temporary_directory
from .preprocessors import (
    DefaultSettingsEventPreprocessor, 
    EventPreprocessor, 
    SettingsEventPreprocessor,
    TargetsEventPreprocessors,
)

PREPROCESSOR_TYPE_CONFIGS = {
    pytest.param(TargetsEventPreprocessors.Normal, "preprocessing_normal", id="type_normal"),
    pytest.param(TargetsEventPreprocessors.Absolute, "preprocessing_abs", id="type_absolute"),
    pytest.param(TargetsEventPreprocessors.NEO, "preprocessing_neo", id="type_neo"),
}
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


def test_sda_sbp_none(settings: SettingsEventPreprocessor, signal: np.ndarray):
    settings.type = "sbp"
    try:
        EventPreprocessor(settings=settings).get_preprocessed(xraw=signal)
    except:
        assert True
    else:
        assert False


def test_sda_sbp(settings: SettingsEventPreprocessor, signal: np.ndarray):
    settings.type = "sbp"
    settings.f_filt = [100.0, 1000.0]
    rslt = EventPreprocessor(settings=settings).get_preprocessed(xraw=signal)
    check = np.abs(rslt)

    assert rslt.size == signal.size
    np.testing.assert_array_equal(rslt, check)
    assert rslt.min() >= 0.0
    assert rslt.max() < 1.11

class TestCreateDesign:
    @pytest.mark.parametrize("target", ["mcu", "pc"])
    @pytest.mark.parametrize("method,c_name", PREPROCESSOR_TYPE_CONFIGS)
    def test_create_design_generates_sda_preprocessor_c_files(
        self,
        target: str,
        method: TargetsEventPreprocessors,
        c_name: str
    ) -> None:
        event_preproc = EventPreprocessor(
            SettingsEventPreprocessor(
                type=method,
                sampling_rate=10e3,
                window_size=[1],
                f_filt=[150.0],
            )
        )
        
        backup = get_path_to_project("build_test") / f"{c_name}"
        with temporary_directory(backup) as tmpdir:
            event_preproc.create_design(
                target=target,
                bitwidth=8,
                id="0",
                path2save=tmpdir,
                signed=True,
            )
            assert(tmpdir / f"{c_name}_0.c").exists()
            assert(tmpdir / f"{c_name}_0.h").exists()
            assert(tmpdir / f"{c_name}_template.h").exists()