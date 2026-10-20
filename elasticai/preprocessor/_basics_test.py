import numpy as np
import pytest

from ._basics import DataWaveform, get_path_to_project


def test_path_to_project() -> None:
    checks = ["elasticai", "preprocessor"]
    rslt = get_path_to_project()

    assert rslt.is_dir()
    assert rslt.parts[-1] == f"{checks[0]}.{checks[1]}"


def test_path_to_project_ref() -> None:
    checks = ["elasticai", "preprocessor", "test"]
    rslt = get_path_to_project(new_folder=checks[2])

    assert not rslt.exists()
    assert rslt.parts[-1] == f"{checks[2]}"
    assert rslt.parts[-2] == f"{checks[0]}.{checks[1]}"


@pytest.mark.parametrize(
    "waveform, xpos, label, length, num_samples, is_labeled",
    [
        (np.zeros((3, 10)), np.arange(3), np.array([0, 1, 2]), 10, 3, True),
        (np.zeros((1, 5)), np.array([7]), np.array([0]), 5, 1, True),
        (np.zeros((2, 4)), np.arange(2), np.array([0, 255]), 4, 2, False),
        (np.zeros((2, 4)), np.arange(2), np.array([255, 255]), 4, 2, False),
        (np.zeros((0, 6)), np.array([]), np.array([]), 6, 0, False),
    ],
    ids=["labeled", "single", "partly_unlabeled", "all_unlabeled", "empty"],
)
def test_data_waveform_properties(waveform, xpos, label, length, num_samples, is_labeled):
    data = DataWaveform(data=waveform, xpos=xpos, label=label, sampling_rate=1000.0)

    assert data.length == length
    assert data.num_samples == num_samples
    assert data.is_data_labeled is is_labeled
