from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from elasticai.preprocessor import get_path_to_project
from elasticai.preprocessor._basics import DataTransient, PreprocessingModule
from elasticai.preprocessor.sequential import (
    BuildPlatformTargets,
    CreateSequentialSettings,
    PreprocessingSequential,
)
from elasticai.preprocessor.translation.cocotb_tmp import temporary_directory


@dataclass
class SettingsAmplifier:
    gain: float


class Amplifier(PreprocessingModule[DataTransient, DataTransient]):
    _settings: SettingsAmplifier

    def __init__(self, settings: SettingsAmplifier) -> None:
        super().__init__()
        self._settings = settings

    def __call__(self, x: DataTransient) -> DataTransient:
        return DataTransient(
            data=self.forward(x=x.data),
            sampling_rate=x.sampling_rate,
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self._settings.gain * x

    def create_design(self, id: str, **settings: CreateSequentialSettings):
        with open(settings["path2save"] / f"amplifier_{id}.txt", "w") as f:
            f.write("Hallo Welt!\n")


@dataclass
class SettingsAdderOffset:
    offset: float


class Offset(PreprocessingModule[DataTransient, DataTransient]):
    _settings: SettingsAdderOffset

    def __init__(self, settings: SettingsAdderOffset) -> None:
        super().__init__()
        self._settings = settings

    def __call__(self, x: DataTransient) -> DataTransient:
        return DataTransient(
            data=self.forward(x=x.data),
            sampling_rate=x.sampling_rate,
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self._settings.offset + x

    def create_design(self, id: str, **settings: CreateSequentialSettings):
        with open(settings["path2save"] / f"adder_off_{id}.txt", "w") as f:
            f.write("Hallo Welt!\n")


@pytest.fixture
def data_in() -> np.ndarray:
    return np.random.randn(100)


@pytest.fixture
def fs() -> float:
    return 100.0


def test_sequential_call_empty():
    pipeline = PreprocessingSequential()

    assert len(pipeline) == 0
    assert pipeline.modules == []
    assert repr(pipeline) == "PreprocessingSequential()"


def test_sequential_run_empty(data_in: np.ndarray, fs: float):
    pipeline = PreprocessingSequential()

    with pytest.raises(AttributeError):
        pipeline(data_in, fs)


def test_sequential_create_design_empty():
    pipeline = PreprocessingSequential()

    backup = get_path_to_project() / "build_test" / "seq_empty"
    with temporary_directory(backup) as tmpdir:
        path2build = Path(tmpdir).absolute()
        sets = CreateSequentialSettings(
            target=BuildPlatformTargets.Workstation,
            total_bitwidth=16,
            frac_bitwidth=0,
            do_signed=False,
            path2build=path2build,
        )
        with pytest.raises(AttributeError):
            pipeline.create_design(settings=sets)


@pytest.mark.parametrize(
    "num_stages, check",
    [
        (1, "PreprocessingSequential(\n\tAmplifier\n)"),
        (2, "PreprocessingSequential(\n\tAmplifier,\n\tAmplifier\n)"),
    ],
)
def test_sequential_stages_build(num_stages: int, check: str) -> None:
    modules = [Amplifier(settings=SettingsAmplifier(gain=1.0)) for _ in range(num_stages)]
    pipeline = PreprocessingSequential(*modules)

    assert len(pipeline) == num_stages
    assert all(["Amplifier" in module.__repr__() for module in pipeline.modules])
    assert repr(pipeline) == check


@pytest.mark.parametrize(
    "take_stage, check",
    [
        (0, "Offset"),
        (1, "Amplifier"),
    ],
)
def test_sequential_getitem(take_stage: int, check: str) -> None:
    pipeline = PreprocessingSequential(
        Offset(settings=SettingsAdderOffset(offset=1.0)),
        Amplifier(settings=SettingsAmplifier(gain=1.0)),
    )

    assert len(pipeline) == 2
    assert type(pipeline[take_stage]).__name__ == check


@pytest.mark.parametrize(
    "num_stages, gain",
    [
        (1, [2.0]),
        (2, [-1.0, 2.0]),
    ],
)
def test_sequential_stages_run_and_create(
    data_in: np.ndarray, fs: float, num_stages: int, gain: list[float]
) -> None:
    modules = [Amplifier(settings=SettingsAmplifier(gain=val)) for val in gain]
    pipeline = PreprocessingSequential(*modules)

    assert len(pipeline) == num_stages
    assert all(["Amplifier" in module.__repr__() for module in pipeline.modules])
    data_out = pipeline(data_in, fs)

    total_gain = 1.0
    for g in [module._settings.gain for module in pipeline.modules]:
        total_gain *= g
    data_check = total_gain * data_in
    np.testing.assert_array_equal(data_out.data, data_check)
    np.testing.assert_array_equal(data_out.sampling_rate, fs)

    backup = get_path_to_project() / "build_test" / f"seq_stage_{num_stages}"
    with temporary_directory(backup) as tmpdir:
        path2build = Path(tmpdir).absolute()
        sets1 = CreateSequentialSettings(
            target=BuildPlatformTargets.Workstation,
            total_bitwidth=16,
            frac_bitwidth=0,
            do_signed=False,
            path2build=path2build,
        )
        pipeline.create_design(settings=sets1)

        for idx, _ in enumerate(modules):
            assert (path2build / "src" / f"amplifier_{idx}.txt").exists()


def test_sequential_stages_mixed_run_and_create(data_in: np.ndarray, fs: float) -> None:
    pipeline = PreprocessingSequential(
        Offset(settings=SettingsAdderOffset(offset=1.0)),
        Amplifier(settings=SettingsAmplifier(gain=1.0)),
    )
    assert len(pipeline) == 2
    assert all(["Amplifier" or "Offset" in module.__repr__() for module in pipeline.modules])
    data_check = data_in + 1
    np.testing.assert_array_equal(pipeline(data_in, fs).data, data_check)

    pipeline.append(Amplifier(settings=SettingsAmplifier(gain=-2.0)))
    assert len(pipeline) == 3
    data_check = -2 * (data_in + 1)
    np.testing.assert_array_equal(pipeline(data_in, fs).data, data_check)

    pipeline.append(Offset(settings=SettingsAdderOffset(offset=4.0)))
    assert len(pipeline) == 4
    data_check += 4
    np.testing.assert_array_equal(pipeline(data_in, fs).data, data_check)

    backup = get_path_to_project() / "build_test" / "seq_mixed"
    with temporary_directory(backup) as tmpdir:
        path2build = Path(tmpdir).absolute()
        sets1 = CreateSequentialSettings(
            target=BuildPlatformTargets.FPGA,
            total_bitwidth=16,
            frac_bitwidth=0,
            do_signed=False,
            path2build=path2build,
        )
        pipeline.create_design(settings=sets1)

        path2src = path2build / "src"
        assert (path2build / "sequential_0.v").exists()
        assert (path2src / "adder_off_0.txt").exists()
        assert (path2src / "amplifier_1.txt").exists()
        assert (path2src / "amplifier_2.txt").exists()
        assert (path2src / "adder_off_3.txt").exists()
