from pathlib import Path
from shutil import which

import pytest

from elasticai.preprocessor.downsampling import DownSampling, SettingsDownSampling

pytestmark = pytest.mark.skipif(which("cc") is None, reason="requires a C compiler")


@pytest.mark.parametrize("target", ["mcu", "pc"])
def test_create_design_generates_subsampling_c_files(tmp_path: Path, target: str) -> None:
    downsampler = DownSampling(SettingsDownSampling(sampling_rate=1000.0, dsr=3))

    downsampler.create_design(target, 8, "0", tmp_path, signed=True)

    assert (tmp_path / "downsampling_subsampling_0.c").exists()
    assert (tmp_path / "downsampling_subsampling_0.h").exists()
    assert (tmp_path / "downsampling_subsampling_template.h").exists()


def test_create_design_rejects_unknown_target(tmp_path: Path) -> None:
    downsampler = DownSampling(SettingsDownSampling(sampling_rate=1000.0, dsr=3))

    with pytest.raises(ValueError, match="Target unknown is not supported"):
        downsampler.create_design("unknown", 8, "0", tmp_path)


def test_create_design_rejects_fpga_target(tmp_path: Path) -> None:
    downsampler = DownSampling(SettingsDownSampling(sampling_rate=1000.0, dsr=3))

    with pytest.raises(NotImplementedError, match="FPGA downsampling"):
        downsampler.create_design("fpga", 8, "0", tmp_path)


def test_create_design_rejects_invalid_downsampling_ratio(tmp_path: Path) -> None:
    downsampler = DownSampling(SettingsDownSampling(sampling_rate=1000.0, dsr=0))

    with pytest.raises(ValueError, match="dsr must be >= 1"):
        downsampler.create_design("mcu", 8, "0", tmp_path)
