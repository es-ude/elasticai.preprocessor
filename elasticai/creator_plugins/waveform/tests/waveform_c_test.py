from pathlib import Path
import subprocess

import pytest
from elasticai.equichecker import CompileLoader
from elasticai.preprocessor import get_path_to_project
from elasticai.preprocessor.waveform_generator import WaveformGenerator
from elasticai.creator_plugins.waveform.utils import prepare_waveform


@pytest.fixture(scope="session", autouse=True)
def build_path():
    BUILD_PATH = get_path_to_project("build_test") / "waveform"
    BUILD_PATH.mkdir(parents=True, exist_ok=True)
    yield BUILD_PATH


def test_equivalence_compiled(build_path: Path):
    data = WaveformGenerator(sampling_rate=1.)._create_design_c(
        waveform="SINE_FULL",
        num_params=101,
        is_signed=True,
        bitwidth=8,
        id="0",
        path2save=build_path,
        do_opt=False,
        path2include=""
    )

    temp_dir = build_path / "temp"
    header_files = [file.as_posix() for file in build_path.glob("*.h")]
    source_files = [file.as_posix() for file in build_path.glob("*.c")]

    compile_loader = CompileLoader(
        headers=header_files,
        sources=source_files,
        build_dir=temp_dir.as_posix(),
    )
    compile_loader.load()
    #c_add2 = compile_loader.get("add")
