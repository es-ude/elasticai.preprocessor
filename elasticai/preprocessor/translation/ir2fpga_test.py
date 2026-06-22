from pathlib import Path
from shutil import rmtree

import pytest

from elasticai.preprocessor import get_path_to_project

from .ir2fpga import load_and_build_form_plugin


@pytest.fixture(scope="module", autouse=True)
def path_to_build():
    path = get_path_to_project() / "build_files" / "ir"
    if path.exists():
        rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    yield path


def test_load_and_build_form_plugin(path_to_build: Path) -> None:
    load_and_build_form_plugin(
        type="adder_ripple_carry_signed",
        id="0",
        params={"BITWIDTH": 8},
        packages=["adders"],
        path2save=path_to_build,
    )

    files_check = ["adder_half.v", "adder_ripple_carry_signed_0.v", "adder_full.v"]
    files_check.sort()
    files_avai = [file.name for file in path_to_build.glob("*.*")]
    files_avai.sort()
    assert files_avai == files_check
