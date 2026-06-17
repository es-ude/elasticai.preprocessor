from pathlib import Path
import subprocess

import pytest
from elasticai.equichecker import PrecompiledLoader, CompileLoader
from elasticai.preprocessor import get_path_to_project


def add(a, b):
    return a + b


@pytest.fixture(scope="session", autouse=True)
def build_path():
    BUILD_PATH = get_path_to_project("build_test") / "adder"
    BUILD_PATH.mkdir(parents=True, exist_ok=True)
    yield BUILD_PATH


@pytest.fixture(scope="session", autouse=True)
def template_path():
    TEMPLATE_PATH = get_path_to_project("tests") / "templates"
    TEMPLATE_PATH.mkdir(parents=True, exist_ok=True)
    yield TEMPLATE_PATH


def test_equivalence_precompiled(build_path: Path, template_path: Path) -> None:
    precompiled_path = build_path / "libadd.so"
    header_files = [file.as_posix() for file in template_path.glob("*.h")]
    source_files = [file.as_posix() for file in template_path.glob("*.c")]

    subprocess.run(
        ["cc", "-shared", "-fPIC", "-o", precompiled_path.as_posix(), source_files[0]],
        check=True,
    )
    loader = PrecompiledLoader(
        library_path=precompiled_path.as_posix(),
        headers=header_files
    )
    loader.load()
    c_add = loader.get("add")

    a = 5
    b = 7
    assert add(a, b) == c_add(a, b)


def test_equivalence_compiled(build_path: Path, template_path: Path):
    temp_dir = build_path / "temp"
    header_files = [file.as_posix() for file in template_path.glob("*.h")]
    source_files = [file.as_posix() for file in template_path.glob("*.c")]

    compile_loader = CompileLoader(
        headers=header_files,
        sources=source_files,
        build_dir=temp_dir.as_posix(),
    )
    compile_loader.load()
    c_add2 = compile_loader.get("add")

    a = 10
    b = 15
    assert add(a, b) == c_add2(a, b)
