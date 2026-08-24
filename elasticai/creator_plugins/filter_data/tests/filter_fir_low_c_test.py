from pathlib import Path
from shutil import which

import numpy as np
import pytest

from elasticai.creator_plugins.filter_data.tests._filter_c_check import check_filter_c_equivalence
from elasticai.preprocessor import get_path_to_project
from elasticai.preprocessor.filter import Filtering, SettingsFilter
from elasticai.preprocessor.translation.cocotb_tmp import temporary_directory

pytestmark = pytest.mark.skipif(which("cc") is None, reason="requires a C compiler")


INPUT_DATA = np.array(
    [
        0,
        64,
        -32,
        48,
        -64,
        16,
        32,
        -48,
        64,
        0,
        -16,
        48,
        -32,
        64,
        -64,
        32,
        16,
        -8,
        24,
        -40,
        8,
        0,
        12,
        -20,
    ],
    dtype=np.int8,
)

INTEGER_CONFIGS = [
    pytest.param(8, np.int8, "signed char", id="int8"),
    pytest.param(32, np.int32, "signed int", id="int32"),
]


@pytest.fixture()
def tmp_path() -> Path:
    path2folder = get_path_to_project("build_test") / "fir_simple_low"
    path2folder.mkdir(parents=True, exist_ok=True)
    yield path2folder


@pytest.mark.parametrize("target", ["mcu", "pc"])
def test_build(tmp_path: Path, target: str) -> None:
    backup = tmp_path / f"build_{target}"
    with temporary_directory(backup) as tmpdir:
        settings = SettingsFilter(1.0, 1000.0, 1, [500.0], "fir", "butter", "lowpass")
        Filtering(settings).create_design(target, bitwidth=8, id="0", path2save=tmpdir)
        assert (tmpdir / "filter_fir_mavg_0.c").exists()
        assert (tmpdir / "filter_fir_mavg_0.h").exists()
        assert (tmpdir / "filter_mavg_template.h").exists()


@pytest.mark.parametrize("bitwidth,numpy_dtype,c_type", INTEGER_CONFIGS)
def test_build_equal(
    tmp_path: Path,
    bitwidth: int,
    numpy_dtype: type[np.generic],
    c_type: str,
) -> None:
    settings = SettingsFilter(1.0, 1000.0, 1, [500.0], "fir", "butter", "lowpass")
    check_filter_c_equivalence(
        settings,
        tmp_path,
        "filter_fir_mavg_0.c",
        "filt_fir_simple_0",
        bitwidth,
        numpy_dtype,
        c_type,
        INPUT_DATA,
    )
