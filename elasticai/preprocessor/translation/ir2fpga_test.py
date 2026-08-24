from pathlib import Path
from tempfile import TemporaryDirectory

from .ir2fpga import load_and_build_form_plugin


def test_load_and_build_form_plugin() -> None:
    with TemporaryDirectory() as directory:
        path2save = Path(directory)
        path2save.mkdir(parents=True, exist_ok=True)

        load_and_build_form_plugin(
            type="adder_ripple_carry_signed",
            id="0",
            params={"BITWIDTH": 8},
            packages=["adders"],
            path2save=path2save,
        )

        files_check = ["adder_half.v", "adder_ripple_carry_signed_0.v", "adder_full.v"]
        files_check.sort()
        files_avai = [file.name for file in path2save.glob("*.*")]
        files_avai.sort()
        assert files_avai == files_check
