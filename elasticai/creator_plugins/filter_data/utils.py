from pathlib import Path
from typing import Any

from elasticai.preprocessor import get_path_to_project
from elasticai.preprocessor.translation import load_and_build_form_plugin


def load_and_plugin(
    type: str,
    id: str,
    params: dict[str, Any],
    packages: list,
    path2save: Path = get_path_to_project() / "build",
    use_dsp_mult: bool = True,
    add_mac: bool = True,
    add_ringbuffer: bool = False,
) -> None:
    load_and_build_form_plugin(type, id, params, packages, path2save)
    if add_ringbuffer:
        load_and_build_form_plugin(
            "ring_buffer",
            "",
            params={"BITWIDTH": params["BITWIDTH"], "SAMPLES": params["LENGTH"]},
            packages=["windower"],
            path2save=path2save,
        )
    if add_mac:
        load_and_build_form_plugin(
            "mac_array",
            "",
            params={
                "INPUT_BITWIDTH": params["BITWIDTH"],
                "INPUT_NUM_DATA": params["LENGTH"],
                "NUM_MULT_PARALLEL": 1,
            },
            packages=["mac"],
            path2save=path2save,
        )
        if use_dsp_mult:
            load_and_build_form_plugin(
                "mult_dsp_signed",
                "",
                params={"BITWIDTH": params["BITWIDTH"]},
                packages=["multipliers"],
                path2save=path2save,
            )
        else:
            load_and_build_form_plugin(
                "mult_lut_signed",
                "",
                params={"BITWIDTH": params["BITWIDTH"]},
                packages=["multipliers", "adders"],
                path2save=path2save,
            )
