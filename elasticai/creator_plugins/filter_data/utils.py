from pathlib import Path
from typing import Any

import elasticai.creator.ir2verilog as ir
from elasticai.creator.file_generation import find_project_root as get_path_to_build
from elasticai.creator.ir import Registry, attribute
from elasticai.creator.ir2verilog import Ir2Verilog, factory


def load_and_plugin(
    type: str,
    id: str,
    params: dict[str, Any],
    packages: list,
    path2save: Path = get_path_to_build() / "build",
    use_dsp_mult: bool = True,
    add_mac: bool = True,
    add_ringbuffer: bool = False,
) -> None:
    def _load_and_plugin_design(
        type: str, id: str, params: dict[str, Any], packages: list, path2save: Path
    ) -> None:
        design = _build_verilog_implementation(type=type, id=id, params=params)

        build_dir = Path(f"{path2save}/")
        build_dir.mkdir(exist_ok=True)

        translate = _prepare_translator(packages)
        for name, content in translate(design, Registry()):
            (build_dir / name).write_text("".join(content))

    _load_and_plugin_design(type, id, params, packages, path2save)
    if add_ringbuffer:
        _load_and_plugin_design(
            "ring_buffer",
            "",
            params={"BITWIDTH": params["BITWIDTH"], "SAMPLES": params["LENGTH"]},
            packages=["windower"],
            path2save=path2save,
        )
    if add_mac:
        _load_and_plugin_design(
            "mac",
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
            _load_and_plugin_design(
                "mult_dsp_signed",
                "",
                params={"BITWIDTH": params["BITWIDTH"]},
                packages=["multipliers"],
                path2save=path2save,
            )
        else:
            _load_and_plugin_design(
                "mult_lut_signed",
                "",
                params={"BITWIDTH": params["BITWIDTH"]},
                packages=["multipliers", "adders"],
                path2save=path2save,
            )


def _build_verilog_implementation(type: str, id: str, params: dict[str, Any]) -> ir.DataGraph:
    mod_name = f"{type}_{id}" if id else f"{type}"
    return factory.graph(
        attributes=attribute(**params),
        type=type,
        name=mod_name.lower(),
    )


def _prepare_translator(plugin_types: list[str]) -> Ir2Verilog:
    _translate = Ir2Verilog()
    loader = ir.PluginLoader(_translate)
    for plugin in plugin_types:
        loader.load_from_package(plugin)
    return _translate
