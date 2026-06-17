from pathlib import Path
from typing import Any

import elasticai.creator.ir2verilog as ir
from elasticai.creator.arithmetic import int_arithmetic
from elasticai.creator.file_generation import find_project_root as get_path_to_build
from elasticai.creator.ir import Registry, attribute
from elasticai.creator.ir2verilog import Ir2Verilog, factory

from elasticai.preprocessor.waveform_generator import WaveformGenerator


def prepare_waveform(
    waveform: str, bitwidth: int, num_params: int, do_opt: bool = False, is_signed: bool = True
) -> list[int]:
    params = num_params if not do_opt else 4 * num_params - 3

    arith = int_arithmetic(total_bits=bitwidth, signed=True)
    sig = (
        WaveformGenerator(sampling_rate=float(params - 1), add_noise=False)
        .generate_waveform_quant_fxp(
            time_points=[0.0],
            time_duration=[1.0],
            waveform_select=[waveform],
            polarity_cathodic=[False],
            bitwidth=bitwidth,
            bitfrac=bitwidth - 1,
            do_opt=do_opt,
            signed=is_signed,
        )
        .signal.tolist()
    )

    if not do_opt:
        sig.append(0.0 if is_signed else 0.5)

    sig = [arith.clamp(arith.cut_as_integer(val * arith.minimum_as_integer * (-1))) for val in sig]
    sig0 = sig.copy()
    sig0.reverse()
    return sig0


def load_and_plugin(
    type: str,
    id: str,
    params: dict[str, Any],
    packages: list = ["waveform"],
    path2save: Path = get_path_to_build() / "build",
    use_bram: bool = False,
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
    if use_bram:
        _load_and_plugin_design(
            "bram_single_port",
            id,
            {
                "BITWIDTH": params["BITWIDTH"],
                "RAMWIDTH": params["RAMWIDTH"],
                "DATAFILE": params["PATH2MEM"],
            },
            ["bram"],
            path2save,
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
