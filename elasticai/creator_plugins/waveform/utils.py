from pathlib import Path
from typing import Any

from elasticai.creator.arithmetic import int_arithmetic

from elasticai.preprocessor import get_path_to_project
from elasticai.preprocessor.translation import load_and_build_form_plugin


def prepare_waveform(
    waveform: str, bitwidth: int, num_params: int, do_opt: bool = False, is_signed: bool = True
) -> list[int]:
    from elasticai.preprocessor.waveform_generator import WaveformGenerator

    params = num_params if not do_opt else 4 * num_params - 3

    arith = int_arithmetic(total_bits=bitwidth if not do_opt else bitwidth - 1, signed=is_signed)
    sig = (
        WaveformGenerator(sampling_rate=float(params - 1))
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

    scale = 2**bitwidth if not is_signed and not do_opt else 2 ** (bitwidth - 1)
    sig = [arith.clamp(arith.cut_as_integer(int(val * scale))) for val in sig]
    sig0 = sig.copy()
    sig0.reverse()
    return sig0


def load_and_plugin(
    type: str,
    id: str,
    params: dict[str, Any],
    packages: list = ["waveform"],
    path2save: Path = get_path_to_project() / "build",
    use_bram: bool = False,
) -> None:
    load_and_build_form_plugin(type, id, params, packages, path2save)
    if use_bram:
        load_and_build_form_plugin(
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
