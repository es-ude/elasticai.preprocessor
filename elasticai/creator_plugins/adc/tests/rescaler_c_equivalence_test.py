from pathlib import Path
from shutil import which
from uuid import uuid4

import numpy as np
import pytest
from elasticai.creator.arithmetic import int_arithmetic
from elasticai.equichecker import CompileLoader, compare_values

from elasticai.creator_plugins.adc.src.c_compile import build_rescaler
from elasticai.preprocessor import get_path_to_project
from elasticai.preprocessor.adc import Rescaler, RescalerSettings, RescalerTargets
from elasticai.preprocessor.translation.cocotb_tmp import temporary_directory

pytestmark = pytest.mark.skipif(which("cc") is None, reason="requires a C compiler")

ADC_CONFIGS = [
    pytest.param(
        RescalerSettings(method=RescalerTargets.FxP, total_bits=8, frac_bits=0, is_signed=True),
        "signed char",
        id="int8_signed",
    ),
    pytest.param(
        RescalerSettings(
            method=RescalerTargets.IntSym,
            total_bits=8,
            frac_bits=0,
            is_signed=False,
        ),
        "unsigned char",
        id="uint8_unsigned",
    ),
    pytest.param(
        RescalerSettings(
            method=RescalerTargets.IntAsym,
            total_bits=12,
            frac_bits=0,
            is_signed=True,
        ),
        "signed short",
        id="int16_12bit",
    ),
]


def _py_adc(settings: RescalerSettings) -> int:
    if settings.is_signed:
        min_int = -(2 ** (settings.total_bits - 1))
        max_int = 2 ** (settings.total_bits - 1) - 1
    else:
        min_int = 0
        max_int = 2**settings.total_bits - 1
    ival = min_int
    return max(min_int, min(max_int, ival))


def test_build_adc_quant_generates_c_files(tmp_path: Path) -> None:
    settings = RescalerSettings(
        method=RescalerTargets.FxP,
        total_bits=12,
        frac_bits=0,
        is_signed=True,
    )
    build_rescaler(settings=settings, path2save=tmp_path, id="0", bitwidth_input=8)
    assert (tmp_path / "rescaler_template.h").exists()
    assert (tmp_path / "rescaler_0.h").exists()
    assert (tmp_path / "rescaler_0.c").exists()


@pytest.mark.simulation
@pytest.mark.parametrize("settings,c_type", ADC_CONFIGS)
def test_adc_c_matches_python(
    tmp_path: Path,
    settings: RescalerSettings,
    c_type: str,
) -> None:
    backup = get_path_to_project() / "build_test" / f"rescaler_{settings.method.value}"
    with temporary_directory(backup) as tmp_path:
        output_dir = tmp_path / "src"
        build_rescaler(
            settings=settings,
            path2save=output_dir,
            id="0",
            define_path=".",
            bitwidth_input=settings.total_bits,
        )

        adapter = tmp_path / "adapter.h"
        adapter.write_text(f"_Bool rescaling_data_0(int16_t data, {c_type} *out);\n")
        loader = CompileLoader(
            headers=str(adapter),
            sources=[str(output_dir / "rescaler_0.c")],
            build_dir=str(tmp_path / "cffi-build"),
            module_name=f"rescaler_equivalence_{uuid4().hex}",
        )
        loader.load()

        out = loader.ffi().new(f"{c_type} *")
        calc_adc = loader.get("rescaling_data_0")
        arith = int_arithmetic(total_bits=settings.total_bits, signed=settings.is_signed)
        data_in = np.random.randint(
            low=arith.minimum_as_integer, high=arith.maximum_as_integer, size=1000
        )

        dut = Rescaler(settings=settings, update_params=False)
        data_py = dut.forward(x=data_in)
        for idx, (inp, outp) in enumerate(zip(data_in, data_py)):
            ready = calc_adc(inp, out)
            assert ready
            c_value = int(out[0])
            passed, reason = compare_values(int(outp), c_value)
            assert passed, f"index={idx}: input={inp}, python={outp}, c={c_value}: {reason}"
