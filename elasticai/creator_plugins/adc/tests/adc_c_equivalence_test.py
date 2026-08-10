from pathlib import Path
from shutil import which
from uuid import uuid4

import numpy as np
import pytest
from elasticai.equichecker import CompileLoader, compare_values

from elasticai.creator_plugins.adc.src.c_compile import build_adc_quant
from elasticai.preprocessor.adc import SettingsResampler

pytestmark = pytest.mark.skipif(which("cc") is None, reason="requires a C compiler")

ADC_CONFIGS = [
    pytest.param(
        SettingsResampler(
            total_bits=8,
            frac_bits=0,
            is_signed=True,
            srate_orig=1.0,
            srate_new=1.0,
            vneg=-1.0,
            vpos=1.0,
        ),
        "signed char",
        id="int8_signed",
    ),
    pytest.param(
        SettingsResampler(
            total_bits=8,
            frac_bits=0,
            is_signed=False,
            srate_orig=1.0,
            srate_new=1.0,
            vneg=0.0,
            vpos=3.3,
        ),
        "unsigned char",
        id="uint8_unsigned",
    ),
    pytest.param(
        SettingsResampler(
            total_bits=12,
            frac_bits=0,
            is_signed=True,
            srate_orig=1.0,
            srate_new=1.0,
            vneg=-3.3,
            vpos=3.3,
        ),
        "signed short",
        id="int16_12bit",
    ),
    pytest.param(
        SettingsResampler(
            total_bits=16,
            frac_bits=0,
            is_signed=False,
            srate_orig=1.0,
            srate_new=1.0,
            vneg=0.0,
            vpos=5.0,
        ),
        "unsigned short",
        id="uint16_16bit",
    ),
]


def _py_adc(voltage: float, settings: SettingsResampler) -> int:
    """Python-Referenz: spiegelt die C-Formel aus adc_template.h exakt wider."""
    lsb = (settings.vpos - settings.vneg) / (2**settings.total_bits)
    if settings.is_signed:
        min_int = -(2 ** (settings.total_bits - 1))
        max_int = 2 ** (settings.total_bits - 1) - 1
    else:
        min_int = 0
        max_int = 2**settings.total_bits - 1
    clamped = max(float(settings.vneg), min(float(settings.vpos), voltage))
    steps = round((clamped - settings.vneg) / lsb)
    ival = steps + min_int
    return max(min_int, min(max_int, ival))


def _make_test_voltages(settings: SettingsResampler) -> np.ndarray:
    """Erzeugt Testeingangsspannungen: in-range, Grenzen und clamping-Fälle."""
    v_range = settings.vpos - settings.vneg
    fractions = np.array([-0.15, 0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0, 1.15])
    return settings.vneg + fractions * v_range


def test_build_adc_quant_generates_c_files(tmp_path: Path) -> None:
    settings = SettingsResampler(
        total_bits=12,
        frac_bits=0,
        is_signed=True,
        srate_orig=1.0,
        srate_new=1.0,
        vneg=-3.3,
        vpos=3.3,
    )
    build_adc_quant(settings=settings, path2save=tmp_path, adc_id="0")
    assert (tmp_path / "adc_template.h").exists()
    assert (tmp_path / "adc_0.h").exists()
    assert (tmp_path / "adc_0.c").exists()


@pytest.mark.parametrize("settings,c_type", ADC_CONFIGS)
def test_adc_c_matches_python(
    tmp_path: Path,
    settings: SettingsResampler,
    c_type: str,
) -> None:
    output_dir = tmp_path / "src"
    build_adc_quant(settings=settings, path2save=output_dir, adc_id="0", define_path=".")

    adapter = tmp_path / "adapter.h"
    adapter.write_text(f"_Bool calc_adc_0(float data, {c_type} *out);\n")
    loader = CompileLoader(
        headers=str(adapter),
        sources=[str(output_dir / "adc_0.c")],
        build_dir=str(tmp_path / "cffi-build"),
        module_name=f"adc_equivalence_{uuid4().hex}",
    )
    loader.load()

    voltages = _make_test_voltages(settings)

    out = loader.ffi().new(f"{c_type} *")
    calc_adc = loader.get("calc_adc_0")
    for index, voltage in enumerate(voltages.tolist()):
        python_value = _py_adc(voltage, settings)
        ready = calc_adc(float(voltage), out)
        assert ready, f"index={index}: calc_adc_0 returned false (should always be true)"
        c_value = int(out[0])
        passed, reason = compare_values(python_value, c_value)
        assert passed, (
            f"index={index}: voltage={voltage:.4f}V, python={python_value}, c={c_value}: {reason}"
        )
