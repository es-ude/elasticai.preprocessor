from datetime import datetime
from pathlib import Path
from shutil import copyfile

import numpy as np

from elasticai.preprocessor.translation.ir2c import get_embedded_datatype

from .pipeline_settings import (
    SettingsPipeline,
    SettingsPipelineDownsampling,
    SettingsPipelineFilter,
    TargetsDownsamplingC,
    TargetsFilterC,
)

# Navigate from this file up to elasticai/creator_plugins/ and then into each
# plugin's c/ directory.  Using __file__ directly avoids issues with namespace
# packages (e.g. 'downsampling') that have no __init__.py and return __file__=None.
_CREATOR_PLUGINS = Path(__file__).parents[3]  # elasticai/creator_plugins/
_WINDOWER_C = _CREATOR_PLUGINS / "windower" / "c"
_FILTER_C = _CREATOR_PLUGINS / "filter_data" / "c"
_POLY_C = _CREATOR_PLUGINS / "datarate" / "c"


# Lookup tables: for each filter/downsampling type,
# Each entry: (template_header | None, fn_name_template, macro_renderer)
def _iir_macro(id: str, dtype: str, f: SettingsPipelineFilter) -> str:
    coeff_lgth = len(f.iir_a)
    tap_lgth = coeff_lgth - 1
    coeffs = ", ".join(map(str, f.iir_a)) + ", " + ", ".join(map(str, f.iir_b))
    return f"DEF_NEW_IIR_FILTER_IMPL({id}, {dtype}, {coeff_lgth}, {tap_lgth}, {coeffs})"


def _fir_macro(id: str, dtype: str, f: SettingsPipelineFilter) -> str:
    coeffs = ", ".join(map(str, f.fir_coefficients))
    return f"DEF_NEW_FIR_FILTER_IMPL({id}, {dtype}, {f.fir_order}, {coeffs})"


def _mavg_macro(id: str, dtype: str, f: SettingsPipelineFilter) -> str:
    coeff = 1.0 / f.mavg_order
    return f"DEF_NEW_MAVG_FILTER_IMPL({id}, {dtype}, {f.mavg_order}, {coeff})"


def _delay_macro(id: str, dtype: str, f: SettingsPipelineFilter) -> str:
    return f"DEF_NEW_FIR_DELAY_IMPL({id}, {dtype}, {f.delay_order})"


_FILTER_TABLE: dict[TargetsFilterC, tuple[str | None, str | None, object]] = {
    #                               template file              fn_name               macro renderer
    TargetsFilterC.IIR: ("filter_iir_template.h", "filt_iir", _iir_macro),
    TargetsFilterC.FIR: ("filter_fir_template.h", "filt_fir", _fir_macro),
    TargetsFilterC.MovingAverage: ("filter_mavg_template.h", "filt_fir_simple", _mavg_macro),
    TargetsFilterC.FirDelay: ("filter_fir_delay_template.h", "filt_fir_delay", _delay_macro),
    TargetsFilterC.Bypass: (None, None, None),
}


def _poly1_macro(id: str, dtype: str, d: SettingsPipelineDownsampling) -> str:
    return f"DEF_DOWNSAMPLING_POLY_ONE_IMPL({id}, {dtype}, {d.ratio})"


def _poly2_macro(id: str, dtype: str, d: SettingsPipelineDownsampling) -> str:
    return f"DEF_DOWNSAMPLING_POLY_TWO_IMPL({id}, {dtype}, {d.ratio})"


def _cic_macro(id: str, dtype: str, d: SettingsPipelineDownsampling) -> str:
    return f"DEF_DOWNSAMPLING_CIC_IMPL({id}, {dtype}, {d.cic_stages}, {d.ratio})"


def _simple_macro(id: str, dtype: str, d: SettingsPipelineDownsampling) -> str:
    return f"DEF_NEW_DO_SIMPLE_TAP_IMPL({id}, {dtype}, {d.ratio})"


_DECIMATION_TABLE: dict[TargetsDownsamplingC, tuple[str | None, str | None, object]] = {
    #                                    template file                      fn_name          macro renderer
    TargetsDownsamplingC.PolyOne: ("downsampling_poly_template.h", "calc_do_poly_one", _poly1_macro),
    TargetsDownsamplingC.PolyTwo: ("downsampling_poly_template.h", "calc_do_poly_two", _poly2_macro),
    TargetsDownsamplingC.CIC: ("downsampling_cic_template.h", "calc_cic", _cic_macro),
    TargetsDownsamplingC.Simple: ("downsampling_simple_template.h", "calc_do_simple", _simple_macro),
    TargetsDownsamplingC.Bypass: (None, None, None),
}


def build_pipeline(
    settings: SettingsPipeline,
    path2save: Path,
    pipeline_id: str = "0",
    define_path: str = "src",
) -> None:
    """Generate C files for a streaming pipeline from a SettingsPipeline container.
    Args:
        settings:     Complete pipeline configuration (SettingsPipeline).
        path2save:    Directory where generated files are written.
        pipeline_id:  ID appended to all generated function names.
        define_path:  Include prefix written into the generated #include lines.
    """
    _validate_pipeline_settings(settings)

    path2save.mkdir(parents=True, exist_ok=True)
    module_id = pipeline_id.lower()
    dtype = get_embedded_datatype(settings.bitwidth, settings.signed)
    ts = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    wl = settings.window.window_length
    nshift = wl - settings.window.overlap_length

    filt_tmpl, filt_fn_base, filt_macro_fn = _FILTER_TABLE[settings.filter.method]
    deci_tmpl, deci_fn_base, deci_macro_fn = _DECIMATION_TABLE[settings.downsampling.method]

    # --- collect which template headers are needed ---
    templates_needed: list[tuple[Path, str]] = [(_WINDOWER_C, "windower_sliding_template.h")]
    if filt_tmpl:
        templates_needed.append((_FILTER_C, filt_tmpl))
    if deci_tmpl:
        templates_needed.append((_POLY_C, deci_tmpl))

    # Deduplicate (PolyOne and PolyTwo share the same template file)
    seen: set[str] = set()
    unique_templates = [(d, n) for d, n in templates_needed if not (n in seen or seen.add(n))]

    for src_dir, name in unique_templates:
        copyfile(src=str(src_dir / name), dst=str(path2save / name))

    # --- build the #include lines ---
    includes = "\n".join(f'#include "{define_path}/{name}"' for _, name in unique_templates)

    # --- build stage macro lines ---
    stage_macros: list[str] = []
    if filt_macro_fn:
        stage_macros.append(filt_macro_fn(module_id, dtype, settings.filter))
    if deci_macro_fn:
        stage_macros.append(deci_macro_fn(module_id, dtype, settings.downsampling))
    stage_macros.append(f"DEF_WINDOWER_SLIDING_IMPL({module_id}, {dtype}, {wl}, {nshift})")

    # --- build wrapper function body ---
    wrapper_lines: list[str] = [f"bool calc_pipeline_{module_id}({dtype} data, {dtype} *out) {{"]
    current = "data"
    if filt_fn_base:
        wrapper_lines += [
            f"    {dtype} filt_out;",
            f"    if (!{filt_fn_base}_{module_id}({current}, &filt_out)) {{ return false; }}",
        ]
        current = "filt_out"
    if deci_fn_base:
        wrapper_lines += [
            f"    {dtype} down_out;",
            f"    if (!{deci_fn_base}_{module_id}({current}, &down_out)) {{ return false; }}",
        ]
        current = "down_out"
    wrapper_lines.append(f"    return calc_windower_sliding_{module_id}({current}, out);")
    wrapper_lines.append("}")

    # --- stage descriptions for the header comment ---
    filter_desc = settings.filter.method.value
    deci_desc = f"{settings.downsampling.method.value}(ratio={settings.downsampling.ratio})"
    wind_desc = f"wl={wl}, nshift={nshift}"

    header_comment = (
        f"// --- Generated pipeline: {filter_desc} → {deci_desc} → windower({wind_desc})\n"
        f"// Copyright @ UDE-IES\n"
        f"// Code generated on: {ts}\n"
        f"// ID = {module_id}, type = {dtype}"
    )

    # --- write pipeline_{id}.h ---
    proto_path = path2save / f"pipeline_{module_id}.h"
    proto_path.write_text(
        f"{header_comment}\n"
        f"#include <stdbool.h>\n"
        f"#include <stdint.h>\n"
        f"bool calc_pipeline_{module_id}({dtype} data, {dtype} *out);\n"
    )

    # --- write pipeline_{id}.c ---
    impl_path = path2save / f"pipeline_{module_id}.c"
    impl_path.write_text(
        f"{header_comment}\n"
        f"{includes}\n"
        f"\n" + "\n".join(stage_macros) + "\n"
        "\n" + "\n".join(wrapper_lines) + "\n"
    )


def _validate_pipeline_settings(settings: SettingsPipeline) -> None:
    f = settings.filter
    d = settings.downsampling

    match f.method:
        case TargetsFilterC.IIR:
            if not f.iir_a or not f.iir_b:
                raise ValueError("IIR filter: iir_a and iir_b must be non-empty")
            if len(f.iir_a) != len(f.iir_b):
                raise ValueError("IIR filter: iir_a and iir_b must have the same length")
        case TargetsFilterC.FIR:
            if not f.fir_coefficients:
                raise ValueError("FIR filter: fir_coefficients must be non-empty")
            if f.fir_order < len(f.fir_coefficients):
                raise ValueError("FIR filter: fir_order must be >= len(fir_coefficients)")
        case TargetsFilterC.MovingAverage:
            if f.mavg_order < 1:
                raise ValueError("MovingAverage filter: mavg_order must be >= 1")
        case TargetsFilterC.FirDelay:
            if f.delay_order < 1:
                raise ValueError("FirDelay filter: delay_order must be >= 1")

    match d.method:
        case TargetsDownsamplingC.PolyOne | TargetsDownsamplingC.PolyTwo | TargetsDownsamplingC.Simple:
            if d.ratio < 1:
                raise ValueError("dsr must be >= 1")
            if not np.log2(d.ratio).is_integer():
                raise ValueError("dsr must be 2^n for Poly1/Poly2/Simple decimation")
        case TargetsDownsamplingC.CIC:
            if d.ratio < 1:
                raise ValueError("CIC: ratio must be >= 1")
            if d.cic_stages < 1:
                raise ValueError("CIC: cic_stages must be >= 1")

    if settings.bitwidth not in range(2, 33):
        raise ValueError("bitwidth must be between 2 and 32")
