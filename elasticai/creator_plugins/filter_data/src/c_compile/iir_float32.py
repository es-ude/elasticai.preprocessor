from datetime import datetime
from pathlib import Path

import numpy as np

import elasticai.creator_plugins.filter_data as design_plugin
from elasticai.preprocessor import get_path_to_project
from elasticai.preprocessor.filter import Filtering, SettingsFilter
from elasticai.preprocessor.translation.ir2c import generate_c_files, replace_variables_with_parameters


def _format_float32(value: float) -> str:
    return f"{float(np.float32(value)):.9e}f"


def build_filter_iir_float32(
    settings: SettingsFilter,
    filter_id: str = "",
    path2save: Path = get_path_to_project("build"),
    define_path: str = "src",
) -> None:
    """Generate a stateful Float32 IIR filter for MCU or workstation C builds."""
    if settings.type.lower() != "iir":
        raise ValueError(f"Key 'type' must be 'iir' and not '{settings.type.lower()}'")

    coefficients = Filtering(settings=settings).get_coeffs()
    coefficient_a = np.asarray(coefficients.a, dtype=np.float64)
    coefficient_b = np.asarray(coefficients.b, dtype=np.float64)
    if coefficient_a.ndim != 1 or coefficient_b.ndim != 1:
        raise ValueError("IIR coefficients must be one-dimensional")
    if coefficient_a.size != coefficient_b.size:
        raise ValueError("Float32 IIR generation requires equally sized a and b coefficients")
    if coefficient_a.size < 2 or coefficient_a[0] == 0.0:
        raise ValueError("IIR coefficient a[0] must be non-zero")
    if not np.all(np.isfinite(coefficient_a)) or not np.all(np.isfinite(coefficient_b)):
        raise ValueError("IIR coefficients must be finite")

    coefficient_b = settings.gain * coefficient_b / coefficient_a[0]
    coefficient_a = coefficient_a / coefficient_a[0]
    coefficient_values = np.concatenate(
        (coefficient_a.astype(np.float32), coefficient_b.astype(np.float32))
    )

    module_id = f"{settings.b_type.lower().split('pass')[0]}_{filter_id.lower()}"
    generated_header = f"filter_iir_float32_{module_id}.h"
    params = {
        "datetime_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
        "header_guard": f"FILTER_IIR_FLOAT32_{module_id.upper()}_H",
        "path2include": define_path,
        "template_name": "filter_iir_float32_template.h",
        "generated_header": generated_header,
        "device_id": module_id,
        "fs": f"{settings.fs}",
        "filter_type": f"{settings.b_type}, {settings.f_type}",
        "filter_corner": ", ".join(map(str, settings.f_filt)),
        "filter_order": str(settings.n_order),
        "coeff_order": str(coefficient_a.size),
        "tap_order": str(coefficient_a.size - 1),
        "coeffs_string": ", ".join(_format_float32(value) for value in coefficient_values),
    }

    generated = _generate_filter_iir_float32_template()
    generate_c_files(
        path2save=path2save,
        template_name=params["template_name"],
        file_name="filter_iir_float32",
        module_id=module_id,
        proto_file=replace_variables_with_parameters(generated["head"], params),
        impl_file=replace_variables_with_parameters(generated["func"], params),
        path2template=Path(design_plugin.__file__).parent / "c",
    )


def _generate_filter_iir_float32_template() -> dict[str, list[str]]:
    header = [
        "// --- Generating a stateful Float32 IIR filter (Direct Form II)",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: N = {$filter_order}, f_c = [{$filter_corner}] Hz @ {$fs} Hz ({$filter_type})",
        "#ifndef {$header_guard}",
        "#define {$header_guard}",
        '#include "{$path2include}/{$template_name}"',
        "#ifdef __cplusplus",
        'extern "C" {',
        "#endif",
        "DEF_NEW_IIR_FLOAT32_FILTER_PROTO({$device_id}, {$tap_order})",
        "#ifdef __cplusplus",
        "}",
        "#endif",
        "#endif",
    ]
    implementation = [
        "// --- Generating a stateful Float32 IIR filter (Direct Form II)",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: N = {$filter_order}, f_c = [{$filter_corner}] Hz @ {$fs} Hz ({$filter_type})",
        '#include "{$path2include}/{$generated_header}"',
        "DEF_NEW_IIR_FLOAT32_FILTER_IMPL({$device_id}, {$coeff_order}, {$tap_order}, {$coeffs_string})",
    ]
    return {"head": header, "func": implementation}
