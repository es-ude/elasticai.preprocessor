from datetime import datetime
from pathlib import Path

import elasticai.creator_plugins.rescaler as design_plugin
from elasticai.preprocessor.adc import RescalerSettings
from elasticai.preprocessor.translation.ir2c import (
    generate_c_files,
    get_embedded_datatype,
    replace_variables_with_parameters,
)


def _c_float(v: float) -> str:
    s = f"{v:.8g}"
    if "." not in s and "e" not in s:
        s += ".0"
    return s + "f"


def build_rescaler(
    settings: RescalerSettings,
    path2save: Path,
    id: str,
    bitwidth_input: int,
    define_path: str = "src",
    scaling_factor: float = 1.0,
    zero_point: float = 0.0,
) -> None:
    """Generate C files for ADC quantization (voltage → integer, streaming interface).
    Args:
        settings:       ADC settings defining resolution and voltage range
        path2save:      Path to save the .h / .c output files
        id:             String with ID appended to the generated function name
        bitwidth_input: Integer with input total bitwidth
        define_path:    Include path written into the generated #include line
        scaling_factor: Floating with data scaling value
        zero_point:     Floating with offset to define the zero point
    """
    assert settings.total_bits in range(1, 33), "total_bits must be between 1 and 32"

    module_id = id.lower()
    params = {
        "datetime_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
        "path2include": define_path,
        "template_name": "rescaler_template.h",
        "device_id": module_id.upper(),
        "in_type": get_embedded_datatype(bitwidth_input, settings.is_signed),
        "out_type": get_embedded_datatype(settings.total_bits, settings.is_signed),
        "total_bits": str(settings.total_bits),
        "is_signed": "true" if settings.is_signed else "false",
        "scale_val": _c_float(scaling_factor),
        "zero_point": _c_float(zero_point),
    }
    template_c = _generate_adc_quant_template()
    generate_c_files(
        path2save=path2save,
        template_name=params["template_name"],
        file_name="rescaler",
        module_id=module_id,
        proto_file=replace_variables_with_parameters(template_c["head"], params),
        impl_file=replace_variables_with_parameters(template_c["func"], params),
        path2template=Path(design_plugin.__file__).parent / "c",
    )


def _generate_adc_quant_template() -> dict[str, list[str]]:
    header_template = [
        "// --- Generating ADC quantization",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: ID = {$device_id}, in_type = {$in_type}, out_type = {$out_type}, bits = {$total_bits}",
        '#include "{$path2include}/{$template_name}"',
        "DEF_RESCALER_QUANT_PROTO({$device_id}, {$in_type}, {$out_type})",
    ]
    implementation_template = [
        "// --- Generating ADC quantization",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: ID = {$device_id}, in_type = {$in_type}, out_type = {$out_type}, bits = {$total_bits}",
        '#include "{$path2include}/{$template_name}"',
        "DEF_RESCALER_QUANT_IMPL({$device_id}, {$in_type}, {$out_type}, {$total_bits}, {$is_signed}, {$scale_val}, {$zero_point})",
    ]
    return {"head": header_template, "func": implementation_template}
