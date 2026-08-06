from datetime import datetime
from pathlib import Path

import elasticai.creator_plugins.adc as design_plugin
from elasticai.preprocessor.adc import SettingsResampler
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


def build_adc_quant(
    settings: SettingsResampler,
    path2save: Path,
    adc_id: str = "0",
    define_path: str = "src",
) -> None:
    """Generate C files for ADC quantization (voltage → integer, streaming interface).
    Args:
        settings:    ADC settings defining resolution and voltage range
        path2save:   Path to save the .h / .c output files
        adc_id:      ID appended to the generated function name
        define_path: Include path written into the generated #include line
    """
    assert settings.total_bits in range(1, 33), "total_bits must be between 1 and 32"

    module_id = adc_id.lower()
    params = {
        "datetime_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
        "path2include": define_path,
        "template_name": "adc_template.h",
        "device_id": module_id.upper(),
        "data_type": get_embedded_datatype(settings.total_bits, settings.is_signed),
        "total_bits": str(settings.total_bits),
        "is_signed": "1" if settings.is_signed else "0",
        "vneg": _c_float(settings.vneg),
        "vpos": _c_float(settings.vpos),
    }
    template_c = _generate_adc_quant_template()
    generate_c_files(
        path2save=path2save,
        template_name=params["template_name"],
        file_name="adc",
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
        "// Params: ID = {$device_id}, type = {$data_type}, bits = {$total_bits}, vneg = {$vneg}, vpos = {$vpos}",
        '#include "{$path2include}/{$template_name}"',
        "DEF_ADC_QUANT_PROTO({$device_id}, {$data_type})",
    ]
    implementation_template = [
        "// --- Generating ADC quantization",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: ID = {$device_id}, type = {$data_type}, bits = {$total_bits}, vneg = {$vneg}, vpos = {$vpos}",
        '#include "{$path2include}/{$template_name}"',
        "DEF_ADC_QUANT_IMPL({$device_id}, {$data_type}, {$total_bits}, {$is_signed}, {$vneg}, {$vpos})",
    ]
    return {"head": header_template, "func": implementation_template}
