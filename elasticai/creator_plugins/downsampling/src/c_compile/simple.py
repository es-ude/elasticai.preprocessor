from datetime import datetime
from pathlib import Path

import elasticai.creator_plugins.downsampling as design_plugin
from elasticai.preprocessor.translation.ir2c import (
    generate_c_files,
    get_embedded_datatype,
    replace_variables_with_parameters,
)


def build_downsampling_simple(
    downsampling_ratio: int,
    bitwidth: int,
    signed: bool,
    path2save: Path,
    downsampling_id: str = "0",
    define_path: str = "src",
) -> None:
    """Generate C files for downsampling simple.
    Args:
        downsampling_ratio: Integer with downsampling ratio for reducing the input sampling rate (SR_out = SR_in / OSR)
        bitwidth:           Bit width of each sample
        signed:             Decision if data values are signed [otherwise unsigned]
        path2save:          Path to save the .h / .c output-files.
        downsampling_id:    ID appended to function names
        define_path:        Include path written into the generated #include line.
    """
    assert bitwidth in range(2, 65), "Bitwidth must be between 2 and 64"
    assert downsampling_ratio > 0, "dsr must be >= 1"

    module_id = downsampling_id.lower()
    params = {
        "datetime_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
        "path2include": define_path,
        "template_name": "downsampling_simple_template.h",
        "device_id": module_id.upper(),
        "data_type": get_embedded_datatype(bitwidth, signed),
        "downsampling_ratio": str(downsampling_ratio),
    }
    template_c = _generate_downsampling_simple_template()
    generate_c_files(
        path2save=path2save,
        template_name=params["template_name"],
        file_name="downsampling_simple",
        module_id=module_id,
        proto_file=replace_variables_with_parameters(template_c["head"], params),
        impl_file=replace_variables_with_parameters(template_c["func"], params),
        path2template=Path(design_plugin.__file__).parent / "c",
    )


def _generate_downsampling_simple_template() -> dict[str, list[str]]:
    header_template = [
        "// --- Generating do_simple",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: ID = {$device_id}, type = {$data_type}, dsr = {$downsampling_ratio}",
        '#include "{$path2include}/{$template_name}"',
        "DEF_NEW_DO_SIMPLE_PROTO({$device_id}, {$data_type})",
    ]
    implementation_template = [
        "// --- Generating do_simple",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: ID = {$device_id}, type = {$data_type}, dsr = {$downsampling_ratio}",
        '#include "{$path2include}/{$template_name}"',
        "DOWNSAMPLING_SIMPLE_OUTPUT_LENGTH({$device_id}, {$downsampling_ratio})",
        "DEF_NEW_DO_SIMPLE_TAP_IMPL({$device_id}, {$data_type}, {$downsampling_ratio})",
        "DEF_GET_DO_SIMPLE_VAL_IMPL({$device_id}, {$data_type})",
    ]
    return {"head": header_template, "func": implementation_template}
