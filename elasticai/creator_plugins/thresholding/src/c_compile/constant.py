from datetime import datetime
from pathlib import Path

import elasticai.creator_plugins.thresholding as design_plugin
from elasticai.preprocessor.translation.ir2c import (
    generate_c_files,
    get_embedded_datatype,
    replace_variables_with_parameters,
)


def build_thresholding_const(
    threshold: int,
    bitwidth: int,
    signed: bool,
    path2save: Path,
    thresholding_id: str = "0",
    define_path: str = "src",
) -> None:
    """Generate C files for constant thresholding.
    Args:
        threshold:       constant threshold
        bitwidth:        bitwidth of each sample
        path2save:       Path to save the .h/.c output-files.
        thresholding_id: ID appended to function name
        define_path:     include path written into the generated #include line
    """
    
    assert bitwidth in range(2, 33), "bitwidth must be beteween 2 and 32."
    
    module_id = thresholding_id.lower()
    params = {
        "threshold": str(threshold),
        "datetime_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
        "path2include": define_path,
        "template_name": "thresholding_constant_template.h",
        "module_id": module_id,
        "device_id": module_id.upper(),
        "data_type": get_embedded_datatype(bitwidth, signed),
    }
    template_c = _generate_thresholding_constant_template()
    generate_c_files(
        path2save=path2save,
        template_name=params["template_name"],
        file_name="thresholding_constant",
        module_id=module_id,
        proto_file=replace_variables_with_parameters(template_c["head"], params),
        impl_file=replace_variables_with_parameters(template_c["func"], params),
        path2template=Path(design_plugin.__file__).parent / "c",
    )

def _generate_thresholding_constant_template() -> dict[str, list[str]]:
    header_template = [
        "// --- Generationg thresholding_constant",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: ID = {$device_id}, type = {$data_type}, threshold = {$threshold}",
        '#include "{$path2include}/{$template_name}"',
        "DEF_CONSTANT_THR_PROTO({$device_id}, {$data_type})",
        ]
    implementation_template = [
        "// --- Generating thresholding_constant",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: ID = {$device_id}, type = {$data_type}, threshold = {$threshold}",
        '#include "{$path2include}/{$template_name}"',
        "DEF_CONSTANT_THR_IMPL({$device_id}, {$data_type}, {$threshold})",
    ]
    return {"head": header_template, "func": implementation_template}
