from datetime import datetime
from pathlib import Path

import elasticai.creator_plugins.sda as design_plugin
from elasticai.preprocessor.translation.ir2c import (
    generate_c_files,
    get_embedded_datatype,
    replace_variables_with_parameters,
)


def build_sda_normal_const(
    threshold: int,
    bitwidth: int,
    signed: bool,
    path2save: Path,
    sda_id: str = "0",
    define_path: str = "src",
) -> None:
    """Generate C files for SDA with Normal preprocessor and constant threshold.
    :param threshold:   Constant threshold value (integer, already quantized).
    :param bitwidth:    Bitwidth of each sample.
    :param signed:      True if the data type is signed.
    :param path2save:   Path to save the generated .h/.c files.
    :param sda_id:      ID appended to function names (e.g. "0" → calc_sda_0).
    :param define_path: Include path written into the generated #include line.
    """
    module_id = sda_id.lower()
    params = {
        "datetime_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
        "path2include": define_path,
        "template_name": "sda_normal_const_template.h",
        "module_id": module_id,
        "device_id": module_id.upper(),
        "data_type": get_embedded_datatype(bitwidth, signed),
        "threshold": str(threshold),
    }
    template_c = _generate_sda_normal_const_template()
    generate_c_files(
        path2save=path2save,
        template_name=params["template_name"],
        file_name="sda_normal_const",
        module_id=module_id,
        proto_file=replace_variables_with_parameters(template_c["head"], params),
        impl_file=replace_variables_with_parameters(template_c["func"], params),
        path2template=Path(design_plugin.__file__).parent / "c",
    )


def _generate_sda_normal_const_template() -> dict[str, list[str]]:
    header_template = [
        "// --- Generating SDA Normal+Constant",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: ID = {$device_id}, type = {$data_type}, threshold = {$threshold}",
        '#include "{$path2include}/{$template_name}"',
        "DEF_NEW_SDA_NORMAL_CONST_PROTO({$device_id}, {$data_type})",
    ]
    implementation_template = [
        "// --- Generating SDA Normal+Constant",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: ID = {$device_id}, type = {$data_type}, threshold = {$threshold}",
        '#include "{$path2include}/sda_normal_const_{$module_id}.h"',
        "DEF_NEW_SDA_NORMAL_CONST_IMPL({$device_id}, {$data_type}, {$threshold})",
    ]
    return {"head": header_template, "func": implementation_template}


def build_sda_absolute_const(
    threshold: int,
    bitwidth: int,
    signed: bool,
    path2save: Path,
    sda_id: str = "0",
    define_path: str = "src",
) -> None:
    """Generate C files for SDA with Absolute preprocessor and constant threshold.
    :param threshold:   Constant threshold value (integer, already quantized).
    :param bitwidth:    Bitwidth of each sample.
    :param signed:      True if the data type is signed.
    :param path2save:   Path to save the generated .h/.c files.
    :param sda_id:      ID appended to function names (e.g. "0" → calc_sda_0).
    :param define_path: Include path written into the generated #include line.
    """
    module_id = sda_id.lower()
    params = {
        "datetime_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
        "path2include": define_path,
        "template_name": "sda_absolute_const_template.h",
        "module_id": module_id,
        "device_id": module_id.upper(),
        "data_type": get_embedded_datatype(bitwidth, signed),
        "threshold": str(threshold),
    }
    template_c = _generate_sda_absolute_const_template()
    generate_c_files(
        path2save=path2save,
        template_name=params["template_name"],
        file_name="sda_absolute_const",
        module_id=module_id,
        proto_file=replace_variables_with_parameters(template_c["head"], params),
        impl_file=replace_variables_with_parameters(template_c["func"], params),
        path2template=Path(design_plugin.__file__).parent / "c",
    )


def _generate_sda_absolute_const_template() -> dict[str, list[str]]:
    header_template = [
        "// --- Generating SDA Absolute+Constant",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: ID = {$device_id}, type = {$data_type}, threshold = {$threshold}",
        '#include "{$path2include}/{$template_name}"',
        "DEF_NEW_SDA_ABSOLUTE_CONST_PROTO({$device_id}, {$data_type})",
    ]
    implementation_template = [
        "// --- Generating SDA Absolute+Constant",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: ID = {$device_id}, type = {$data_type}, threshold = {$threshold}",
        '#include "{$path2include}/sda_absolute_const_{$module_id}.h"',
        "DEF_NEW_SDA_ABSOLUTE_CONST_IMPL({$device_id}, {$data_type}, {$threshold})",
    ]
    return {"head": header_template, "func": implementation_template}
