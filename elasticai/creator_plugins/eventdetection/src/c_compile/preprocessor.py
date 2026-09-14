from datetime import datetime
from pathlib import Path

import elasticai.creator_plugins.eventdetection as design_plugin
from elasticai.preprocessor.translation.ir2c import (
    generate_c_files,
    get_embedded_datatype,
    replace_variables_with_parameters,
)


def build_preprocessor_normal(
    bitwidth: int,
    signed: bool,
    path2save: Path,
    preprocessor_id: str,
    define_path: str,
) -> None:
    """Generate C files for SDA with normal preprocessor.
    :param bitwidth:        Bitwidth of each sample.
    :param signed:          True if the data type is signed.
    :param path2save:       Path to save the generated .h/.c files.
    :param preprocessor_id: ID appended to function names (e.g. "0" → calc_preprocessing_0).
    :param define_path:     Include path written into the generated #include line.
    """
    module_id = preprocessor_id.lower()
    params = {
        "datetime_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
        "path2include": define_path,
        "template_name": "preprocessing_normal_template.h",
        "module_id": module_id,
        "device_id": module_id.upper(),
        "data_type": get_embedded_datatype(bitwidth, signed),
    }
    template_c = _generate_preprocessing_normal_template()
    generate_c_files(
        path2save=path2save,
        template_name=params["template_name"],
        file_name="preprocessing_normal",
        module_id=module_id,
        proto_file=replace_variables_with_parameters(template_c["head"], params),
        impl_file=replace_variables_with_parameters(template_c["func"], params),
        path2template=Path(design_plugin.__file__).parent / "c",
    )


def _generate_preprocessing_normal_template() -> dict[str, list[str]]:
    header_template = [
        "// --- Generating SDA Preprocessor normal",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: ID = {$device_id}, type = {$data_type}",
        '#include "{$path2include}/{$template_name}"',
        "DEF_NEW_PREPROCESSING_NORMAL_PROTO({$device_id}, {$data_type})",
    ]
    implementation_template = [
        "// --- Generating SDA Preprocessor normal",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: ID = {$device_id}, type = {$data_type}",
        '#include "{$path2include}/preprocessing_normal_{$module_id}.h"',
        "DEF_NEW_PREPROCESSING_NORMAL_IMPL({$device_id}, {$data_type})",
    ]
    return {"head": header_template, "func": implementation_template}


def build_preprocessor_abs(
    bitwidth: int,
    signed: bool,
    path2save: Path,
    preprocessor_id: str,
    define_path: str,
) -> None:
    """Generate C files for SDA with absolute preprocessor.
    :param bitwidth:        Bitwidth of each sample.
    :param signed:          True if the data type is signed.
    :param path2save:       Path to save the generated .h/.c files.
    :param preprocessor_id: ID appended to function names (e.g. "0" → calc_preprocessing_0).
    :param define_path:     Include path written into the generated #include line.
    """
    module_id = preprocessor_id.lower()
    params = {
        "datetime_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
        "path2include": define_path,
        "template_name": "preprocessing_abs_template.h",
        "module_id": module_id,
        "device_id": module_id.upper(),
        "data_type": get_embedded_datatype(bitwidth, signed),
    }
    template_c = _generate_preprocessing_abs_template()
    generate_c_files(
        path2save=path2save,
        template_name=params["template_name"],
        file_name="preprocessing_abs",
        module_id=module_id,
        proto_file=replace_variables_with_parameters(template_c["head"], params),
        impl_file=replace_variables_with_parameters(template_c["func"], params),
        path2template=Path(design_plugin.__file__).parent / "c",
    )


def _generate_preprocessing_abs_template() -> dict[str, list[str]]:
    header_template = [
        "// --- Generating SDA Preprocessor absolute",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: ID = {$device_id}, type = {$data_type}",
        '#include "{$path2include}/{$template_name}"',
        "DEF_NEW_PREPROCESSING_ABS_PROTO({$device_id}, {$data_type})",
    ]
    implementation_template = [
        "// --- Generating SDA Preprocessor absolute",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: ID = {$device_id}, type = {$data_type}",
        '#include "{$path2include}/preprocessing_abs_{$module_id}.h"',
        "DEF_NEW_PREPROCESSING_ABS_IMPL({$device_id}, {$data_type})",
    ]
    return {"head": header_template, "func": implementation_template}


def build_preprocessor_neo(
    bitwidth: int,
    signed: bool,
    path2save: Path,
    preprocessor_id: str,
    define_path: str,
) -> None:
    """Generate C files for SDA with NEO preprocessor.
    :param bitwidth:        Bitwidth of each sample.
    :param signed:          True if the data type is signed.
    :param path2save:       Path to save the generated .h/.c files.
    :param preprocessor_id: ID appended to function names (e.g. "0" → calc_preprocessing_0).
    :param define_path:     Include path written into the generated #include line.
    """
    module_id = preprocessor_id.lower()
    params = {
        "datetime_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
        "path2include": define_path,
        "template_name": "preprocessing_neo_template.h",
        "module_id": module_id,
        "device_id": module_id.upper(),
        "data_type": get_embedded_datatype(bitwidth, signed),
    }
    template_c = _generate_preprocessing_neo_template()
    generate_c_files(
        path2save=path2save,
        template_name=params["template_name"],
        file_name="preprocessing_neo",
        module_id=module_id,
        proto_file=replace_variables_with_parameters(template_c["head"], params),
        impl_file=replace_variables_with_parameters(template_c["func"], params),
        path2template=Path(design_plugin.__file__).parent / "c",
    )


def _generate_preprocessing_neo_template() -> dict[str, list[str]]:
    header_template = [
        "// --- Generating SDA Preprocessor NEO",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: ID = {$device_id}, type = {$data_type}",
        '#include "{$path2include}/{$template_name}"',
        "DEF_NEW_PREPROCESSING_NEO_PROTO({$device_id}, {$data_type})",
    ]
    implementation_template = [
        "// --- Generating SDA Preprocessor NEO",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: ID = {$device_id}, type = {$data_type}",
        '#include "{$path2include}/preprocessing_neo_{$module_id}.h"',
        "DEF_NEW_PREPROCESSING_NEO_IMPL({$device_id}, {$data_type})",
    ]
    return {"head": header_template, "func": implementation_template}
