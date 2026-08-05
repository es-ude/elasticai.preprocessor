from datetime import datetime
from pathlib import Path

import numpy as np

import elasticai.creator_plugins.downsampling as design_plugin
from elasticai.preprocessor.translation.ir2c import (
    generate_c_files,
    get_embedded_datatype,
    replace_variables_with_parameters,
)


def build_downsampling_polyphase(
    downsampling_ratio: int,
    take_first_order: bool,
    bitwidth: int,
    signed: bool,
    path2save: Path,
    downsampling_id: str = "0",
    define_path: str = "src",
) -> None:
    """Generate C files for polyphase downsampling.
    Args:
        downsampling_ratio: Decimation factor (SR_out = SR_in / downsampling_ratio)
        take_first_order:   true: order_one; false: order_two
        bitwidth:           Bit width of each sample
        signed:             Whether values are signed
        path2save:          Directory to write the generated .h / .c files into
        downsampling_id:    ID appended to function names
        define_path:        Include path written into the generated #include line
    """
    if downsampling_ratio < 1:
        raise ValueError("dsr must be >= 1")
    if not np.log2(downsampling_ratio).is_integer():
        raise ValueError("dsr must be 2^n")
    assert bitwidth in range(2, 33), "Bitwidth must be between 2 and 32"

    module_id = downsampling_id.lower()
    params = {
        "datetime_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
        "path2include": define_path,
        "template_name": "downsampling_poly_template.h",
        "device_id": module_id.upper(),
        "data_type": get_embedded_datatype(bitwidth, signed),
        "downsampling_ratio": str(downsampling_ratio),
    }
    if take_first_order:
        template_c = _generate_downsampling_poly_one_template()
        generate_c_files(
            path2save=path2save,
            template_name=params["template_name"],
            file_name="downsampling_poly_one",
            module_id=module_id,
            proto_file=replace_variables_with_parameters(template_c["head"], params),
            impl_file=replace_variables_with_parameters(template_c["func"], params),
            path2template=Path(design_plugin.__file__).parent / "c",
        )
    else:
        template_c = _generate_downsampling_poly_two_template()
        generate_c_files(
            path2save=path2save,
            template_name=params["template_name"],
            file_name="downsampling_poly_two",
            module_id=module_id,
            proto_file=replace_variables_with_parameters(template_c["head"], params),
            impl_file=replace_variables_with_parameters(template_c["func"], params),
            path2template=Path(design_plugin.__file__).parent / "c",
        )


def _generate_downsampling_poly_one_template() -> dict[str, list[str]]:
    header_template = [
        "// --- Generating do_poly_one",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: ID = {$device_id}, type = {$data_type}, dsr = {$downsampling_ratio}",
        '#include "{$path2include}/{$template_name}"',
        "DEF_DOWNSAMPLING_POLY_ONE_PROTO({$device_id}, {$data_type})",
    ]
    implementation_template = [
        "// --- Generating do_poly_one",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: ID = {$device_id}, type = {$data_type}, dsr = {$downsampling_ratio}",
        '#include "{$path2include}/{$template_name}"',
        "DEF_DOWNSAMPLING_POLY_ONE_IMPL({$device_id}, {$data_type}, {$downsampling_ratio})",
    ]
    return {"head": header_template, "func": implementation_template}


def _generate_downsampling_poly_two_template() -> dict[str, list[str]]:
    header_template = [
        "// --- Generating do_poly_two",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: ID = {$device_id}, type = {$data_type}, dsr = {$downsampling_ratio}",
        '#include "{$path2include}/{$template_name}"',
        "DEF_DOWNSAMPLING_POLY_TWO_PROTO({$device_id}, {$data_type})",
    ]
    implementation_template = [
        "// --- Generating do_poly_two",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: ID = {$device_id}, type = {$data_type}, dsr = {$downsampling_ratio}",
        '#include "{$path2include}/{$template_name}"',
        "DEF_DOWNSAMPLING_POLY_TWO_IMPL({$device_id}, {$data_type}, {$downsampling_ratio})",
    ]
    return {"head": header_template, "func": implementation_template}
