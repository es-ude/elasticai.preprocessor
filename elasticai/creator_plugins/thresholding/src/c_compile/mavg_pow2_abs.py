from datetime import datetime
from pathlib import Path

import elasticai.creator_plugins.thresholding as design_plugin
from elasticai.preprocessor.translation.ir2c import (
    generate_c_files,
    get_embedded_datatype,
    replace_variables_with_parameters,
)


def build_thresholding_mavg_pow2_abs(
    log_size: int,
    window_size: int,
    bitwidth: int,
    signed: bool,
    path2save: Path,
    thresholding_id: str = "0",
    define_path: str = "src",
) -> None:
    """Generate C files for power-of-2 moving absolute average thresholding.
    Args:
        log_size:        number of bits for bitshift
        window_size:     number of samples (must be a power of 2)
        bitwidth:        bitwidth of each sample
        signed:          Decision if data values are signed [otherwise unsigned]
        path2save:       Path to save the .h/.c output-files
        thresholding_id: ID appended to function names
        define_path:     Include path written into the generated #include line.
    """
    assert bitwidth in range(2, 33), "bitwidth must be between 2 and 32."
    assert window_size > 0 and (window_size & (window_size - 1)) == 0, \
        "window_size must be a power of 2."

    module_id = thresholding_id.lower()
    params = {
        "logsize":   str(log_size),
        "size":      str(window_size),
        "datetime_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
        "path2include": define_path,
        "template_name": "thresholding_mavg_pow2_abs_template.h",
        "module_id": module_id,
        "device_id": module_id.upper(),
        "data_type": get_embedded_datatype(bitwidth, signed),
    }
    template_c = _generate_thresholding_mavg_pow2_abs_template()
    generate_c_files(
        path2save=path2save,
        template_name=params["template_name"],
        file_name="thresholding_mavg_pow2_abs",
        module_id=module_id,
        proto_file=replace_variables_with_parameters(template_c["head"], params),
        impl_file=replace_variables_with_parameters(template_c["func"], params),
        path2template=Path(design_plugin.__file__).parent / "c",
    )


def _generate_thresholding_mavg_pow2_abs_template() -> dict[str, list[str]]:
    header_template = [
        "// --- Generating thresholding_mavg_pow2_abs",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: ID = {$device_id}, type = {$data_type}",
        '#include "{$path2include}/{$template_name}"',
        "DEF_NEW_MAVG_POW2_ABS_WINDOW_PROTO({$device_id}, {$data_type})",
    ]
    implementation_template = [
        "// --- Generating thresholding_mavg_pow2_abs",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: ID = {$device_id}, type = {$data_type},",
        "// window_size = {$size}, log_size = {$logsize}",
        '#include "{$path2include}/{$template_name}"',
        "DEF_CALC_MAVG_POW2_ABS_THR({$device_id}, {$data_type})",
        "DEF_NEW_MAVG_POW2_ABS_WINDOW_IMPL({$device_id}, {$data_type}, {$size}, {$logsize})",
    ]
    return {"head": header_template, "func": implementation_template}
