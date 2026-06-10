from datetime import datetime
from os.path import abspath, dirname

import elasticai.creator_plugins.plugins_c.filter_data as design_plugin
from elasticai.preprocessor import get_path_to_project
from elasticai.preprocessor.translation.ir2c import (
    generate_c_files,
    get_embedded_datatype,
    replace_variables_with_parameters,
)


def build_filter_moving_average(
    bitwidth: int,
    signed: bool,
    filter_order: int,
    sampling_rate: float,
    module_id: str = "0",
    path2save: str = get_path_to_project("build"),
    define_path: str = "src",
) -> None:
    """Generating C files for moving average on microcontroller
    Args:
        bitwidth:       Used quantization level for data stream
        signed:         Decision if LUT values are signed [otherwise unsigned]
        module_id:      ID of used filter structure
        filter_order:   Order of the filter
        sampling_rate:  Sampling clock of data stream processing
        path2save:      Path for saving the verilog_filter output files
        define_path:    Path for loading the header file in IDE [Default: 'src']
    Return:
        None
    """
    assert bitwidth in range(2, 32), "Bitwidth must be between 2 and 32"

    module_id_used = f"{module_id.lower()}"
    data_type_filter = get_embedded_datatype(bitwidth, signed)
    params = {
        "datetime_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
        "path2include": define_path,
        "template_name": "filter_mavg_template.h",
        "device_id": module_id_used.upper(),
        "data_type": data_type_filter,
        "fs": f"{sampling_rate}",
        "filter_order": str(filter_order),
        "filter_coeff": str(1 / filter_order),
    }

    template_c = __generate_filter_mavg_template()
    generate_c_files(
        path2save=path2save,
        template_name=params["template_name"],
        file_name="filter_mavg",
        module_id=module_id_used,
        proto_file=replace_variables_with_parameters(template_c["head"], params),
        impl_file=replace_variables_with_parameters(template_c["func"], params),
        path2template=dirname(abspath(design_plugin.__file__)),
    )


def __generate_filter_mavg_template() -> dict:
    """Generate the template for writing *.c and *.h file for generate a moving average FIR on MCUs
    Args:
        None
    Return:
        Dictionary with infos for prototype ['head'], implementation ['func'] and used parameters ['params']
    """
    header_temp = [
        "// --- Generating a moving average",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: N = {$filter_order}, f_s = {$fs} Hz",
        '# include "{$path2include}/{$template_name}"',
        "DEF_NEW_MAVG_FILTER_PROTO({$device_id}, {$data_type})",
    ]
    func_temp = [
        "// --- Generating a moving average filter",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: N = {$filter_order}, f_s = {$fs} Hz",
        '# include "{$path2include}/{$template_name}"',
        "DEF_NEW_MAVG_FILTER_IMPL({$device_id}, {$data_type}, {$filter_order}, {$filter_coeff})",
    ]
    return {"head": header_temp, "func": func_temp, "params": []}
