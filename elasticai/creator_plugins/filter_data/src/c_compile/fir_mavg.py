from datetime import datetime
from pathlib import Path

import elasticai.creator_plugins.filter_data as design_plugin
from elasticai.preprocessor.translation.ir2c import (
    generate_c_files,
    get_embedded_datatype,
    replace_variables_with_parameters,
)


def build_filter_mavg(
    order: int,
    bitwidth: int,
    signed: bool,
    path2save: Path,
    filter_id: str = "0",
    define_path: str = "src",
) -> None:
    """Generating C files of a moving-average filter for using on microcontrollers
    Args:
        order:          Number of taps (window size); coefficient = 1/order
        bitwidth:       Used quantization level for data stream
        signed:         Decision if values are signed [otherwise unsigned]
        path2save:      Path for saving the output files
        filter_id:      ID of used filter structure
        define_path:    Path for loading the header file in IDE [Default: 'src']
    Return:
        None
    """
    assert bitwidth in range(2, 33), "Bitwidth must be between 2 and 32"
    assert order >= 1, "order must be >= 1"

    data_type = get_embedded_datatype(bitwidth, signed)
    coeff = 1.0 / order
    params = {
        "datetime_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
        "path2include": define_path,
        "template_name": "filter_mavg_template.h",
        "device_id": f"MAVG{filter_id.upper()}",
        "data_type": data_type,
        "filter_order": str(order),
        "coeff": str(coeff),
    }

    template_c = __generate_filter_mavg_template()
    generate_c_files(
        path2save=path2save,
        template_name=params["template_name"],
        file_name="filter_mavg",
        module_id=filter_id.lower(),
        proto_file=replace_variables_with_parameters(template_c["head"], params),
        impl_file=replace_variables_with_parameters(template_c["func"], params),
        path2template=Path(design_plugin.__file__).parent / "c",
    )


def __generate_filter_mavg_template() -> dict:
    header_temp = [
        "// --- Generating a moving-average filter template",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: order = {$filter_order}, coeff = {$coeff}",
        '# include "{$path2include}/{$template_name}"',
        "DEF_NEW_MAVG_FILTER_PROTO({$device_id}, {$data_type})",
    ]
    func_temp = [
        "// --- Generating a moving-average filter template",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: order = {$filter_order}, coeff = {$coeff}",
        '# include "{$path2include}/{$template_name}"',
        "DEF_NEW_MAVG_FILTER_IMPL({$device_id}, {$data_type}, {$filter_order}, {$coeff})",
    ]
    return {"head": header_temp, "func": func_temp}
