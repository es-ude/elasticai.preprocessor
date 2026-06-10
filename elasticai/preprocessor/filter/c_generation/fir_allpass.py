from datetime import datetime
from os.path import abspath, dirname

import elasticai.creator_plugins.plugins_c.filter_data as design_plugin
from elasticai.preprocessor import get_path_to_project
from elasticai.preprocessor.filter import SettingsFilter
from elasticai.preprocessor.translation.ir2c import (
    generate_c_files,
    get_embedded_datatype,
    replace_variables_with_parameters,
)


def build_filter_fir_allpass(
    settings: SettingsFilter,
    bitwidth: int,
    signed: bool,
    filter_id: str = "0",
    path2save: str = get_path_to_project("build"),
    define_path: str = "src",
) -> None:
    """Generating C files for IIR filtering on microcontroller
    Args:
        settings:       Settings filter
        bitwidth:       Used quantization level for data stream
        signed:         Decision if LUT values are signed [otherwise unsigned]
        filter_id:      ID of used filter structure
        path2save:      Path for saving the verilog_filter output files
        define_path:    Path for loading the header file in IDE [Default: 'src']
    Return:
        None
    """
    assert bitwidth in range(2, 32), "Bitwidth must be between 2 and 32"
    assert settings.b_type.lower() == "allpass"
    assert settings.type.lower() == "fir", (
        f"Key 'type' must be 'fir' and not '{settings.type.lower()}'"
    )

    module_id = f"{settings.b_type.lower().split('pass')[0]}{filter_id.lower()}"
    data_type_filter = get_embedded_datatype(bitwidth, signed)
    filter_order = int(settings.fs / settings.f_filt[0])
    params = {
        "datetime_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
        "path2include": define_path,
        "template_name": "filter_fir_all_template.h",
        "device_id": module_id.upper(),
        "data_type": data_type_filter,
        "fs": f"{settings.fs}",
        "t_dly": str(filter_order / settings.fs * 1e6),
        "filter_order": str(filter_order),
    }

    template_c = __generate_filter_fir_allpass_template()
    generate_c_files(
        path2save=path2save,
        template_name=params["template_name"],
        file_name="filter_fir",
        module_id=module_id,
        proto_file=replace_variables_with_parameters(template_c["head"], params),
        impl_file=replace_variables_with_parameters(template_c["func"], params),
        path2template=dirname(abspath(design_plugin.__file__)),
    )


def __generate_filter_fir_allpass_template() -> dict:
    """Generate the template for writing *.c and *.h file for generate a FIR filter on MCUs
    Return:
        Dictionary with infos for prototype ['head'], implementation ['func'] and used parameters ['params']
    """
    header_temp = [
        "// --- Generating a FIR-Allpass filter template",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: N = {$filter_order}, t_dly = {$t_dly} us @ {$fs} Hz",
        '# include "{$path2include}/{$template_name}"',
        "DEF_NEW_FIR_ALL_FILTER_PROTO({$device_id}, {$data_type})",
    ]
    func_temp = [
        "// --- Generating a FIR-Allpass filter template",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: N = {$filter_order}, t_dly = {$t_dly} us @ {$fs} Hz",
        '# include "{$path2include}/{$template_name}"',
        "DEF_NEW_FIR_ALL_FILTER_IMPL({$device_id}, {$data_type}, {$filter_order})",
    ]
    return {"head": header_temp, "func": func_temp, "params": []}
