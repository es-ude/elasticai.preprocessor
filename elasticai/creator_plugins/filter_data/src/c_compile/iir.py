from datetime import datetime
from os.path import abspath, dirname

import elasticai.creator_plugins.plugins_c.filter_data as design_plugin
from elasticai.preprocessor import get_path_to_project
from elasticai.preprocessor.filter import Filtering, SettingsFilter
from elasticai.preprocessor.translation.ir2c import (
    generate_c_files,
    get_embedded_datatype,
    replace_variables_with_parameters,
)


def build_filter_iir(
    settings: SettingsFilter,
    bitwidth: int,
    signed: bool,
    filter_id: str = "",
    path2save: str = get_path_to_project("build"),
    define_path: str = "src",
) -> None:
    """Generating C files for IIR filtering on microcontroller
    Args:
        settings:       Settings filter
        bitwidth:       Used quantization level for data stream
        signed:         Decision if LUT values are signed [otherwise unsigned]
        filter_id:      ID of used filter
        path2save:      Path for saving the verilog_filter output files
        define_path:    Path for loading the header file in IDE [Default: 'src']
    Return:
        None
    """
    assert settings.type.lower() == "iir", (
        f"Key 'type' must be 'iir' and not '{settings.type.lower()}'"
    )
    assert bitwidth in range(2, 32), "Bitwidth must be between 2 and 32"

    coeff = Filtering(setting=settings).get_coeffs()
    module_id_used = f"{settings.b_type.lower().split('pass')[0]}{filter_id.lower()}"
    data_type_filter = get_embedded_datatype(bitwidth, signed)
    params = {
        "datetime_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
        "path2include": define_path,
        "template_name": "filter_iir_template.h",
        "device_id": module_id_used.upper(),
        "data_type": data_type_filter,
        "fs": f"{settings.fs}",
        "filter_type": f"{settings.b_type}, {settings.f_type}",
        "filter_corner": ", ".join(map(str, settings.f_filt)),
        "filter_order": str(settings.n_order),
        "coeff_order": str(len(coeff.a)),
        "tap_order": str(len(coeff.b) - 1),
        "coeffs_string": ", ".join(map(str, coeff.a))
        + ", "
        + ", ".join(map(str, coeff.b)),
    }

    template_c = __generate_filter_iir_template()
    generate_c_files(
        path2save=path2save,
        template_name=params["template_name"],
        file_name="filter_iir",
        module_id=module_id_used,
        proto_file=replace_variables_with_parameters(template_c["head"], params),
        impl_file=replace_variables_with_parameters(template_c["func"], params),
        path2template=dirname(abspath(design_plugin.__file__)),
    )


def __generate_filter_iir_template() -> dict:
    """Generate the template for writing *.c and *.h file for generate an IIR filter on MCUs
    Return:
        Dictionary with infos for prototype ['head'], implementation ['func'] and used parameters ['params']
    """
    header_temp = [
        "// --- Generating an IIR filter template (Direct Form II)",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: N = {$filter_order}, f_c = [{$filter_corner}] Hz @ {$fs} Hz ({$filter_type})",
        "// Used filter coefficient order (a_0, a_1, ... a_N, b_0, b_1, ..., b_N)",
        '# include "{$path2include}/{$template_name}"',
        "DEF_NEW_IIR_FILTER_PROTO({$device_id}, {$data_type})",
    ]
    func_temp = [
        "// --- Generating an IIR filter template (Direct Form II)",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: N = {$filter_order}, f_c = [{$filter_corner}] Hz @ {$fs} Hz ({$filter_type})",
        "// Used filter coefficient order (a_0, a_1, ... a_N, b_0, b_1, ..., b_N)",
        '# include "{$path2include}/{$template_name}"',
        "DEF_NEW_IIR_FILTER_IMPL({$device_id}, {$data_type}, {$coeff_order}, {$tap_order}, {$coeffs_string})",
    ]
    return {"head": header_temp, "func": func_temp, "params": []}
