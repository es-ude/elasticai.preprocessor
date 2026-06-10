from datetime import datetime
from os.path import abspath, dirname

import elasticai.creator_plugins.plugins_c.filter_data as design_plugin
from elasticai.preprocessor.filter import Filtering, SettingsFilter
from elasticai.preprocessor.translation.ir2c import (
    generate_c_files,
    get_embedded_datatype,
    replace_variables_with_parameters,
)


def build_filter_fir(
    settings: SettingsFilter,
    bitwidth: int,
    signed: bool,
    do_optimized: bool,
    filter_id: str = "0",
    path2save: str = "",
    define_path: str = "src",
) -> None:
    """Generating C files of a FIR (Finite Impuls Response) filter for using on microcontrollers
    Args:
        settings:       Settings filter
        bitwidth:       Used quantization level for data stream
        signed:         Decision if LUT values are signed [otherwise unsigned]
        filter_id:      ID of used filter structure
        do_optimized:   Decision if LUT resources should be minimized [only quarter and mirroring]
        path2save:      Path for saving the verilog_filter output files
        define_path:    Path for loading the header file in IDE [Default: 'src']
    Return:
        None
    """
    assert settings.type.lower() == "fir", (
        f"Key 'type' must be 'fir' and not '{settings.type.lower()}'"
    )
    assert bitwidth in range(2, 32), "Bitwidth must be between 2 and 32"
    coeff_b = Filtering(setting=settings).get_coeffs().b
    if do_optimized and len(coeff_b) % 2 == 0:
        raise NotImplementedError("Please add an odd number to filter order!")

    module_id = f"{settings.b_type.lower().split('pass')[0]}{filter_id.lower()}"
    coeff_used = coeff_b if not do_optimized else coeff_b[: int(len(coeff_b) / 2) + 1]
    data_type_filter = get_embedded_datatype(bitwidth, signed)
    params = {
        "datetime_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
        "path2include": define_path,
        "template_name": "filter_fir_template.h",
        "device_id": module_id.upper(),
        "data_type": data_type_filter,
        "fs": f"{settings.fs}",
        "filter_type": f"{settings.b_type}, {settings.f_type}",
        "filter_corner": ", ".join(map(str, settings.f_filt)),
        "filter_order": str(len(coeff_b)),
        "coeffs_string": ", ".join(map(str, coeff_used)),
    }

    template_c = __generate_filter_fir_template(do_optimized)
    generate_c_files(
        path2save=path2save,
        template_name=params["template_name"],
        file_name="filter_fir",
        module_id=module_id,
        proto_file=replace_variables_with_parameters(template_c["head"], params),
        impl_file=replace_variables_with_parameters(template_c["func"], params),
        path2template=dirname(abspath(design_plugin.__file__)),
    )


def __generate_filter_fir_template(do_opt: bool) -> dict:
    """Generate the template for writing *.c and *.h file for generate a FIR filter on MCUs
    Args:
        do_opt:     Boolean decision if optimized version is used (odd version)
    Return:
        Dictionary with infos for prototype ['head'], implementation ['func'] and used parameters ['params']
    """
    version_fir = "full" if not do_opt else "opt"
    header_temp = [
        f"// --- Generating a FIR filter template ({version_fir})",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: N = {$filter_order}, f_c = [{$filter_corner}] Hz @ {$fs} Hz ({$filter_type})",
        "// Used filter coefficient order (b_0, b_1, b_2, ..., b_N)",
        '# include "{$path2include}/{$template_name}"',
        "DEF_NEW_FIR_FILTER_PROTO({$device_id}, {$data_type})",
    ]
    func_temp = [
        f"// --- Generating a FIR filter template ({version_fir})",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: N = {$filter_order}, f_c = [{$filter_corner}] Hz @ {$fs} Hz ({$filter_type})",
        "// Used filter coefficient order (b_0, b_1, b_2, ..., b_N)",
        '# include "{$path2include}/{$template_name}"',
        "DEF_NEW_FIR_FILTER_IMPL({$device_id}, {$data_type}, {$filter_order}, {$coeffs_string})",
    ]
    return {"head": header_temp, "func": func_temp, "params": []}
