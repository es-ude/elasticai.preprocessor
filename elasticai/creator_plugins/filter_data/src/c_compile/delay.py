from datetime import datetime
from pathlib import Path

import elasticai.creator_plugins.filter_data as design_plugin
from elasticai.preprocessor import get_path_to_project
from elasticai.preprocessor.filter import SettingsFilter
from elasticai.preprocessor.translation.ir2c import (
    generate_c_files,
    get_embedded_datatype,
    replace_variables_with_parameters,
)


def build_filter_delay(
    settings: SettingsFilter,
    bitwidth: int,
    signed: bool,
    filter_id: str = "0",
    path2save: Path = get_path_to_project("build"),
    define_path: str = "src",
) -> None:
    """Generating C files for Delay Line filtering on microcontroller
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
    assert bitwidth in range(2, 33), "Bitwidth must be between 2 and 32"
    assert settings.b_type.lower() == "allpass"
    assert settings.type.lower() == "fir", f"Key 'type' must be 'fir' and not '{settings.type.lower()}'"

    module_id = f"{filter_id.lower()}"
    data_type_filter = get_embedded_datatype(bitwidth, signed)
    filter_order = settings._num_delay_taps
    params = {
        "datetime_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
        "path2include": define_path,
        "template_name": "filter_fir_delay_template.h",
        "device_id": module_id.lower(),
        "data_type": data_type_filter,
        "fs": f"{settings.fs}",
        "t_dly_us": str(filter_order / settings.fs * 1e6),
        "filter_order": str(filter_order),
    }

    template_c = __generate_filter_delay_template()
    generate_c_files(
        path2save=path2save,
        template_name=params["template_name"],
        file_name="filter_fir_delay",
        module_id=module_id,
        proto_file=replace_variables_with_parameters(template_c["head"], params),
        impl_file=replace_variables_with_parameters(template_c["func"], params),
        path2template=Path(design_plugin.__file__).parent / "c",
    )


def __generate_filter_delay_template() -> dict:
    """Generate the template for writing *.c and *.h file for generate a FIR filter on MCUs
    Return:
        Dictionary with infos for prototype ['head'], implementation ['func'] and used parameters ['params']
    """
    header_temp = [
        "// --- Generating a FIR-Delay filter template",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: N = {$filter_order}, t_dly = {$t_dly_us} us @ {$fs} Hz",
        '# include "{$path2include}/{$template_name}"',
        "DEF_NEW_FIR_DELAY_PROTO({$device_id}, {$data_type})",
    ]
    func_temp = [
        "// --- Generating a FIR-Delay filter template",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: N = {$filter_order}, t_dly = {$t_dly_us} us @ {$fs} Hz",
        '# include "{$path2include}/{$template_name}"',
        "DEF_NEW_FIR_DELAY_IMPL({$device_id}, {$data_type}, {$filter_order})",
    ]
    return {"head": header_temp, "func": func_temp, "params": []}
