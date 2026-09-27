from datetime import datetime
from pathlib import Path

import elasticai.creator_plugins.windower as design_plugin
from elasticai.preprocessor.translation.ir2c import (
    generate_c_files,
    get_embedded_datatype,
    replace_variables_with_parameters,
)
from elasticai.preprocessor.windower.window import SettingsWindow


def build_windower(
    settings: SettingsWindow,
    bitwidth: int,
    signed: bool,
    path2save: Path,
    windower_id: str = "0",
    define_path: str = "src",
) -> None:
    """Generate C files for a sliding-window windower (streaming interface).
    Args:
        settings:       Window settings (sampling_rate, window_sec, overlap_sec)
        bitwidth:       Bit width of each sample
        signed:         Whether the data type is signed
        path2save:      Path to save the .h / .c output files
        windower_id:    ID appended to the generated function name
        define_path:    Include path written into the generated #include line
    """
    assert bitwidth in range(2, 33), "bitwidth must be between 2 and 32"

    window_length = settings.window_length
    num_shift = window_length - settings.overlap_length

    module_id = windower_id.lower()
    params = {
        "datetime_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
        "path2include": define_path,
        "template_name": "windower_template.h",
        "device_id": module_id.upper(),
        "data_type": get_embedded_datatype(bitwidth, signed),
        "window_length": str(window_length),
        "num_shift": str(num_shift),
    }
    template_c = _generate_windower_template()
    generate_c_files(
        path2save=path2save,
        template_name=params["template_name"],
        file_name="windower",
        module_id=module_id,
        proto_file=replace_variables_with_parameters(template_c["head"], params),
        impl_file=replace_variables_with_parameters(template_c["func"], params),
        path2template=Path(design_plugin.__file__).parent / "c",
    )


def _generate_windower_template() -> dict[str, list[str]]:
    header_template = [
        "// --- Generating windower",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: ID = {$device_id}, type = {$data_type}, window = {$window_length}, shift = {$num_shift}",
        '#include "{$path2include}/{$template_name}"',
        "DEF_WINDOWER_PROTO({$device_id}, {$data_type})",
    ]
    implementation_template = [
        "// --- Generating windower",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: ID = {$device_id}, type = {$data_type}, window = {$window_length}, shift = {$num_shift}",
        '#include "{$path2include}/{$template_name}"',
        "DEF_WINDOWER_IMPL({$device_id}, {$data_type}, {$window_length}, {$num_shift})",
    ]
    return {"head": header_template, "func": implementation_template}
