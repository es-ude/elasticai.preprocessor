from datetime import datetime
from pathlib import Path

import elasticai.creator_plugins.player as design_plugin
from elasticai.preprocessor.translation.ir2c import (
    generate_c_files,
    get_embedded_datatype,
    replace_variables_with_parameters,
)


def build_replayer(
    data: list[int],
    bitwidth: int,
    signed: bool,
    path2save: Path,
    replayer_id: str = "0",
    define_path: str = "src",
) -> None:
    """Generating C files for replaying pre-recorded integer data (data only) for using on microcontrollers
    Args:
        data:        List of integer sample values
        bitwidth:    Bit width of each sample
        signed:      Decision if data values are signed [otherwise unsigned]
        path2save:   Path to save the .h / .c output-files.
        replayer_id: ID appended to function names.
        define_path: Include path written into the generated #include line.
    """
    assert bitwidth in range(1, 65), "Bitwidth must be between 1 and 64"
    assert len(data) > 0, "data must not be empty"

    module_id = replayer_id.lower()
    params = {
        "datetime_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
        "path2include": define_path,
        "template_name": "replayer_template.h",
        "device_id": module_id.upper(),
        "data_type": get_embedded_datatype(bitwidth, signed),
        "num_values": str(len(data)),
        "data_values": ", ".join(str(v) for v in data),
    }
    template_c = _generate_replayer_template()
    generate_c_files(
        path2save=path2save,
        template_name=params["template_name"],
        file_name="replayer",
        module_id=module_id,
        proto_file=replace_variables_with_parameters(template_c["head"], params),
        impl_file=replace_variables_with_parameters(template_c["func"], params),
        path2template=Path(design_plugin.__file__).parent / "c",
    )


def build_replayer_with_trigger(
    data: list[int],
    trigger: list[int],
    bitwidth: int,
    signed: bool,
    path2save: Path,
    replayer_id: str = "0",
    define_path: str = "src",
) -> None:
    """Generating C files for replaying pre-recorded data with a trigger channel for using on microcontrollers
    Args:
        data:        List of integer sample values
        trigger:     List of trigger bits, same length as data
        bitwidth:    Bit width of each sample.
        signed:      Decision if data values are signed [otherwise unsigned]
        path2save:   Path to save the .h / .c files.
        replayer_id: ID appended to function names.
        define_path: Include path written into the generated #include line.
    """
    assert bitwidth in range(1, 65), "Bitwidth must be between 1 and 64"
    assert len(data) > 0, "data must not be empty"
    assert len(data) == len(trigger), "data and trigger must have the same length"

    module_id = replayer_id.lower()
    trgg_array_name = f"replayer_trgg_data_{module_id.upper()}"
    params = {
        "datetime_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
        "path2include": define_path,
        "template_name": "replayer_template.h",
        "device_id": module_id.upper(),
        "data_type": get_embedded_datatype(bitwidth, signed),
        "num_values": str(len(data)),
        "data_values": ", ".join(str(v) for v in data),
        "trgg_values": ", ".join(str(v) for v in trigger),
        "trgg_array_name": trgg_array_name,
    }
    template_c = _generate_replayer_trgg_template()
    generate_c_files(
        path2save=path2save,
        template_name=params["template_name"],
        file_name="replayer",
        module_id=module_id,
        proto_file=replace_variables_with_parameters(template_c["head"], params),
        impl_file=replace_variables_with_parameters(template_c["func"], params),
        path2template=Path(design_plugin.__file__).parent / "c",
    )


def _generate_replayer_template() -> dict[str, list[str]]:
    header_template = [
        "// --- Generating a Replayer (data only)",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: ID = {$device_id},  N = {$num_values},  type = {$data_type}",
        '# include "{$path2include}/{$template_name}"',
        "DEF_NEW_REPLAYER_PROTO({$device_id}, {$data_type})",
    ]
    impl_template = [
        "// --- Generating a Replayer (data only)",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: ID = {$device_id},  N = {$num_values},  type = {$data_type}",
        '# include "{$path2include}/{$template_name}"',
        "DEF_NEW_REPLAYER_IMPL({$device_id}, {$data_type}, {$num_values}, {$data_values})",
    ]
    return {"head": header_template, "func": impl_template}


def _generate_replayer_trgg_template() -> dict[str, list[str]]:
    header_template = [
        "// --- Generating a Replayer (data + trigger)",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: ID = {$device_id},  N = {$num_values},  type = {$data_type}",
        '# include "{$path2include}/{$template_name}"',
        "DEF_NEW_REPLAYER_TRGG_PROTO({$device_id}, {$data_type})",
    ]
    impl_template = [
        "// --- Generating a Replayer (data + trigger)",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: ID = {$device_id},  N = {$num_values},  type = {$data_type}",
        '# include "{$path2include}/{$template_name}"',
        "static uint8_t {$trgg_array_name}[] = { {$trgg_values} };",
        "DEF_NEW_REPLAYER_TRGG_IMPL({$device_id}, {$data_type}, {$num_values}, {$trgg_array_name}, {$data_values})",
    ]
    return {"head": header_template, "func": impl_template}
