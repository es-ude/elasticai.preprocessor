from datetime import datetime
from pathlib import Path

import elasticai.creator_plugins.sequential as design_plugin
from elasticai.preprocessor.translation.ir2c import (
    generate_c_files,
    get_embedded_datatype,
    replace_variables_with_parameters,
)


def build_sequential(
    seq_id: str,
    path2save: Path,
    bitwidth_input: int,
    bitwidth_output: int,
    num_inputs: int,
    num_outputs: int,
    is_signed: bool,
) -> None:
    """Generating C files for building the sequential layer
    Args:
        seq_id: ID appended to function names.
        path2save:          Path to save the .h / .c output-files.

        bitwidth_input:     Integer with bitwidth of each input sample
        bitwidth_output:    Integer with bitwidth of each output sample
        num_inputs:         Integer with number of input samples
        num_outputs:        Integer with number of output samples
        is_signed:          Decision if data values are signed [otherwise unsigned]
    """
    assert bitwidth_input in range(1, 65), "Bitwidth must be between 1 and 64"
    assert bitwidth_output in range(1, 65), "Bitwidth must be between 1 and 64"

    module_id = seq_id.lower()
    params = {
        "datetime_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
        "template_name": "sequential_template.h",
        "device_id": module_id.upper(),
        "input_type": get_embedded_datatype(bitwidth_input, is_signed),
        "output_type": get_embedded_datatype(bitwidth_output, is_signed),
        "num_inputs": str(num_inputs),
        "num_outputs": str(num_outputs),
    }
    template_c = _generate_sequential_template()
    generate_c_files(
        path2save=path2save,
        template_name=params["template_name"],
        file_name="sequential",
        module_id=module_id,
        proto_file=replace_variables_with_parameters(template_c["head"], params),
        impl_file=replace_variables_with_parameters(template_c["func"], params),
        path2template=Path(design_plugin.__file__).parent / "c",
    )


def _generate_sequential_template() -> dict[str, list[str]]:
    header_template = [
        "// --- Generating a Sequential Layer",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: ID = {$device_id}",
        '# include "./{$template_name}"',
        "DEF_SEQUENTIAL_PROTO({$device_id}, {$input_type}, {$output_type})",
    ]
    impl_template = [
        "// --- Generating a Sequential Layer",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: ID = {$device_id}",
        '# include "./{$template_name}"',
        "DEF_SEQUENTIAL_IMPL({$device_id}, {$input_type}, {$output_type}, {$num_inputs}, {$num_outputs})",
    ]
    return {"head": header_template, "func": impl_template}
