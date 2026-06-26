from datetime import datetime
from pathlib import Path

import elasticai.creator_plugins.downsampling as design_plugin
from elasticai.preprocessor.translation.ir2c import (
    generate_c_files,
    get_embedded_datatype,
    replace_variables_with_parameters,
)


def build_downsampling_subsampling(
    downsampling_ratio: int,
    bitwidth: int,
    signed: bool,
    path2save: Path,
    downsampling_id: str = "0",
    define_path: str = "src",
) -> None:
    """Generate C files for subsampling."""
    if downsampling_ratio < 1:
        raise ValueError("dsr must be >= 1")
    assert bitwidth in range(2, 33), "Bitwidth must be between 2 and 32"

    module_id = downsampling_id.lower()
    params = {
        "datetime_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
        "path2include": define_path,
        "template_name": "downsampling_subsampling_template.h",
        "device_id": module_id.upper(),
        "data_type": get_embedded_datatype(bitwidth, signed),
        "downsampling_ratio": str(downsampling_ratio),
    }
    template_c = _generate_downsampling_subsampling_template()
    generate_c_files(
        path2save=path2save,
        template_name=params["template_name"],
        file_name="downsampling_subsampling",
        module_id=module_id,
        proto_file=replace_variables_with_parameters(template_c["head"], params),
        impl_file=replace_variables_with_parameters(template_c["func"], params),
        path2template=Path(design_plugin.__file__).parent / "c",
    )


def _generate_downsampling_subsampling_template() -> dict[str, list[str]]:
    header_template = [
        "// --- Generating subsampling",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: dsr = {$downsampling_ratio}",
        '#include "{$path2include}/{$template_name}"',
        "DEF_DOWNSAMPLING_SUBSAMPLING_PROTO({$device_id}, {$data_type})",
    ]
    implementation_template = [
        "// --- Generating subsampling",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: dsr = {$downsampling_ratio}",
        '#include "{$path2include}/{$template_name}"',
        "DEF_DOWNSAMPLING_SUBSAMPLING_OUTPUT_LENGTH({$device_id}, {$downsampling_ratio})",
        "DEF_DOWNSAMPLING_SUBSAMPLING_IMPL({$device_id}, {$data_type}, {$downsampling_ratio})",
    ]
    return {"head": header_template, "func": implementation_template}
