from datetime import datetime
from pathlib import Path

import elasticai.creator_plugins.normalization as design_plugin
from elasticai.preprocessor.translation.ir2c import (
    generate_c_files,
    get_embedded_datatype,
    replace_variables_with_parameters,
)


def build_normalization_zscore(
    bitwidth: int,
    signed: bool,
    path2save: Path,
    normalization_id: str = "0",
    define_path: str = "src",
) -> None:
    """Generate C files for frame-wise zscore normalization."""
    assert bitwidth in range(2, 33), "Bitwidth must be between 2 and 32"

    module_id = normalization_id.lower()
    params = {
        "datetime_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
        "path2include": define_path,
        "template_name": "normalization_zscore_template.h",
        "device_id": module_id.upper(),
        "data_type": get_embedded_datatype(bitwidth, signed),
    }
    template_c = _generate_normalization_zscore_template()
    generate_c_files(
        path2save=path2save,
        template_name=params["template_name"],
        file_name="normalization_zscore",
        module_id=module_id,
        proto_file=replace_variables_with_parameters(template_c["head"], params),
        impl_file=replace_variables_with_parameters(template_c["func"], params),
        path2template=Path(design_plugin.__file__).parent / "c",
    )


def _generate_normalization_zscore_template() -> dict[str, list[str]]:
    header_template = [
        "// --- Generating zscore normalization",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        '#include "{$path2include}/{$template_name}"',
        "DEF_NEW_NORMALIZATION_ZSCORE_PROTO({$device_id}, {$data_type})",
    ]
    implementation_template = [
        "// --- Generating zscore normalization",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        '#include "{$path2include}/{$template_name}"',
        "DEF_NEW_NORMALIZATION_ZSCORE_IMPL({$device_id}, {$data_type})",
    ]
    return {"head": header_template, "func": implementation_template}
