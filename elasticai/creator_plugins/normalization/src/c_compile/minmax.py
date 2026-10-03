from datetime import datetime
from pathlib import Path

import elasticai.creator_plugins.normalization as design_plugin
from elasticai.preprocessor.translation.ir2c import (
    generate_c_files,
    get_embedded_datatype,
    replace_variables_with_parameters,
)


def build_normalization_minmax(
    bitwidth: int,
    signed: bool,
    path2save: Path,
    normalization_id: str = "0",
    define_path: str = "src",
) -> None:
    """Generate C files for frame-wise minmax normalization."""
    assert bitwidth in range(2, 33), "Bitwidth must be between 2 and 32"

    module_id = normalization_id.lower()
    _build_normalization_minmax(
        data_type=get_embedded_datatype(bitwidth, signed),
        module_id=module_id,
        device_id=module_id.upper(),
        path2save=path2save,
        define_path=define_path,
    )


def build_normalization_minmax_float32(
    path2save: Path,
    normalization_id: str = "0",
    define_path: str = "src",
) -> None:
    """Generate a Float32 abs-peak minmax normalization."""
    module_id = f"float32_{normalization_id.lower()}"
    _build_normalization_minmax(
        data_type="float",
        module_id=module_id,
        device_id=module_id,
        path2save=path2save,
        define_path=define_path,
    )


def _build_normalization_minmax(
    *,
    data_type: str,
    module_id: str,
    device_id: str,
    path2save: Path,
    define_path: str,
) -> None:
    generated_header = f"normalization_minmax_{module_id}.h"

    params = {
        "datetime_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
        "header_guard": f"NORMALIZATION_MINMAX_{module_id.upper()}_H",
        "path2include": define_path,
        "template_name": "normalization_minmax_template.h",
        "generated_header": generated_header,
        "device_id": device_id,
        "data_type": data_type,
    }
    template_c = _generate_normalization_minmax_template()
    generate_c_files(
        path2save=path2save,
        template_name=params["template_name"],
        file_name="normalization_minmax",
        module_id=module_id,
        proto_file=replace_variables_with_parameters(template_c["head"], params),
        impl_file=replace_variables_with_parameters(template_c["func"], params),
        path2template=Path(design_plugin.__file__).parent / "c",
    )


def _generate_normalization_minmax_template() -> dict[str, list[str]]:
    header_template = [
        "// --- Generating minmax normalization",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "#ifndef {$header_guard}",
        "#define {$header_guard}",
        '#include "{$path2include}/{$template_name}"',
        "#ifdef __cplusplus",
        'extern "C" {',
        "#endif",
        "DEF_NEW_NORMALIZATION_MINMAX_PROTO({$device_id}, {$data_type})",
        "#ifdef __cplusplus",
        "}",
        "#endif",
        "#endif",
    ]
    implementation_template = [
        "// --- Generating minmax normalization",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        '#include "{$path2include}/{$generated_header}"',
        "DEF_NEW_NORMALIZATION_MINMAX_IMPL({$device_id}, {$data_type})",
    ]
    return {"head": header_template, "func": implementation_template}
