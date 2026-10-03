from datetime import datetime
from numbers import Integral
from pathlib import Path

import elasticai.creator_plugins.datarate as design_plugin
from elasticai.preprocessor.translation.ir2c import (
    generate_c_files,
    replace_variables_with_parameters,
)


def build_downsampling_float32(
    downsampling_ratio: int,
    path2save: Path,
    downsampling_id: str = "0",
    define_path: str = "src",
) -> None:
    """Generate stateless frame-wise Float32 subsampling starting at index zero."""
    if (
        isinstance(downsampling_ratio, bool)
        or not isinstance(downsampling_ratio, Integral)
        or downsampling_ratio < 1
    ):
        raise ValueError("downsampling_ratio must be a positive integer")
    if downsampling_ratio == 1:
        return

    module_id = downsampling_id.lower()
    generated_header = f"downsampling_float32_{module_id}.h"
    params = {
        "datetime_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
        "header_guard": f"DOWNSAMPLING_FLOAT32_{module_id.upper()}_H",
        "path2include": define_path,
        "template_name": "downsampling_float32_template.h",
        "generated_header": generated_header,
        "device_id": module_id,
        "downsampling_ratio": str(int(downsampling_ratio)),
    }
    generated = _generate_downsampling_float32_template()
    generate_c_files(
        path2save=path2save,
        template_name=params["template_name"],
        file_name="downsampling_float32",
        module_id=module_id,
        proto_file=replace_variables_with_parameters(generated["head"], params),
        impl_file=replace_variables_with_parameters(generated["func"], params),
        path2template=Path(design_plugin.__file__).parent / "c",
    )


def _generate_downsampling_float32_template() -> dict[str, list[str]]:
    header = [
        "// --- Generating stateless Float32 subsampling from index zero",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: dsr = {$downsampling_ratio}",
        "#ifndef {$header_guard}",
        "#define {$header_guard}",
        '#include "{$path2include}/{$template_name}"',
        "#ifdef __cplusplus",
        'extern "C" {',
        "#endif",
        "DEF_DOWNSAMPLING_FLOAT32_PROTO({$device_id})",
        "#ifdef __cplusplus",
        "}",
        "#endif",
        "#endif",
    ]
    implementation = [
        "// --- Generating stateless Float32 subsampling from index zero",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: dsr = {$downsampling_ratio}",
        '#include "{$path2include}/{$generated_header}"',
        "DEF_DOWNSAMPLING_FLOAT32_IMPL({$device_id}, {$downsampling_ratio})",
    ]
    return {"head": header, "func": implementation}
