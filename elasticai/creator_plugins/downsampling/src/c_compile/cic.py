from datetime import datetime
from pathlib import Path

import elasticai.creator_plugins.downsampling as design_plugin
from elasticai.preprocessor.translation.ir2c import (
    generate_c_files,
    get_embedded_datatype,
    replace_variables_with_parameters,
)


def build_downsampling_cic(
    downsampling_ratio: int,
    num_stages: int,
    bitwidth: int,
    signed: bool,
    path2save: Path,
    downsampling_id: str = "0",
    define_path: str = "src",
) -> None:
    """Generate C files for CIC downsampling.
    Args:
        downsampling_ratio: Decimation factor (SR_out = SR_in / downsampling_ratio)
        num_stages:         Number of CIC stages (integrators and comb filters)
        bitwidth:           Bit width of each sample
        signed:             Whether values are signed
        path2save:          Directory to write the generated .h / .c files into
        downsampling_id:    ID appended to function names
        define_path:        Include path written into the generated #include line
    """
    module_id = downsampling_id.lower()
    params = {
        "datetime_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
        "path2include": define_path,
        "template_name": "downsampling_cic_template.h",
        "device_id": module_id.upper(),
        "data_type": get_embedded_datatype(bitwidth, signed),
        "downsampling_ratio": str(downsampling_ratio),
        "num_stages": str(num_stages),
    }
    template_c = _generate_downsampling_cic_template()
    generate_c_files(
        path2save=path2save,
        template_name=params["template_name"],
        file_name="downsampling_cic",
        module_id=module_id,
        proto_file=replace_variables_with_parameters(template_c["head"], params),
        impl_file=replace_variables_with_parameters(template_c["func"], params),
        path2template=Path(design_plugin.__file__).parent / "c",
    )


def _generate_downsampling_cic_template() -> dict[str, list[str]]:
    header_template = [
        "// --- Generating do_cic",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: ID = {$device_id}, type = {$data_type}, dsr = {$downsampling_ratio}, stages = {$num_stages}",
        '#include "{$path2include}/{$template_name}"',
        "DEF_DOWNSAMPLING_CIC_PROTO({$device_id}, {$data_type})",
    ]
    implementation_template = [
        "// --- Generating do_cic",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: ID = {$device_id}, type = {$data_type}, dsr = {$downsampling_ratio}, stages = {$num_stages}",
        '#include "{$path2include}/{$template_name}"',
        "DOWNSAMPLING_CIC_OUTPUT_LENGTH({$device_id}, {$downsampling_ratio})",
        "DEF_DOWNSAMPLING_CIC_IMPL({$device_id}, {$data_type}, {$num_stages}, {$downsampling_ratio})",
    ]
    return {"head": header_template, "func": implementation_template}
