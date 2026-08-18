from datetime import datetime
from pathlib import Path

import elasticai.creator_plugins.eventdetection as design_plugin
from elasticai.preprocessor.translation.ir2c import (
    generate_c_files,
    get_embedded_datatype,
    replace_variables_with_parameters,
)


def build_eventdetection(
    threshold:       int,
    hysteresis:      float,
    hysteresis_type: str,
    bitwidth: int,
    signed: bool,
    path2save: Path,
    eventdetection_id: str = "0",
    define_path: str = "src",
) -> None:
    """Generate C files for eventdetection.
    Args: 
        threshold:  Integer with threshold if sample is event. 
        hysteresis: relative hysteresis factor. 
        hysteresis_type: Applied types of hysteresis[ 
            'eventdetection_normal': no hysteresis,
            'eventdetection_pos_hyst': event on greater hysteresis,
            'eventdetection_neg_hyst': event off less hysteresis,
            'eventdetection_double_hyst': combined pos_hyst and neg_hyst, 
            ].
        bitwidth: bitwidth of each sample.
        signed: Decision of data values are signed [otherwise unsigned]. 
        path2save: Path to save the .h/.c output-files. 
        eventdetection_id: ID appended to function names. 
        define_path: include path written into the generated #include line.
    """
    assert bitwidth in range(2,33), "bitwidth must be between 2 and 32"

    module_id = eventdetection_id.lower()
    hyster_type = hysteresis_type.upper()
    params = {
        "datetime_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
        "path2include": define_path,
        "template_name": "eventdetection_template.h",
        "module_id": module_id,
        "device_id": module_id.upper(),
        "data_type": get_embedded_datatype(bitwidth, signed),
        "threshold": str(threshold),
        "hysteresis": str(hysteresis),
        "hysteresis_type": hyster_type,
    }
    template_c = _generate_eventdetection_template()
    generate_c_files(
        path2save=path2save,
        template_name=params["template_name"],
        file_name="eventdetection",
        module_id=module_id,
        proto_file=replace_variables_with_parameters(template_c["head"], params),
        impl_file=replace_variables_with_parameters(template_c["func"], params),
        path2template=Path(design_plugin.__file__).parent / "c", 
    )

def _generate_eventdetection_template() -> dict[str, list[str]]:
    header_template = [
        "// --- Generating eventdetection",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: ID = {$device_id}, type = {$data_type}, threshold = {$threshold},",
        "// hysteresis = {$hysteresis}, hysteresis_type = {$hysteresis_type}",
        '#include "{$path2include}/{$template_name}"',
        "EVENT_SETTINGS({$device_id}, {$data_type})",
        "DEF_NEW_EVENTDETECTION_PROTO({$device_id}, {$data_type})",
    ]
    implementation_template = [
        "// --- Generating eventdetection",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: ID = {$device_id}, type = {$data_type}, threshold = {$threshold},",
        "// hysteresis = {$hysteresis}, hysteresis_type = {$hysteresis_type}",
        '#include "{$path2include}/eventdetection_{$module_id}.h"',
        "TYPE_HYSTERESIS({$device_id}, {$data_type})",
        "DEF_NEW_EVENTDETECTION_IMPL({$device_id}, {$data_type}, {$threshold}, {$hysteresis}, {$hysteresis_type})",
    ]
    return {"head": header_template, "func": implementation_template}
