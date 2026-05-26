import numpy as np
from os.path import dirname, abspath
from datetime import datetime


import plugins_c.filter_data as design_plugin
from elasticai.preprocessor import get_path_to_project
from elasticai.preprocessor.framing import SettingsFrame, FrameGenerator
from elasticai.preprocessor.translation.ir2c import (
    get_embedded_datatype,
    replace_variables_with_parameters,
    generate_c_files,
)


def build_waveform_lut(
    bitwidth: int,
    signed: bool,
    f_rpt: float,
    f_wvf: float,
    do_optimized: bool,
    type_sig: str = "sine",
    module_id: str = "0",
    path2save: str = get_path_to_project("build"),
    define_path: str = "src",
) -> None:
    """Generating C file with SINE_LUT for sinusoidal waveform generation
    Args:
        bitwidth:       Used quantization level for generating sinusoidal waveform LUT
        signed:         Decision if LUT values are signed [otherwise unsigned]
        f_rpt:          Frequency of the timer interrupt
        f_wvf:          Target frequency of the output waveform
        do_optimized:   Decision if LUT resources should be minimized [only quarter and mirroring]
        type_sig:       String with signal type (only: ['SINE', 'TRI', 'RECT'])
        module_id:      Device ID
        path2save:      Path for saving the verilog_filter output files
        define_path:    Path for loading the header file in IDE [Default: 'src']
    Return:
        None
    """
    assert int(f_rpt / f_wvf) > 12, "Ratio f_rpt/f_sine must be greater than 12"
    assert type_sig.upper() in ["SINE", "TRI", "RECT"], (
        "Only 'type' of ['SINE', 'TRI', 'RECT'] are supported"
    )
    module_id_used = f"{type_sig.lower()}{module_id.lower()}"
    datatype_data_ext = get_embedded_datatype(bitwidth, signed)
    bitwidth_mcu = int(datatype_data_ext.split("int")[-1].split("_")[0])

    sets = SettingsFrame(
        mode_thr=None,
        mode_align=None,
        sampling_rate=f_rpt,
        window_sec=None,
        offset_sec=None,
        align_sec=None,
        thr_gain=1.0,
    )
    data_lut = FrameGenerator(sets).g(
        time_points=[0.0],
        time_duration=[1 / f_wvf],
        waveform_select=[f"{type_sig.upper()}_FULL"],
        polarity_cathodic=[False],
        bitwidth=bitwidth,
        bitfrac=0,
        signed=signed,
        do_opt=do_optimized,
    )["sig"]
    data_lut = np.append(data_lut, data_lut[0]) if not do_optimized else data_lut

    # --- Step #2: Generating the values for parameter dict
    params = {
        "datetime_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
        "path2include": define_path,
        "template_name": "waveform_lut_template.h",
        "device_id": module_id_used.upper(),
        "datatype_cnt": get_embedded_datatype(
            np.ceil(np.log2(data_lut.size)), signed=False
        ),
        "datatype_int": get_embedded_datatype(bitwidth, signed=signed),
        "num_lutsine": str(data_lut.size),
        "lut_offset": str(
            0 if not do_optimized else (0 if signed else (2 ** (bitwidth_mcu - 1)))
        ),
        "lut_data": ", ".join(map(str, data_lut)),
        "lut_type": type_sig.lower(),
    }

    # --- Step #3: Replace string parameters with real values
    template = __generate_waveform_lut_template(do_optimized)
    generate_c_files(
        path2save=path2save,
        template_name=params["template_name"],
        file_name="waveform_lut",
        module_id=module_id_used.lower(),
        proto_file=replace_variables_with_parameters(template["head"], params),
        impl_file=replace_variables_with_parameters(template["func"], params),
        path2template=dirname(abspath(design_plugin.__file__)),
    )


def __generate_waveform_lut_template(do_full_opt: bool) -> dict:
    """Generating the C template for generating the waveforms
    Args:
        do_full_opt:    Get C functions for full LUT representation or reduced LUT with mirror techniques
    Return:
        Dictionary with files
    """
    use_option = "OPT" if do_full_opt else "FULL"
    header_temp = [
        f"// --- Generating a Waveform LUT Caller ({use_option})",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: ",
        "// \tID = {$device_id},",
        "// \tdatatypes (data, counter),",
        '# include "{$path2include}/{$template_name}"',
        "DEF_NEW_WAVEFORM_LUT_PROTO({$device_id}, {$datatype_int}, {$datatype_cnt})",
    ]
    func_temp = [
        f"// --- Generating a Waveform LUT Caller ({use_option})",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: ",
        "// \tID = {$device_id},",
        "// \tdatatypes (data, counter),",
        "// \tN_LUT = {$num_lutsine}",
        "// \tOffset = {$lut_offset}",
        "// \t Used LUT data order (a_0, a_1, ... a_N)",
        '# include "{$path2include}/{$template_name}"',
        "DEF_NEW_WAVEFORM_LUT_"
        + use_option
        + "_IMPL({$device_id}, {$datatype_int}, {$datatype_cnt}, {$num_lutsine}, {$lut_offset}, {$lut_data})",
    ]
    return {"head": header_temp, "func": func_temp, "params": []}
