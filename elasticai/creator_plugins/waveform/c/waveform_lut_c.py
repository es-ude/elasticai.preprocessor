def generate_waveform_lut_template(do_opt: bool) -> dict:
    """Generating the C template for generating the waveforms
    :param do_opt:  Get C functions for full LUT representation or reduced LUT with mirror techniques
    :return:        Dictionary with files
    """
    use_option = "OPT" if do_opt else "FULL"
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
