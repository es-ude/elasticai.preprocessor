from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from shutil import copyfile

import elasticai.creator_plugins.windower as design_plugin
from elasticai.preprocessor.eventdetection.preprocessors import EventPreprocessorsTargets
from elasticai.preprocessor.thresholding import ThresholdTargets
from elasticai.preprocessor.translation.ir2c import (
    generate_c_files,
    get_embedded_datatype,
    replace_variables_with_parameters,
)
from elasticai.preprocessor.windower.window import WindowSettings

_CREATOR_PLUGINS  = Path(__file__).parents[3]
_EVENTDETECTION_C = _CREATOR_PLUGINS / "eventdetection" / "c"
_THRESHOLDING_C   = _CREATOR_PLUGINS / "thresholding" / "c"
_WINDOWER_C_DIR   = _CREATOR_PLUGINS / "windower" / "c"

# (template_file, c_function_base_name, macro_renderer(uid, dtype), has_output_delay)
# has_output_delay=True: preprocessing outputs the result for the PREVIOUS sample
# (i.e. neo[k-1] when receiving signal[k]).  The wrapper must then store signal[k-1]
# in output_buf instead of signal[k] so that detect_buf and output_buf stay in sync.
_PREPROCESSING_TABLE: dict[EventPreprocessorsTargets, tuple[str, str, Callable[..., str], bool]] = {
    EventPreprocessorsTargets.Normal: (
        "preprocessing_normal_template.h",
        "calc_preprocessing_normal",
        lambda uid, dtype: f"DEF_NEW_PREPROCESSING_NORMMAL_IMPL({uid}, {dtype})",
        False,
    ),
    EventPreprocessorsTargets.Absolute: (
        "preprocessing_abs_template.h",
        "calc_preprocessing_abs",
        lambda uid, dtype: f"DEF_NEW_PREPROCESSING_ABS_IMPL({uid}, {dtype})",
        False,
    ),
    EventPreprocessorsTargets.NEO: (
        "preprocessing_neo_template.h",
        "calc_neo",
        lambda uid, dtype: f"DEF_NEW_PREPROCESSING_NEO_IMPL({uid}, {dtype})",
        True,   # outputs neo[k-1] at step k; use prev_data in output_buf
    ),
}

# (template_file, c_function_base_name, macro_renderer(uid, dtype, settings))
# Offline methods (Constant, AbsoluteMean, MAD, RmsNorm, RmsBlackrock): the threshold
# value is computed externally and supplied via settings.threshold as a compile-time constant.
_THRESHOLDING_TABLE: dict[ThresholdTargets, tuple[str, str, Callable[..., str]]] = {
    ThresholdTargets.Constant: (
        "thresholding_constant_template.h",
        "calc_thresholding_constant",
        lambda uid, dtype, s: f"DEF_CONSTANT_THR_IMPL({uid}, {dtype}, {int(s.threshold)})",
    ),
    ThresholdTargets.AbsoluteMean: (
        "thresholding_constant_template.h",
        "calc_thresholding_constant",
        lambda uid, dtype, s: f"DEF_CONSTANT_THR_IMPL({uid}, {dtype}, {int(s.threshold)})",
    ),
    ThresholdTargets.MedianAbsoluteDeviation: (
        "thresholding_constant_template.h",
        "calc_thresholding_constant",
        lambda uid, dtype, s: f"DEF_CONSTANT_THR_IMPL({uid}, {dtype}, {int(s.threshold)})",
    ),
    ThresholdTargets.RmsNorm: (
        "thresholding_constant_template.h",
        "calc_thresholding_constant",
        lambda uid, dtype, s: f"DEF_CONSTANT_THR_IMPL({uid}, {dtype}, {int(s.threshold)})",
    ),
    ThresholdTargets.RmsBlackrock: (
        "thresholding_constant_template.h",
        "calc_thresholding_constant",
        lambda uid, dtype, s: f"DEF_CONSTANT_THR_IMPL({uid}, {dtype}, {int(s.threshold)})",
    ),
    ThresholdTargets.MovingAverage: (
        "thresholding_mavg_template.h",
        "calc_thresholding_mavg",
        lambda uid, dtype, s: (
            f"DEF_CALC_MAVG_THR({uid}, {dtype})\n"
            f"DEF_NEW_MAVG_WINDOW_IMPL({uid}, {dtype}, {int(s.window_sec / 2 * s.sampling_rate)})"
        ),
    ),
    ThresholdTargets.Welford: (
        "thresholding_welford_template.h",
        "calc_thresholding_welford",
        lambda uid, dtype, s: f"DEF_WELFORD_THR_IMPL({uid}, {dtype})",
    ),
}


def build_windower_sequence(
    settings: WindowSettings,
    bitwidth: int,
    signed: bool,
    path2save: Path,
    windower_id: str = "0",
    define_path: str = "src",
) -> None:
    """Generate C files for a sequence windower (streaming interface).
    Args:
        settings:       Window settings (sampling_rate, window_sec, overlap_sec)
        bitwidth:       Bit width of each sample
        signed:         Whether the data type is signed
        path2save:      Path to save the .h / .c output files
        windower_id:    ID appended to the generated function name
        define_path:    Include path written into the generated #include line
    """
    assert bitwidth in range(2, 33), "bitwidth must be between 2 and 32"
    module_id = windower_id.lower()
    params = {
        "datetime_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
        "path2include": define_path,
        "template_name": "windower_sequence_template.h",
        "device_id": module_id.upper(),
        "data_type": get_embedded_datatype(bitwidth, signed),
        "window_length": str(settings.window_length),
        "num_shift": str(settings.window_length),
    }
    template_c = _generate_windower_sequence_template()
    generate_c_files(
        path2save=path2save,
        template_name=params["template_name"],
        file_name="windower_sequence",
        module_id=module_id,
        proto_file=replace_variables_with_parameters(template_c["head"], params),
        impl_file=replace_variables_with_parameters(template_c["func"], params),
        path2template=Path(design_plugin.__file__).parent / "c",
    )


def _generate_windower_sequence_template() -> dict[str, list[str]]:
    header_template = [
        "// --- Generating windower_sequence",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: ID = {$device_id}, type = {$data_type}, window = {$window_length}",
        '#include "{$path2include}/{$template_name}"',
        "DEF_WINDOWER_SEQUENCE_PROTO({$device_id}, {$data_type})",
    ]
    implementation_template = [
        "// --- Generating windower_sequence",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: ID = {$device_id}, type = {$data_type}, window = {$window_length}",
        '#include "{$path2include}/{$template_name}"',
        "DEF_WINDOWER_SEQUENCE_IMPL({$device_id}, {$data_type}, {$window_length})",
    ]
    return {"head": header_template, "func": implementation_template}


def build_windower_sliding(
    settings: WindowSettings,
    bitwidth: int,
    signed: bool,
    path2save: Path,
    windower_id: str = "0",
    define_path: str = "src",
) -> None:
    """Generate C files for a sliding-window windower (streaming interface).
    Args:
        settings:       Window settings (sampling_rate, window_sec, overlap_sec)
        bitwidth:       Bit width of each sample
        signed:         Whether the data type is signed
        path2save:      Path to save the .h / .c output files
        windower_id:    ID appended to the generated function name
        define_path:    Include path written into the generated #include line
    """
    assert bitwidth in range(2, 33), "bitwidth must be between 2 and 32"

    window_length = settings.window_length
    num_shift = window_length - settings.overlap_length

    module_id = windower_id.lower()
    params = {
        "datetime_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
        "path2include": define_path,
        "template_name": "windower_sliding_template.h",
        "device_id": module_id.upper(),
        "data_type": get_embedded_datatype(bitwidth, signed),
        "window_length": str(window_length),
        "num_shift": str(num_shift),
    }
    template_c = _generate_windower_sliding_template()
    generate_c_files(
        path2save=path2save,
        template_name=params["template_name"],
        file_name="windower_sliding",
        module_id=module_id,
        proto_file=replace_variables_with_parameters(template_c["head"], params),
        impl_file=replace_variables_with_parameters(template_c["func"], params),
        path2template=Path(design_plugin.__file__).parent / "c",
    )


def _generate_windower_sliding_template() -> dict[str, list[str]]:
    header_template = [
        "// --- Generating windower_sliding",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: ID = {$device_id}, type = {$data_type}, window = {$window_length}, shift = {$num_shift}",
        '#include "{$path2include}/{$template_name}"',
        "DEF_WINDOWER_SLIDING_PROTO({$device_id}, {$data_type})",
    ]
    implementation_template = [
        "// --- Generating windower_sliding",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: ID = {$device_id}, type = {$data_type}, window = {$window_length}, shift = {$num_shift}",
        '#include "{$path2include}/{$template_name}"',
        "DEF_WINDOWER_SLIDING_IMPL({$device_id}, {$data_type}, {$window_length}, {$num_shift})",
    ]
    return {"head": header_template, "func": implementation_template}


def build_windower_event(
    settings: WindowSettings,
    threshold: int,
    pre_padding: int,
    bitwidth: int,
    signed: bool,
    path2save: Path,
    windower_id: str = "0",
    define_path: str = "src",
) -> None:
    """Generate C files for a event-triggered windower (streaming interface).
    Args:
        settings:       Window settings (sampling_rate, window_sec, overlap_sec)
        threshold:      theshold for detecting event
        pre_padding:    number of samples before detected event
        bitwidth:       Bit width of each sample
        signed:         Whether the data type is signed
        path2save:      Path to save the .h / .c output files
        windower_id:    ID appended to the generated function name
        define_path:    Include path written into the generated #include line
    """
    assert bitwidth in range(2, 33), "bitwidth must be between 2 and 32"

    window_length = settings.window_length
    pre_samples = pre_padding

    module_id = windower_id.lower()
    params = {
        "datetime_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
        "path2include": define_path,
        "template_name": "windower_event_template.h",
        "device_id": module_id.upper(),
        "data_type": get_embedded_datatype(bitwidth, signed),
        "window_length": str(window_length),
        "pre_samples": str(pre_samples),
        "threshold": str(threshold),
    }
    template_c = _generate_windower_event_template()
    generate_c_files(
        path2save=path2save,
        template_name=params["template_name"],
        file_name="windower_event",
        module_id=module_id,
        proto_file=replace_variables_with_parameters(template_c["head"], params),
        impl_file=replace_variables_with_parameters(template_c["func"], params),
        path2template=Path(design_plugin.__file__).parent / "c",
    )


def build_windower_event_pipeline(
    settings: WindowSettings,
    bitwidth: int,
    signed: bool,
    path2save: Path,
    windower_id: str = "0",
    define_path: str = "src",
) -> None:
    """Generate C files for a full event-windower pipeline: preprocessing → thresholding → windower.

    Event positions are detected on the preprocessed signal; window content comes from
    the original signal.

    Args:
        settings:     Window settings including method_input, method_thr, threshold, pre_time
        bitwidth:     Bit width of each sample
        signed:       Whether the data type is signed
        path2save:    Directory where generated files are written
        windower_id:  ID appended to all generated function names
        define_path:  Include prefix written into the generated #include lines
    """
    if settings.method_input not in _PREPROCESSING_TABLE:
        raise NotImplementedError(
            f"No C template for preprocessing method '{settings.method_input.value}'"
        )
    if settings.method_thr not in _THRESHOLDING_TABLE:
        raise NotImplementedError(
            f"No C template for threshold method '{settings.method_thr.value}'"
        )

    assert bitwidth in range(2, 33), "bitwidth must be between 2 and 32"
    path2save.mkdir(parents=True, exist_ok=True)

    mid   = windower_id.lower()
    uid   = mid.upper()
    dtype = get_embedded_datatype(bitwidth, signed)
    ts    = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    wl    = settings.window_length
    pre   = int(settings.pre_time * settings.sampling_rate)

    preproc_tmpl, preproc_fn, preproc_macro_fn, preproc_has_delay = _PREPROCESSING_TABLE[settings.method_input]
    thr_tmpl,     thr_fn,     thr_macro_fn                       = _THRESHOLDING_TABLE[settings.method_thr]

    # Copy template headers (deduplicated, order preserved).
    # windower_event_template.h is NOT needed: the wrapper uses dual ring buffers
    # generated inline, not WindowerEventTaps / DEF_WINDOWER_EVENT.
    raw_templates = [
        (_EVENTDETECTION_C, preproc_tmpl),
        (_THRESHOLDING_C,   thr_tmpl),
    ]
    unique_templates: list[tuple[Path, str]] = []
    seen: set[str] = set()
    for src_dir, name in raw_templates:
        if name not in seen:
            copyfile(src=str(src_dir / name), dst=str(path2save / name))
            unique_templates.append((src_dir, name))
            seen.add(name)

    includes = "\n".join(
        f'#include "{define_path}/{name}"' for _, name in unique_templates
    )

    stage_macros = "\n".join([
        preproc_macro_fn(uid, dtype),
        thr_macro_fn(uid, dtype, settings),
    ])

    # Dual-buffer wrapper:
    #   detect_buf  — preprocessed signal, used for threshold comparison
    #   output_buf  — original signal, copied into out[] on an event
    #
    # When has_output_delay=True (NEO): preprocessing outputs neo[k-1] at step k.
    # To keep both buffers in sync we store signal[k-1] (prev_data) in output_buf,
    # not signal[k].  This aligns the window content with the event position and
    # produces the same result as Python's centered NEO formula.
    if preproc_has_delay:
        early_ret   = f"    if (!{preproc_fn}_{mid}(data, &preprocessed)) {{ prev_data = data; return false; }}"
        prev_decl   = f"    static {dtype} prev_data = 0;"
        buf_assign  = f"    detect_buf[{wl} - 1] = preprocessed;\n    output_buf[{wl} - 1] = prev_data;\n    prev_data = data;"
    else:
        early_ret   = f"    if (!{preproc_fn}_{mid}(data, &preprocessed)) return false;"
        prev_decl   = ""
        buf_assign  = f"    detect_buf[{wl} - 1] = preprocessed;\n    output_buf[{wl} - 1] = data;"

    wrapper_lines = [
        f"bool calc_windower_event_pipeline_{mid}({dtype} data, {dtype} *out) {{",
        f"    {dtype} preprocessed;",
    ]
    if prev_decl:
        # Must be declared before early_ret uses it
        wrapper_lines.append(prev_decl)
    wrapper_lines += [
        early_ret,
        f"    {dtype} threshold;",
        f"    {thr_fn}_{mid}(preprocessed, &threshold);",
        f"    static {dtype} detect_buf[{wl}];",
        f"    static {dtype} output_buf[{wl}];",
        "    static uint16_t count = 0;",
        "    static bool is_event = false;",
    ]
    wrapper_lines += [
        f"    for (uint16_t i = 1; i < {wl}; i++) {{",
        "        detect_buf[i - 1] = detect_buf[i];",
        "        output_buf[i - 1] = output_buf[i];",
        "    }",
        buf_assign,
        f"    if (count < {wl}) {{",
        "        count++;",
        f"        if (count < {wl}) return false;",
        "    }",
        f"    if (detect_buf[{pre}] >= threshold) {{",
        "        if (!is_event) {",
        "            is_event = true;",
        f"            for (uint16_t i = 0; i < {wl}; i++) out[i] = output_buf[i];",
        "            return true;",
        "        }",
        "        return false;",
        "    }",
        "    if (is_event) is_event = false;",
        "    return false;",
        "}",
    ]
    wrapper = "\n".join(wrapper_lines)

    header_comment = (
        f"// --- Generated event windower pipeline: "
        f"{settings.method_input.value} → {settings.method_thr.value} → windower_event\n"
        f"// Copyright @ UDE-IES\n"
        f"// Code generated on: {ts}\n"
        f"// ID = {mid}, type = {dtype}, window_length = {wl}, pre_samples = {pre}"
    )

    (path2save / f"windower_event_pipeline_{mid}.h").write_text(
        f"{header_comment}\n"
        f"#include <stdbool.h>\n"
        f"#include <stdint.h>\n"
        f"bool calc_windower_event_pipeline_{mid}({dtype} data, {dtype} *out);\n"
    )
    (path2save / f"windower_event_pipeline_{mid}.c").write_text(
        f"{header_comment}\n"
        f"{includes}\n\n"
        f"{stage_macros}\n\n"
        f"{wrapper}\n"
    )


def _generate_windower_event_template() -> dict[str, list[str]]:
    header_template = [
        "// --- Generating windower_event",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: ID = {$device_id}, type = {$data_type}, window = {$window_length}",
        "// Params: threshold = {$threshold}, pre_samples = {$pre_samples}"
        '#include "{$path2include}/{$template_name}"',
        "DEF_WINDOWER_EVENT_PROTO({$device_id}, {$data_type})",
    ]
    implementation_template = [
        "// --- Generating windower_event",
        "// Copyright @ UDE-IES",
        "// Code generated on: {$datetime_created}",
        "// Params: ID = {$device_id}, type = {$data_type}, window = {$window_length},",
        "// Params: threshold = {$threshold}, pre_samples = {$pre_samples}"
        '#include "{$path2include}/{$template_name}"',
        "DEF_WINDOWER_EVENT_IMPL({$device_id}, {$data_type}, {$window_length}, {$num_shift})",
    ]
    return {"head": header_template, "func": implementation_template}
