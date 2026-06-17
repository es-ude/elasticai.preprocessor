from collections.abc import Iterable
from datetime import datetime

from elasticai.creator.file_generation.resource_utils import read_text
from elasticai.creator.hdl_ir import DataGraph
from elasticai.creator.ir2verilog import (
    Code,
    Registry,
    TemplateDirector,
    type_handler_iterable,
)


@type_handler_iterable()
def waveform_lut_opt(impl: DataGraph, _: Registry) -> Iterable[Code]:
    package_path = "elasticai.creator_plugins.waveform"
    path2file = "verilog/waveform_lut_opt.v"

    _template = (
        TemplateDirector()
        .define_scoped_switch("ACCESS_EXTERNAL", False)
        .define_scoped_switch("TRGG_EXTERNAL", False)
        .parameter("LUTWIDTH")
        .parameter("BITWIDTH")
        .parameter("WAIT_WIDTH")
        .parameter("SIGNED_OUT")
        .localparam("LUT_DATA")
        .add_module_name()
        .set_prototype("\n".join(read_text(package_path, path2file)))
        .build()
    )

    code = list()
    code.append(
        (
            impl.name,
            _template.substitute(
                module_name=impl.name.upper(),
                date_copy_created=datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                **impl.attributes,
            ),
        )
    )
    return code
