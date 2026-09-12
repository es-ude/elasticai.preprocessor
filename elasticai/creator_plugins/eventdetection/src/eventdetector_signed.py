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
def eventdetector_normal_signed(impl: DataGraph, _: Registry) -> Iterable[Code]:
    package_path = "elasticai.creator_plugins.eventdetection"
    path2file = "verilog/eventdetector_normal_signed.v"

    _template = (
        TemplateDirector()
        .parameter("BITWIDTH")
        .parameter("OUT_INVERT")
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


@type_handler_iterable()
def eventdetector_sub_signed(impl: DataGraph, _: Registry) -> Iterable[Code]:
    package_path = "elasticai.creator_plugins.eventdetection"
    path2file = "verilog/eventdetector_sub_signed.v"

    _template = (
        TemplateDirector()
        .parameter("BITWIDTH")
        .parameter("OUT_INVERT")
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
