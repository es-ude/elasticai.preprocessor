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
def rescaler_pow2(impl: DataGraph, _: Registry) -> Iterable[Code]:
    package_path = "elasticai.creator_plugins.adc"
    path2file = "verilog/rescaler_pow2.v"

    _template = (
        TemplateDirector()
        .parameter("BITWIDTH_IN")
        .parameter("IS_SIGNED_IN")
        .parameter("BITWIDTH_OUT")
        .parameter("IS_SIGNED_OUT")
        .parameter("NUM_SAMPLES")
        .localparam("SHIFT")
        .localparam("ZERO_POINT")
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
def rescaler_mult(impl: DataGraph, _: Registry) -> Iterable[Code]:
    package_path = "elasticai.creator_plugins.adc"
    path2file = "verilog/rescaler_mult.v"

    _template = (
        TemplateDirector()
        .parameter("BITWIDTH_IN")
        .parameter("IS_SIGNED_IN")
        .parameter("BITWIDTH_OUT")
        .parameter("IS_SIGNED_OUT")
        .parameter("NUM_SAMPLES")
        .localparam("SCALE_MULT")
        .localparam("SCALE_BIAS")
        .localparam("ZERO_POINT")
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
