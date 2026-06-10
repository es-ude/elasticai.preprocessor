from collections.abc import Iterable
from elasticai.creator.ir2verilog import (
    type_handler_iterable,
    Code,
    TemplateDirector,
)
from importlib import resources as res
from elasticai.creator.hdl_ir import DataGraph


@type_handler_iterable
def filter_cic(impl: DataGraph) -> Iterable[Code]:
    package_path = "elasticai.creator_plugins.filter"
    code = list()

    _template = (
        TemplateDirector()
        .parameter("BITWIDTH")
        .parameter("N_DEC")
        .parameter("DEC_RATE")
        .add_module_name()
        .set_prototype(res.read_text(package_path, "verilog/filter_cic.v"))
        .build()
    )
    code.append((impl.name, _template.substitute(impl.attributes)))

    if impl.data['build_tb']:
        _testbench = (
            TemplateDirector()
            .localparam("BITWIDTH")
            .localparam("N_DEC")
            .localparam("DEC_RATE")
            .add_module_name()
            .replace_instance_name("FILTER_CIC", impl.name.upper())
            .set_prototype(res.read_text(package_path, "verilog/filter_cic_tb.v"))
            .build()
        )
        tb_name = f"{impl.name}_tb"
        tb_attributes = impl.attributes | dict(module_name=tb_name.upper())
        code.append((tb_name, _testbench.substitute(tb_attributes)))
    return code
