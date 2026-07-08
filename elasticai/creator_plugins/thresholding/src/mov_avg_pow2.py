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
def mov_avg_pow2(impl: DataGraph, _: Registry) -> Iterable[Code]:  #change def name to current module
    package_path = "elasticai.creator_plugins.thresholding"     #change this to current module
    path2file = "verilog/mov_avg_pow2.v"                        #change this to current module

    #Parameters need to fit to verilog module
    _template = (
        TemplateDirector()
        .parameter("BITWIDTH")
        .parameter("LENGTH")
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
