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
def biquad_df1(impl: DataGraph, _: Registry) -> Iterable[Code]:
    package_path = "elasticai.creator_plugins.filter_data"
    path2file = "verilog/filter_biquad_df1.v"

    _template = (
        TemplateDirector()
        .define_scoped_switch("USE_EXT_WEIGHTS", False)
        .define_scoped_switch("USE_EXT_MAC", False)
        .parameter("BITWIDTH")
        .parameter("NUM_MULT")
        .localparam("FILT_COEFFS")
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
