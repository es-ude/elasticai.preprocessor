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
def waveform_ram_full(impl: DataGraph, _: Registry) -> Iterable[Code]:
    package_path = "elasticai.creator_plugins.waveform"
    path2file = "verilog/waveform_ram_full.v"
    id = impl.name.upper().split("_")[-1]

    _template = (
        TemplateDirector()
        .define_scoped_switch("TRGG_EXTERNAL", False)
        .parameter("RAMWIDTH")
        .parameter("BITWIDTH")
        .parameter("WAIT_WIDTH")
        .parameter("PATH2MEM")
        .replace_instance_name("BRAM_SINGLE", f"BRAM_SINGLE_PORT_{id}")
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
