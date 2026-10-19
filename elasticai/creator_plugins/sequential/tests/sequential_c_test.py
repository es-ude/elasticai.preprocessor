from pathlib import Path
from shutil import which
from uuid import uuid4

import numpy as np
import pytest
from elasticai.creator.arithmetic import int_arithmetic
from elasticai.equichecker import CompileLoader, compare_values

from elasticai.creator_plugins.sequential.src.sequential_c import build_sequential
from elasticai.preprocessor import get_path_to_project
from elasticai.preprocessor.translation.cocotb_tmp import temporary_directory

pytestmark = pytest.mark.skipif(which("cc") is None, reason="requires a C compiler")


def test_build(tmp_path: Path) -> None:
    build_sequential(
        seq_id="0",
        path2save=tmp_path,
        bitwidth_input=8,
        bitwidth_output=12,
        num_inputs=1,
        num_outputs=1,
        is_signed=True,
    )
    assert (tmp_path / "sequential_template.h").exists()
    assert (tmp_path / "sequential_0.h").exists()
    assert (tmp_path / "sequential_0.c").exists()


@pytest.mark.simulation
@pytest.mark.parametrize(
    "idx, bitwidth_inp, bitwidth_out, num_inputs, num_outputs, is_signed",
    [
        (0, 8, 8, 1, 1, True),
        (0, 8, 8, 1, 1, False),
    ],
)
def test_simulation(
    tmp_path: Path,
    idx: int,
    bitwidth_inp: int,
    bitwidth_out: int,
    num_inputs: int,
    num_outputs: int,
    is_signed: bool,
) -> None:
    backup = get_path_to_project() / "build_test" / f"sequential_c_{idx}"
    with temporary_directory(backup) as tmp_path:
        build_sequential(
            seq_id=f"{idx}",
            path2save=tmp_path,
            bitwidth_input=bitwidth_inp,
            bitwidth_output=bitwidth_out,
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            is_signed=is_signed,
        )

        inp_type = "int8_t" if is_signed else "uint8_t"
        out_type = "int8_t" if is_signed else "uint8_t"
        adapter = tmp_path / "adapter.h"
        adapter.write_text(f"_Bool sequential_0({inp_type} *data, {out_type} *out);\n")
        loader = CompileLoader(
            headers=str(adapter),
            sources=[str(tmp_path / "sequential_0.c")],
            build_dir=str(tmp_path / "cffi-build"),
            module_name=f"sequential_{uuid4().hex}",
        )
        loader.load()

        arith = int_arithmetic(total_bits=bitwidth_inp, signed=is_signed)
        data_in = np.random.randint(low=arith.minimum_as_integer, high=arith.maximum_as_integer, size=1)

        seq = loader.get("sequential_0")
        out = loader.ffi().new(f"{out_type} *")
        for idx, inp in enumerate(data_in):
            ready = seq([inp], out)
            assert ready
            c_value = int(out[0])
            passed, reason = compare_values(int(inp), c_value)
            assert passed, f"index={idx}: input={inp}, python={inp}, c={c_value}: {reason}"
