from pathlib import Path
from uuid import uuid4

import numpy as np
from elasticai.equichecker import CompileLoader, compare_values

from elasticai.preprocessor.filter import Filtering, SettingsFilter
from elasticai.preprocessor.translation.cocotb_tmp import temporary_directory


def check_filter_c_equivalence(
    settings: SettingsFilter,
    tmp_path: Path,
    source_name: str,
    function_name: str,
    bitwidth: int,
    numpy_dtype: type[np.generic],
    c_type: str,
    data_in: np.ndarray,
) -> None:
    filtering = Filtering(settings)
    backup = tmp_path / f"simulation_{bitwidth}_{c_type}"
    backup.mkdir(parents=True, exist_ok=True)
    with temporary_directory(backup) as tmpdir:
        output_dir = tmpdir / "src"
        filtering.create_design("mcu", bitwidth, "0", output_dir, signed=True)

        for file in output_dir.glob("*"):
            print(file)
        adapter = tmpdir / "adapter.h"
        adapter.write_text(f"_Bool {function_name}({c_type} data, {c_type} *out);\n")

        loader = CompileLoader(
            headers=str(adapter),
            sources=[str(output_dir / source_name)],
            build_dir=str(tmpdir / "cffi-build"),
            module_name=f"filter_{uuid4().hex}",
        )
        loader.load()
        c_filter = loader.get(function_name)
        out = loader.ffi().new(f"{c_type} *")

        expected = filtering.filt(data_in.astype(float)).astype(numpy_dtype).tolist()
        data_out = list()
        checked = list()
        for val_in, val_ref in zip(data_in.tolist(), expected):
            c_filter(int(val_in), out)
            data_out.append(int(out[0]))
            passed, _ = compare_values(val_ref, data_out[-1])
            checked.append(passed)

        passed = all(checked)
        if not passed:
            print("Input:", data_in.tolist())
            print("Output:", data_out)
            print("Expected:", expected)
        assert passed
