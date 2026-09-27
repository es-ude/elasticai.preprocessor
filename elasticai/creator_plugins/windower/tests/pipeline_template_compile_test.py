"""Compile-test for pipeline_template.h.

Instantiates DEF_PIPELINE_IMPL with a concrete 1st-order IIR + polyphase
decimation + windower configuration, compiles the resulting C program with
gcc/cc, runs it, and checks that the expected number of output windows is
produced (verified via assert inside the C program itself).
"""
import subprocess
from pathlib import Path
from shutil import which

import pytest

pytestmark = pytest.mark.skipif(which("cc") is None, reason="requires a C compiler")

# ---------------------------------------------------------------------------
# Locate template directories relative to this file
# ---------------------------------------------------------------------------
_PLUGINS = Path(__file__).parents[2]   # elasticai/creator_plugins/
_IIR_INC     = _PLUGINS / "filter_data"  / "c"
_POLY_INC    = _PLUGINS / "datarate" / "c"
_WINDOWER_INC = _PLUGINS / "windower"   / "c"

_C_SOURCE = """\
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include "pipeline_template.h"

/*
 * 1st-order IIR low-pass  (coeff_lgth=2, tap_lgth=1)
 *   coeff layout: [a0_unused, a1, b0, b1]
 *   y[n] = 0.5*x[n] + 0.5*x[n-1] - 0.0*y[n-1]
 * Polyphase decimation:  dsr = 4
 * Windower:              wl=8, nshift=8  (non-overlapping)
 *
 * Expected windows from 4*8*4=128 input samples: exactly 4.
 */
DEF_PIPELINE_IMPL(0, int32_t,
    2,    /* coeff_lgth */
    1,    /* tap_lgth   */
    4,    /* dsr        */
    8,    /* wl         */
    8,    /* nshift     */
    0.0, 0.0,   /* a: a0 (unused), a1 */
    0.5, 0.5    /* b: b0, b1          */
)

int main(void) {
    int32_t window[8];
    int windows_received = 0;

    for (int i = 0; i < 4 * 8 * 4; i++) {
        int32_t sample = (int32_t)(i % 64 - 32);
        if (calc_pipeline_0(sample, window)) {
            windows_received++;
        }
    }

    assert(windows_received == 4);
    printf("OK: %d windows\\n", windows_received);
    return 0;
}
"""


def test_pipeline_template_compiles_and_runs(tmp_path: Path) -> None:
    src = tmp_path / "pipeline_test.c"
    exe = tmp_path / "pipeline_test"

    src.write_text(_C_SOURCE)

    cc = which("gcc") or which("cc")
    compile_result = subprocess.run(
        [
            cc, "-Wall", "-Wextra",
            f"-I{_IIR_INC}",
            f"-I{_POLY_INC}",
            f"-I{_WINDOWER_INC}",
            str(src),
            "-o", str(exe),
        ],
        capture_output=True,
        text=True,
    )
    assert compile_result.returncode == 0, (
        f"Compilation failed:\n{compile_result.stderr}"
    )

    run_result = subprocess.run([str(exe)], capture_output=True, text=True)
    assert run_result.returncode == 0, (
        f"Runtime assertion failed:\n{run_result.stdout}\n{run_result.stderr}"
    )
    assert "OK: 4 windows" in run_result.stdout
