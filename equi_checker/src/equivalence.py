from __future__ import annotations

import math
from typing import Any, Tuple


def compare_values(
    py_val: Any,
    c_val: Any,
    abs_tol: float = 1e-6,
    rel_tol: float = 1e-6,
    nan_equal: bool = True,
    inf_equal: bool = True,
) -> Tuple[bool, str]:
    """
    Compare Python and C return values with configurable float tolerance. Works for bool, int, and float types, with special handling for NaN and Inf.

    Args:
        py_val: The value returned by the Python function.
        c_val: The value returned by the C function.
        abs_tol: Absolute tolerance for float comparison.
        rel_tol: Relative tolerance for float comparison.
        nan_equal: Whether to consider NaN values as equal.
        inf_equal: Whether to consider Inf values as equal (only if they have the same sign).

    Returns:
        (passed, reason)
    """

    # bool and int comparison
    if isinstance(py_val, bool) and isinstance(c_val, bool):
        ok = py_val is c_val
        return ok, "bool equal" if ok else f"bool mismatch: py={py_val}, c={c_val}"

    if isinstance(py_val, int) and isinstance(c_val, int):
        ok = py_val == c_val
        return ok, "int equal" if ok else f"int mismatch: py={py_val}, c={c_val}"

    # Float logic with NaN/Inf policy
    if isinstance(py_val, float) and isinstance(c_val, float):
        if math.isnan(py_val) or math.isnan(c_val):
            ok = nan_equal and math.isnan(py_val) and math.isnan(c_val)
            return ok, "both NaN" if ok else f"NaN mismatch: py={py_val}, c={c_val}"

        if math.isinf(py_val) or math.isinf(c_val):
            same_sign_inf = (
                math.isinf(py_val) and math.isinf(c_val) and (py_val > 0) == (c_val > 0)
            )
            ok = inf_equal and same_sign_inf
            return (
                ok,
                "both Inf (same sign)"
                if ok
                else f"Inf mismatch: py={py_val}, c={c_val}",
            )

        diff = abs(py_val - c_val)
        tol = max(abs_tol, rel_tol * max(abs(py_val), abs(c_val)))
        ok = diff <= tol
        if ok:
            return True, f"float within tolerance (diff={diff:.3e}, tol={tol:.3e})"
        return (
            False,
            f"float mismatch (diff={diff:.3e}, tol={tol:.3e}, py={py_val}, c={c_val})",
        )

    # Fallback exact comparison for same-type non-floats
    if type(py_val) is type(c_val):
        ok = py_val == c_val
        return (
            ok,
            "values equal" if ok else f"value mismatch: py={py_val!r}, c={c_val!r}",
        )

    return False, f"type mismatch: py={type(py_val).__name__}, c={type(c_val).__name__}"
