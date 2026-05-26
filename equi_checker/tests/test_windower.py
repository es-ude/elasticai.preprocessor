from equi_checker.src.loader import CompileLoader
import numpy as np


class PyWindower:
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.buffer = []

    def push(self, sample: float) -> None:
        if len(self.buffer) < self.window_size:
            self.buffer.append(sample)
        else:
            self.buffer = self.buffer[1:] + [sample]

    def is_full(self) -> bool:
        return len(self.buffer) >= self.window_size

    def get_window(self):
        if not self.is_full():
            return 0, None
        return 1, list(self.buffer)

    def reset(self) -> None:
        self.buffer = []


def test_windower_simple_equivalence():
    loader = CompileLoader(
        headers="./c_funcs/windower/windower.h",
        sources=["./c_funcs/windower/windower.c"],
        build_dir="build_test_windower",
        module_name="c_windower",
    )
    loader.load()

    # Get C functions and ffi
    windower_init = loader.get("windower_init")
    windower_push = loader.get("windower_push")
    windower_is_full = loader.get("windower_is_full")
    windower_get_window = loader.get("windower_get_window")
    loader.get("windower_reset")
    ffi = loader.ffi()

    # Initialize C windower and Python windower with the same window size
    window_size = 5
    c_buffer = np.zeros(window_size, dtype=np.float32)
    c_out = np.zeros(window_size, dtype=np.float32)

    # Cast numpy arrays to C pointers
    c_buffer_ptr = ffi.cast("float*", c_buffer.ctypes.data)
    c_out_ptr = ffi.cast("float*", c_out.ctypes.data)
    c_w = ffi.new("Windower *")

    windower_init(c_w, c_buffer_ptr, window_size)

    py_w = PyWindower(window_size)

    # Push values into both windower instances and compare their states
    for value in [1.0, 2.0, 3.0, 4.0, 5.0]:
        py_w.push(value)
        windower_push(c_w, value)

    assert py_w.is_full() == bool(windower_is_full(c_w))

    py_ready, py_window = py_w.get_window()
    c_ready = windower_get_window(c_w, c_out_ptr)
    c_window = c_out.tolist()

    assert py_ready == c_ready
    assert np.allclose(py_window, c_window)


def test_windower_reset_equivalence():
    loader = CompileLoader(
        headers="./c_funcs/windower/windower.h",
        sources=["./c_funcs/windower/windower.c"],
        build_dir="build_test_windower",
        module_name="c_windower",
    )
    loader.load()

    # Get C functions and ffi
    windower_init = loader.get("windower_init")
    windower_push = loader.get("windower_push")
    windower_is_full = loader.get("windower_is_full")
    windower_get_window = loader.get("windower_get_window")
    windower_reset = loader.get("windower_reset")
    ffi = loader.ffi()

    # Initialize C windower and Python windower with the same window size
    window_size = 5
    c_buffer = np.zeros(window_size, dtype=np.float32)
    c_out = np.zeros(window_size, dtype=np.float32)

    # Cast numpy arrays to C pointers
    c_buffer_ptr = ffi.cast("float*", c_buffer.ctypes.data)
    c_out_ptr = ffi.cast("float*", c_out.ctypes.data)
    c_w = ffi.new("Windower *")

    # Initialize both windower instances
    windower_init(c_w, c_buffer_ptr, window_size)
    py_w = PyWindower(window_size)

    # Push values into both windower instances
    for value in [1.0, 2.0, 3.0, 4.0, 5.0]:
        py_w.push(value)
        windower_push(c_w, value)

    # Verify both are full
    assert py_w.is_full() == bool(windower_is_full(c_w))

    # Push more values to ensure the window updates correctly
    for value in [6.0, 7.0]:
        py_w.push(value)
        windower_push(c_w, value)

    py_ready_2, py_window_2 = py_w.get_window()
    c_ready_2 = windower_get_window(c_w, c_out_ptr)
    c_window_2 = c_out.tolist()

    assert py_ready_2 == c_ready_2
    assert np.allclose(py_window_2, c_window_2)

    py_w.reset()
    windower_reset(c_w)

    assert py_w.is_full() == bool(windower_is_full(c_w))
