from equi_checker.src.loader import PrecompiledLoader, CompileLoader


# --- python implementation ---
def add(a, b):
    return a + b


# --- precompiled version ---

loader = PrecompiledLoader(
    library_path="./c_funcs/libadd.so", headers="./c_funcs/add.h"
)

lib = loader.load()  # returns the loaded shared library

c_add = loader.get("add")  # alternatively: lib.add


def test_equivalence_precompiled():
    a = 5
    b = 7
    assert add(a, b) == c_add(a, b)


# --- compiled version ---

compile_loader = CompileLoader(
    headers="/home/benni/code/uni/eeg/eeg-software/function_equivalence_test/c_funcs/add.h",
    sources=[
        "/home/benni/code/uni/eeg/eeg-software/function_equivalence_test/c_funcs/add.c"
    ],
    build_dir="/home/benni/code/uni/eeg/eeg-software/function_equivalence_test/c_funcs/build_compile_loader",
)

lib2 = compile_loader.load()  # compiles the C code and returns a library

c_add2 = compile_loader.get("add")  # alternatively: lib2.add


def test_equivalence_compiled():
    a = 10
    b = 15
    assert add(a, b) == c_add2(a, b)
