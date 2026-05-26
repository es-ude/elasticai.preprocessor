import importlib.util
import os
from cffi import FFI


class Loader:
    def __init__(self):
        self._ffi = FFI()  # FFI is the C Foreign Function Interface from cffi
        self._lib = None

    def load(self):
        raise NotImplementedError

    def ffi(self):
        self.ensure_loaded()
        return self._ffi

    def lib(self):
        self.ensure_loaded()
        return self._lib

    def get(self, name):
        self.ensure_loaded()
        return getattr(self._lib, name)

    def has(self, name):
        self.ensure_loaded()
        return hasattr(self._lib, name)

    def functions(self):
        self.ensure_loaded()
        return [name for name in dir(self._lib) if callable(getattr(self._lib, name))]

    def ensure_loaded(self):
        if self._lib is None:
            raise RuntimeError("C library not loaded. Call load() first.")


class CompileLoader(Loader):
    """Loader that compiles C source files using cffi in API mode."""

    def __init__(
        self,
        headers,
        sources,
        build_dir,
        module_name="cffi_module",
    ):
        """
        Parameters:
            headers: str or list of str
                Relative path(s) to C header file(s).
            sources: list of str
                List of relative paths to C source files.
            module_name: str
                Name of the generated C extension module.
            build_dir: str
                Directory to use for building the module.
        """
        super().__init__()
        # make relative paths for headers absolute based on the current working directory
        self._headers = (
            [os.path.abspath(hdr) for hdr in headers]
            if isinstance(headers, list)
            else os.path.abspath(headers)
        )
        # make relative paths for sources absolute based on the current working directory
        self._sources = [os.path.abspath(src) for src in sources]
        self._module_name = module_name
        self._build_dir = build_dir

    def load(self):
        """Compiles the C sources and loads the resulting library.

        Returns:
            The loaded C library.
        """
        # Read and merge header files; pass to cffi (required for python to know function signatures)
        header_source = self._read_headers(self._headers)
        self._ffi.cdef(header_source)

        # Set source for cffi (required to compile the C code)
        include_lines = self._include_header_includes(self._headers)
        self._ffi.set_source(
            self._module_name,
            include_lines,
            sources=self._sources,
        )

        build_dir = self._build_dir
        if build_dir:
            os.makedirs(build_dir, exist_ok=True)

        module_path = self._ffi.compile(verbose=False, tmpdir=build_dir)
        module = self._import_module_from_path(module_path)
        self._ffi = module.ffi
        self._lib = module.lib
        return self._lib

    def _read_headers(self, headers):
        """Reads header content from a file path or list of file paths."""
        if isinstance(headers, str):
            if os.path.exists(headers):
                return self._read_file(headers)
            else:
                raise RuntimeError(f"Header file not found: {headers}")
        return "\n".join(self._read_file(path) for path in headers)

    def _include_header_includes(self, headers):
        """Generates #include directives for header files."""
        if isinstance(headers, str):
            if os.path.exists(headers):
                return f'#include "{headers}"'
            return ""
        return "\n".join(f'#include "{path}"' for path in headers)

    def _read_file(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _import_module_from_path(self, module_path):
        """Dynamically imports a compiled module from the given file path."""
        spec = importlib.util.spec_from_file_location(self._module_name, module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load compiled module: {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


class PrecompiledLoader(Loader):
    """Loader that loads a precompiled C library using cffi in ABI mode."""

    def __init__(self, library_path, headers):
        """
        Parameters:
            library_path: str
                Absolute path to the precompiled shared library.
            headers: str or list of str
                Absolute path(s) to C header file(s).
        """
        super().__init__()
        self._library_path = library_path
        self._headers = headers

    def load(self):
        """Loads the precompiled C library.
        Returns:
            The loaded C library.
        """
        header_source = self._read_headers(self._headers)
        self._ffi.cdef(header_source)
        self._lib = self._ffi.dlopen(self._library_path)
        return self._lib

    def _read_headers(self, headers):
        if isinstance(headers, str):
            if os.path.exists(headers):
                return self._read_file(headers)
            else:
                raise RuntimeError(f"Header file not found: {headers}")
        return "\n".join(self._read_file(path) for path in headers)

    def _read_file(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
