from contextlib import contextmanager
from pathlib import Path
from shutil import copytree, rmtree
from tempfile import mkdtemp
from typing import Iterator


@contextmanager
def temporary_directory(backup: Path) -> Iterator[Path]:
    tmpdir = Path(mkdtemp())
    try:
        yield tmpdir
    except:
        copytree(src=tmpdir, dst=backup, dirs_exist_ok=True)
        raise
    else:
        rmtree(tmpdir, ignore_errors=True)
        rmtree(backup, ignore_errors=True)
