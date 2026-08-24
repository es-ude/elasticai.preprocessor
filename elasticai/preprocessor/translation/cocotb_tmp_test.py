from pathlib import Path

import pytest

from .cocotb_tmp import temporary_directory


def test_success_removes_tmpdir(tmp_path: Path) -> None:
    backup = tmp_path / "backup"
    captured_tmpdir = None

    with temporary_directory(backup=backup) as tmpdir:
        captured_tmpdir = tmpdir
        (tmpdir / "data.txt").write_text("hello")
        assert tmpdir.exists()

    assert not captured_tmpdir.exists()


def test_success_removes_existing_backup(tmp_path: Path) -> None:
    backup = tmp_path / "backup"
    backup.mkdir()
    (backup / "old.txt").write_text("leftover")

    with temporary_directory(backup=backup) as tmpdir:
        (tmpdir / "data.txt").write_text("hello")

    assert not backup.exists()


def test_success_do_parent_removes_backup_parent(tmp_path: Path) -> None:
    parent = tmp_path / "parent_dir"
    backup = parent / "backup"
    backup.mkdir(parents=True)
    (backup / "old.txt").write_text("leftover")

    with temporary_directory(backup=backup, do_parent=True) as tmpdir:
        (tmpdir / "data.txt").write_text("hello")

    assert not parent.exists()


def test_exception_copies_contents_to_backup(tmp_path: Path) -> None:
    backup = tmp_path / "backup"
    with pytest.raises(RuntimeError, match="boom"):
        with temporary_directory(backup=backup) as tmpdir:
            (tmpdir / "data.txt").write_text("important content")
            raise RuntimeError("boom")

    assert backup.exists()
    assert (backup / "data.txt").read_text() == "important content"


def test_exception_copies_subdirectories(tmp_path: Path) -> None:
    backup = tmp_path / "backup"

    with pytest.raises(ValueError):
        with temporary_directory(backup=backup) as tmpdir:
            (tmpdir / "sub").mkdir()
            (tmpdir / "sub" / "nested.txt").write_text("nested content")
            raise ValueError("fail")

    assert (backup / "sub" / "nested.txt").read_text() == "nested content"


def test_exception_reraises_original_exception(tmp_path: Path) -> None:
    backup = tmp_path / "backup"

    with pytest.raises(KeyError):
        with temporary_directory(backup=backup):
            raise KeyError("some_key")
