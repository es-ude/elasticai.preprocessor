pytest_plugins = ["elasticai.creator.testing"]


def pytest_sessionstart(session):
    from elasticai.preprocessor import get_path_to_project
    from shutil import rmtree

    path2temp = get_path_to_project("build_test")
    if path2temp.exists():
        rmtree(path2temp, ignore_errors=True)
    path2temp.mkdir(parents=True, exist_ok=True)


def pytest_sessionfinish(session, exitstatus):
    from elasticai.preprocessor import get_path_to_project
    from pathlib import Path

    def remove_empty_dirs(path, min_level: int = 3) -> None:
        from os import walk

        path = Path(path)
        if not path.exists():
            return

        root_depth = len(path.parts)
        for dirpath, _, _ in walk(path, topdown=False):
            folder = Path(dirpath)
            level = len(folder.parts) - root_depth
            if level < min_level:
                continue
            try:
                folder.rmdir()
            except OSError:
                pass

    path2check = get_path_to_project("build_test")
    remove_empty_dirs(path2check)
