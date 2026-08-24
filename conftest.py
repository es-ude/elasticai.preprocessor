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

    path2check = get_path_to_project("build_test")
    def remove_empty_dirs(path: Path) -> None:
        if not path.exists():
            return
        for dirpath in path2check.iterdir():
            try:
                dirpath.rmdir()
            except:
                pass
    remove_empty_dirs(path2check)
