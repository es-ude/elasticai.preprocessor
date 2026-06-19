from pathlib import Path


def get_path_to_project(new_folder: str = "") -> Path:
    """Function for getting the root path to of the project
    :param new_folder:  String with new folder
    :return:            Path with absolute path to entry point of the project
    """
    max_levels = 5
    cwd = Path(".").absolute()
    current = cwd

    def is_project_root(p):
        return (p / "pyproject.toml").exists()

    for _ in range(max_levels):
        if is_project_root(current):
            return current / new_folder
        current = current.parent

    if is_project_root(current):
        return current / new_folder
    return cwd
