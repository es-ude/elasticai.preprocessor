from pathlib import Path


def get_path_to_project(new_folder: str = "", max_levels: int = 5) -> Path:
    """Function for getting the root path to of the project
    :param new_folder:  New folder path
    :param max_levels:  Max number of levels to get-out for finding pyproject.toml
    :return:            Path with absolute path to entry point of the project
    """
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
