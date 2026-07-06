from elasticai.preprocessor import get_path_to_project


def test_path_to_project() -> None:
    checks = ["elasticai", "preprocessor"]
    rslt = get_path_to_project()

    assert rslt.is_dir()
    assert rslt.parts[-1] == f"{checks[0]}.{checks[1]}"


def test_path_to_project_ref() -> None:
    checks = ["elasticai", "preprocessor", "test"]
    rslt = get_path_to_project(new_folder=checks[2])

    assert not rslt.exists()
    assert rslt.parts[-1] == f"{checks[2]}"
    assert rslt.parts[-2] == f"{checks[0]}.{checks[1]}"
