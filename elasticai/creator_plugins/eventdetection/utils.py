from pathlib import Path
from typing import Any

from elasticai.preprocessor import get_path_to_project
from elasticai.preprocessor.translation import load_and_build_form_plugin


def load_and_plugin(
        type: str,
        id: str,
        params: dict[str, Any],
        packages: list,
        path2save: Path = get_path_to_project() / "build",
) -> None:
    load_and_build_form_plugin(type, id, params, packages, path2save)