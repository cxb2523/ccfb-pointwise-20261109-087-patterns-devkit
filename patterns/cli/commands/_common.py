from pathlib import Path
from typing import Any

import yaml
from typer import Argument

app_argument_help = (
    "The slug or uid of an app or app version, or the path to an app's graph.yml"
)
app_argument = Argument(None, help=app_argument_help, show_default=False)

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


def load_yaml(path: Path) -> Any:
    """Load a yaml file from disk.

    Raises ``yaml.YAMLError`` on syntax errors; callers are responsible for
    translating that into the CLI's standard error message and exit code.
    """
    with Path(path).open() as f:
        return yaml.load(f.read(), Loader=Loader)
