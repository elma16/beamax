"""Keep runtime and distribution version metadata in sync."""

from pathlib import Path
import tomllib

import beamax


def test_runtime_version_matches_project_metadata():
    root = Path(__file__).resolve().parents[1]
    with (root / "pyproject.toml").open("rb") as handle:
        metadata = tomllib.load(handle)

    assert beamax.__version__ == metadata["project"]["version"] == "0.2.2"
