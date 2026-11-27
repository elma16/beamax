"""Parse metadata from public example module docstrings."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


FALSE_VALUES = {"0", "false", "no", "off"}
GITHUB_REPO = "elma16/beamax"
EXAMPLE_REVISION = "v0.4.0"


def colab_url(notebook: str) -> str:
    return f"https://colab.research.google.com/github/{GITHUB_REPO}/blob/{EXAMPLE_REVISION}/{notebook}"


@dataclass(frozen=True)
class ExampleInfo:
    """Metadata shared by the example runner and gallery generators."""

    description: str = ""
    extras: tuple[str, ...] = ()
    smoke: bool = True
    notebook: bool = True


def example_info_from_text(text: str) -> ExampleInfo:
    """Parse an example's description and ``Example key: value`` fields."""
    try:
        docstring = ast.get_docstring(ast.parse(text))
    except SyntaxError:
        return ExampleInfo()
    if not docstring:
        return ExampleInfo()

    metadata: dict[str, str] = {}
    for line in docstring.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.startswith("Example "):
            metadata[key.removeprefix("Example ").strip().lower()] = value.strip()

    extras = tuple(
        item.strip() for item in metadata.get("extras", "").split(",") if item.strip()
    )
    return ExampleInfo(
        description=docstring.splitlines()[0].strip(),
        extras=extras,
        smoke=metadata.get("smoke", "true").lower() not in FALSE_VALUES,
        notebook=metadata.get("notebook", "true").lower() not in FALSE_VALUES,
    )


def read_example_info(path: Path) -> ExampleInfo:
    """Read and parse metadata from an example script."""
    return example_info_from_text(path.read_text())
