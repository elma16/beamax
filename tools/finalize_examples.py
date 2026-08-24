#!/usr/bin/env python3
"""Generate and sync the public examples' Colab notebooks."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from example_metadata import example_info_from_text

GITHUB_REPO = "elma16/beamax"
GITHUB_BRANCH = "main"
PUBLIC_EXAMPLES_ROOT = Path("examples")


def colab_url(rel_nb: str) -> str:
    return f"https://colab.research.google.com/github/{GITHUB_REPO}/blob/{GITHUB_BRANCH}/{rel_nb}"


def banner_md(
    rel_nb: str,
    title: str,
) -> list[str]:
    return [
        f"# {title}\n",
        "\n",
        f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_url(rel_nb)})\n",
    ]


def install_cell_source(extras: list[str]) -> list[str]:
    extras_spec = f"[{','.join(extras)}]" if extras else ""
    return [
        "# Install beamax for Google Colab. Safe to skip when running locally.\n",
        "%%capture\n",
        f'%pip install --quiet "beamax{extras_spec} @ git+https://github.com/{GITHUB_REPO}.git"',
    ]


def cell_id(path: Path, index: int) -> str:
    """Deterministic notebook cell id stable across regeneration."""
    digest = hashlib.sha1(f"{path}:{index}".encode()).hexdigest()
    return digest[:24]


def notebook_code_source(text: str) -> list[str]:
    """Return source lines in the same style Ruff uses for notebook cells."""
    lines = text.splitlines(keepends=True)
    if lines and lines[-1].endswith("\n"):
        lines[-1] = lines[-1][:-1]
    return lines


def title_from_path(path: Path) -> str:
    title = path.stem.replace("_", " ").replace("-", " ").strip()
    title = re.sub(r"\b(\d+)d\b", lambda match: f"{match.group(1)}D", title)
    return title


def generate_nb_from_py(py_path: Path, *, check: bool = False) -> bool:
    """Generate a missing public notebook, or report that it is missing."""
    nb_path = py_path.with_suffix(".ipynb")
    if nb_path.exists():
        return False
    src = py_path.read_text()
    info = example_info_from_text(src)
    if not info.notebook:
        return False
    rel_nb = str(nb_path)
    title = title_from_path(py_path)

    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "id": cell_id(nb_path, 0),
                "metadata": {},
                "source": banner_md(rel_nb, title),
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": cell_id(nb_path, 1),
                "metadata": {},
                "outputs": [],
                "source": install_cell_source(list(info.extras)),
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": cell_id(nb_path, 2),
                "metadata": {},
                "outputs": [],
                "source": notebook_code_source(src),
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    new_text = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
    if not check:
        nb_path.write_text(new_text)
    return True


def is_thin_generated_notebook(nb: dict) -> bool:
    """Return whether ``nb`` follows the generated 3-cell example pattern."""
    cells = nb.get("cells", [])
    if len(cells) != 3:
        return False
    if [c.get("cell_type") for c in cells] != ["markdown", "code", "code"]:
        return False
    install_source = "".join(cells[1].get("source", []))
    return "Install beamax for Google Colab" in install_source


def sync_generated_nb_from_py(py_path: Path, *, check: bool = False) -> bool:
    """Refresh a generated notebook, or report that it is stale."""
    nb_path = py_path.with_suffix(".ipynb")
    if not nb_path.exists():
        return False

    nb = json.loads(nb_path.read_text())
    if not is_thin_generated_notebook(nb):
        return False

    src = py_path.read_text()
    info = example_info_from_text(src)
    rel_nb = str(nb_path)
    nb["cells"][0] = {
        "cell_type": "markdown",
        "id": cell_id(nb_path, 0),
        "metadata": {},
        "source": banner_md(rel_nb, title_from_path(py_path)),
    }
    nb["cells"][1] = {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id(nb_path, 1),
        "metadata": {},
        "outputs": [],
        "source": install_cell_source(list(info.extras)),
    }
    nb["cells"][2] = {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id(nb_path, 2),
        "metadata": {},
        "outputs": [],
        "source": notebook_code_source(src),
    }
    new_text = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
    if nb_path.read_text() == new_text:
        return False
    if not check:
        nb_path.write_text(new_text)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if generated notebooks are stale without changing files",
    )
    args = parser.parse_args()

    scripts = [
        path
        for path in sorted(PUBLIC_EXAMPLES_ROOT.rglob("*.py"))
        if "__pycache__" not in path.parts
    ]

    nbs_synced = sum(
        sync_generated_nb_from_py(path, check=args.check) for path in scripts
    )
    nbs_generated = sum(generate_nb_from_py(path, check=args.check) for path in scripts)

    if args.check:
        if nbs_synced or nbs_generated:
            raise SystemExit(
                f"example notebooks are stale ({nbs_synced} changed, "
                f"{nbs_generated} missing); run tools/finalize_examples.py"
            )
        print("example notebooks are up to date")
    else:
        print(f"notebooks synced: {nbs_synced}")
        print(f"notebooks generated: {nbs_generated}")


if __name__ == "__main__":
    main()
