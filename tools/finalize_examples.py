#!/usr/bin/env python3
"""Generate and sync the public examples' Colab notebooks."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path
from textwrap import dedent

from example_metadata import (
    EXAMPLE_REVISION,
    GITHUB_REPO,
    colab_url,
    example_info_from_text,
)

PUBLIC_EXAMPLES_ROOT = Path("examples")


def banner_md(
    rel_nb: str,
    title: str,
    description: str,
) -> list[str]:
    return [
        f"# {title}\n",
        "\n",
        description + "\n",
        "\n",
        f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_url(rel_nb)})\n",
    ]


def install_cell_source(extras: list[str]) -> list[str]:
    extras_spec = f"[{','.join(extras)}]" if extras else ""
    return [
        "# Install beamax for Google Colab. Safe to skip when running locally.\n",
        f'%pip install --quiet "beamax{extras_spec} @ git+https://github.com/{GITHUB_REPO}.git@{EXAMPLE_REVISION}"',
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


def example_cells(src: str) -> list[tuple[str, list[str]]]:
    """Split the marked setup and main() body into sequential Colab cells."""
    if not re.search(r"^# %% ", src, re.MULTILINE):
        return [("code", notebook_code_source(src))]

    module = ast.parse(src)
    main = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    lines = src.splitlines(keepends=True)
    start = module.body[0].end_lineno if ast.get_docstring(module) else 0
    setup = "".join(lines[start : main.lineno - 1])
    body = dedent("".join(lines[main.lineno : main.end_lineno]))
    sections = re.split(r"^# %% (.+)\n", setup + body, flags=re.MULTILINE)
    cells = []
    if preamble := sections[0].strip():
        cells.append(("code", notebook_code_source(preamble)))
    for title, code in zip(sections[1::2], sections[2::2]):
        # Leading prose belongs with the heading in Colab. Keep comments
        # attached to individual statements inside the executable cell.
        lines = code.strip().splitlines()
        notes = []
        while lines and lines[0].startswith("# "):
            notes.append(lines.pop(0)[2:])
        heading = f"## {title}"
        if notes:
            heading += "\n\n" + " ".join(notes)
        cells.append(("markdown", notebook_code_source(heading)))
        cells.append(("code", notebook_code_source("\n".join(lines).strip())))
    return cells


def sync_notebook(py_path: Path, *, check: bool = False) -> bool:
    """Generate missing notebooks and refresh the ones managed by this tool."""
    nb_path = py_path.with_suffix(".ipynb")
    src = py_path.read_text()
    info = example_info_from_text(src)
    if not info.notebook:
        return False
    if nb_path.exists():
        old = json.loads(nb_path.read_text())
        if not (
            old["metadata"].get("beamax_generated") or is_thin_generated_notebook(old)
        ):
            return False

    content = [
        (
            "markdown",
            banner_md(str(nb_path), title_from_path(py_path), info.description),
        ),
        ("code", install_cell_source(list(info.extras))),
        *example_cells(src),
    ]
    cells = []
    for index, (kind, source) in enumerate(content):
        cell = {
            "cell_type": kind,
            "id": cell_id(nb_path, index),
            "metadata": {},
            "source": source,
        }
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    nb = {
        "cells": cells,
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
    if len(cells) > 3:
        nb["metadata"]["beamax_generated"] = True
    new_text = json.dumps(nb, indent=1, ensure_ascii=False, sort_keys=True) + "\n"
    if nb_path.exists() and nb_path.read_text() == new_text:
        return False
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

    changed = sum(sync_notebook(path, check=args.check) for path in scripts)
    if args.check:
        if changed:
            raise SystemExit(
                f"{changed} example notebooks need tools/finalize_examples.py"
            )
        print("example notebooks are up to date")
    else:
        print(f"notebooks updated: {changed}")


if __name__ == "__main__":
    main()
