#!/usr/bin/env python3
"""
Finalise the restored example tree:
1. Add a heuristic module docstring to any .py that lacks one.
2. For every .py without a paired .ipynb, generate a thin Colab-runnable
   notebook (markdown banner + install cell + single big code cell with the
   full .py source).
3. For every existing .ipynb without an Open-in-Colab badge, prepend the
   banner + install cell (matches the pattern already used in the kept set).
4. Add an extra warning markdown cell for memory-heavy and OA-Breast-dependent
   examples.

Idempotent: skips files that already have what's needed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

GITHUB_REPO = "elma16/beamax"
GITHUB_BRANCH = "main"
PUBLIC_EXAMPLE_ROOTS = {
    "benchmarks",
    "bowtie",
    "decomp",
    "forward",
    "rays",
    "reconstruction",
    "singleGB",
}

# Examples that load the OA-Breast phantom from disk. Notebook gets a
# data-download instructions banner.
OABREAST_DEPENDENT = {
    "examples/forward/forward-3d.py",
    "examples/reconstruction/time-reversal/fwd-tr-img.py",
    "examples/reconstruction/comparison/breast_tr_adj_faces.py",
}

# Examples that need substantial RAM (gated behind BEAMAX_FULL_EXAMPLES,
# 3D, or large grids). Notebook gets a memory-warning banner.
MEMORY_HEAVY = {
    "examples/forward/forward-3d.py",
    "examples/reconstruction/time-reversal/TR-3d2.py",
    "examples/reconstruction/adjoint/autodiff-3d.py",
}


def is_public_example(path: Path) -> bool:
    return len(path.parts) >= 2 and path.parts[1] in PUBLIC_EXAMPLE_ROOTS


def heuristic_description(path: Path) -> str:
    """Best-effort 1-2 sentence summary derived from filename and folder."""
    parts = path.parts
    stem = path.stem.replace("_", " ").replace("-", " ")
    folder = parts[-2] if len(parts) >= 2 else ""

    # Manually-curated mappings for common patterns
    s = stem.lower()
    if "singlegb" in s.replace(" ", ""):
        topic = stem.replace("singleGB", "").replace("singlegb", "").strip()
        if "absorption" in s:
            return "Single Gaussian beam propagation with viscous absorption, comparing the analytic decay against MSGB."
        if "convergence" in s:
            return "Single Gaussian beam convergence study against a spectral reference solution as the grid is refined."
        if "energy" in s:
            return "Track energy conservation along a single Gaussian beam trajectory."
        if "plot" in s:
            return "Visualise a single Gaussian beam's amplitude and ellipse over time."
        if "propagation" in s:
            return "Propagate a single Gaussian beam through a homogeneous medium."
        if "rayleigh" in s:
            return "Single Gaussian beam Rayleigh-range / focal-error diagnostic."
        if "wpt" in s:
            return "Single Gaussian beam wave-packet transform diagnostic."
        return f"Single Gaussian beam example: {topic}."
    if "ray" in folder:
        if "anim" in s:
            return "Animation of ray trajectories propagating through a 2D/3D medium."
        if "stiffness" in s:
            return "Investigate the stiffness of the Gaussian beam ray ODEs."
        if "optim" in s and "focus" in s:
            return "Optimise initial ray parameters to focus at a target location."
        if "colour" in s:
            return "Coloured-by-amplitude ray-trajectory visualisation."
        return "Ray-tracing diagnostic for the Gaussian beam Hamiltonian."
    if "bowtie" in folder:
        if "angle_mapping" in s:
            return "Bowtie sensor configuration: map sensor angles to beam directions."
        if "aliasing" in s:
            return "Aliasing diagnostic for a 3D bowtie sensor configuration."
        return f"Bowtie sensor geometry test ({stem})."
    if "3d_replot" in s.replace(" ", "_"):
        return "Replot saved 3D forward / TR / adjoint outputs (requires saved .npy outputs and OA-Breast phantom paths)."
    return f"Example: {stem}."


def has_module_docstring(text: str) -> bool:
    lines = text.splitlines()
    for line in lines:
        s = line.lstrip()
        if not s or s.startswith("#"):
            continue
        return s.startswith('"""') or s.startswith("'''")
    return False


def insert_docstring(path: Path) -> None:
    text = path.read_text()
    if has_module_docstring(text):
        return
    desc = heuristic_description(path)
    lines = text.splitlines(keepends=True)
    insert_at = 0
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("#!") or s.startswith("# coding") or s == "":
            insert_at = i + 1
            continue
        break
    new_lines = lines[:insert_at] + [f'"""\n{desc}\n"""\n'] + lines[insert_at:]
    path.write_text("".join(new_lines))


def colab_url(rel_nb: str) -> str:
    return f"https://colab.research.google.com/github/{GITHUB_REPO}/blob/{GITHUB_BRANCH}/{rel_nb}"


def banner_md(rel_nb: str, title: str, oabreast: bool, memory_heavy: bool) -> list[str]:
    lines = [
        f"# {title}\n",
        "\n",
        f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_url(rel_nb)})\n",
        "\n",
        "> Select **Runtime → Change runtime type → GPU or TPU** in Colab to demonstrate the hardware-acceleration story.\n",
    ]
    if memory_heavy:
        lines += [
            "\n",
            "> **Note:** this example is memory-heavy. Run it on a machine with plenty of RAM (and / or a high-memory accelerator).\n",
        ]
    if oabreast:
        lines += [
            "\n",
            "> **Data download required.** This example loads the OA-Breast phantom from `data/NumericalBreastPhantoms-selected/hdf5/Neg_07_Left.h5`. Download it manually from the [OA-Breast database](https://anastasio.bioengineering.illinois.edu/downloadable-content/oa-breast-database/) (Illinois) and place it under `<repo-root>/data/`. The notebook will error at the load step until the file is present.\n",
        ]
    return lines


def install_cell_source(needs_kwave: bool) -> list[str]:
    extras = "[kwave]" if needs_kwave else ""
    return [
        "# Install beamax for Google Colab. Safe to skip when running locally.\n",
        "%%capture\n",
        f'%pip install --quiet "beamax{extras} @ git+https://github.com/{GITHUB_REPO}.git"\n',
    ]


def _needs_kwave(text: str) -> bool:
    return bool(re.search(r"\bfrom\s+kwave\b|\bimport\s+kwave\b|\bKWaveSolver\b", text))


def title_from_path(path: Path) -> str:
    return path.stem.replace("_", " ").replace("-", " ").strip()


def already_has_badge(nb: dict) -> bool:
    for c in nb.get("cells", [])[:3]:
        if "colab.research.google.com/assets/colab-badge.svg" in "".join(
            c.get("source", [])
        ):
            return True
    return False


def add_badge_to_existing_nb(path: Path) -> bool:
    nb = json.loads(path.read_text())
    if already_has_badge(nb):
        return False
    full_text = "\n".join(
        "".join(c.get("source", []))
        for c in nb["cells"]
        if c.get("cell_type") == "code"
    )
    rel = str(path)
    title = title_from_path(path)
    oabreast = rel in OABREAST_DEPENDENT
    memory_heavy = rel in MEMORY_HEAVY
    needs_kwave = _needs_kwave(full_text)
    banner = {
        "cell_type": "markdown",
        "metadata": {},
        "source": banner_md(rel, title, oabreast, memory_heavy),
    }
    install = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": install_cell_source(needs_kwave),
    }
    nb["cells"] = [banner, install] + nb["cells"]
    path.write_text(json.dumps(nb, indent=1) + "\n")
    return True


def generate_nb_from_py(py_path: Path) -> bool:
    nb_path = py_path.with_suffix(".ipynb")
    if nb_path.exists():
        return False
    src = py_path.read_text()
    rel_nb = str(nb_path)
    title = title_from_path(py_path)
    oabreast = str(py_path) in OABREAST_DEPENDENT
    memory_heavy = str(py_path) in MEMORY_HEAVY
    needs_kwave = _needs_kwave(src)

    src_lines = src.splitlines(keepends=True)
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": banner_md(rel_nb, title, oabreast, memory_heavy),
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": install_cell_source(needs_kwave),
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": src_lines,
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
    nb_path.write_text(json.dumps(nb, indent=1) + "\n")
    return True


def main() -> None:
    root = Path("examples")

    docs_added = 0
    for p in sorted(root.rglob("*.py")):
        if "__pycache__" in p.parts or not is_public_example(p):
            continue
        if not has_module_docstring(p.read_text()):
            insert_docstring(p)
            docs_added += 1
    print(f"docstrings added: {docs_added}")

    badges_added = 0
    for p in sorted(root.rglob("*.ipynb")):
        if "__pycache__" in p.parts or not is_public_example(p):
            continue
        if add_badge_to_existing_nb(p):
            badges_added += 1
    print(f"badges added to existing notebooks: {badges_added}")

    nbs_generated = 0
    for p in sorted(root.rglob("*.py")):
        if "__pycache__" in p.parts or not is_public_example(p):
            continue
        if generate_nb_from_py(p):
            nbs_generated += 1
    print(f"notebooks generated: {nbs_generated}")


if __name__ == "__main__":
    main()
