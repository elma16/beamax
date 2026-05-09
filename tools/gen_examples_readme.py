#!/usr/bin/env python3
"""
Regenerate examples/README.md as an authoritative index of every example,
grouped by directory, each with a one-line description (from the .py module
docstring) and an Open-in-Colab badge for the paired .ipynb if it exists.
"""

from __future__ import annotations

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

# Friendly section titles, ordered
GROUPS = [
    ("examples/decomp", "Frequency decomposition & MSWPT"),
    ("examples/forward", "Forward propagation"),
    ("examples/reconstruction/time-reversal", "Time-reversal reconstruction"),
    ("examples/reconstruction/adjoint", "Adjoint reconstruction"),
    ("examples/reconstruction/comparison", "Inverse-operator comparisons"),
    ("examples/benchmarks", "Benchmarks"),
    ("examples/singleGB", "Single Gaussian beam diagnostics"),
    ("examples/rays", "Ray tracing & Hamiltonian"),
    ("examples/bowtie", "Bowtie sensor configurations"),
]


def short_desc(py_path: Path) -> str:
    """Pull the first paragraph of the module docstring (one-line)."""
    text = py_path.read_text()
    m = re.search(r'^"""\s*\n(.*?)\n\s*"""', text, re.DOTALL | re.MULTILINE)
    if not m:
        m = re.search(r'^"""(.*?)"""', text, re.DOTALL | re.MULTILINE)
    if not m:
        return ""
    body = m.group(1).strip()
    # First line (or first sentence)
    first = body.split("\n", 1)[0].strip()
    return first


def colab_url(rel_nb: str) -> str:
    return f"https://colab.research.google.com/github/{GITHUB_REPO}/blob/{GITHUB_BRANCH}/{rel_nb}"


def gallery_entry(py_path: Path) -> str:
    nb_path = py_path.with_suffix(".ipynb")
    desc = short_desc(py_path)
    rel_py = str(py_path)
    rel_nb = str(nb_path) if nb_path.exists() else None
    line = f"- [`{py_path.name}`]({rel_py})"
    if desc:
        line += f" — {desc}"
    if rel_nb:
        line += f" [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_url(rel_nb)})"
    return line


def is_public_example(py_path: Path) -> bool:
    return len(py_path.parts) >= 2 and py_path.parts[1] in PUBLIC_EXAMPLE_ROOTS


def main() -> None:
    root = Path("examples")
    seen: set[Path] = set()
    sections: list[str] = []

    for group_dir, group_title in GROUPS:
        gd = Path(group_dir)
        if not gd.is_dir():
            continue
        py_files = []
        for p in sorted(gd.iterdir()):
            if p.suffix != ".py":
                continue
            if p in seen:
                continue
            # Skip files that are inside a *deeper* group directory.
            deeper = any(
                str(p).startswith(deeper_dir + "/")
                for deeper_dir, _ in GROUPS
                if Path(deeper_dir) != gd
                and str(p).startswith(group_dir + "/")
                and Path(deeper_dir).is_relative_to(gd)
            )
            if deeper:
                continue
            py_files.append(p)
            seen.add(p)
        if not py_files:
            continue
        section = [f"### {group_title}", ""]
        for p in py_files:
            section.append(gallery_entry(p))
        section.append("")
        sections.append("\n".join(section))

    # Catch-all for any .py we missed
    leftovers = sorted(
        p
        for p in root.rglob("*.py")
        if "__pycache__" not in p.parts and p not in seen and is_public_example(p)
    )
    if leftovers:
        section = ["### Uncategorised", ""]
        for p in leftovers:
            section.append(gallery_entry(p))
        section.append("")
        sections.append("\n".join(section))

    body = "\n".join(sections)
    output = f"""# Examples

This directory holds the beamax example gallery. Almost every script has a
matching notebook with an **Open in Colab** badge so you can run it on a free
GPU or TPU without any local setup — click the badge, then pick a GPU or TPU
runtime in Colab (`Runtime → Change runtime type`).

Each notebook installs beamax from this repository in its first code cell:

```
%pip install --quiet "beamax[kwave] @ git+https://github.com/{GITHUB_REPO}.git"
```

When running locally from a checkout, that cell is a no-op (skip it, or leave
it — `pip install` will simply reinstall the current working copy).

A few examples need additional data — most notably the OA-Breast phantom from
the [Illinois OA-Breast database](https://anastasio.bioengineering.illinois.edu/downloadable-content/oa-breast-database/).
Download `Neg_07_Left.h5` from there and place it under
`<repo-root>/data/NumericalBreastPhantoms-selected/hdf5/`. Notebooks that
need this data carry a banner pointing here.

## Style

Examples that customise matplotlib import `use_beamax_style` from
`beamax.plotter` — the style file is bundled inside the installed package, so
it resolves identically in a checkout, an installed wheel, or on Colab.

## Gallery

{body}

## Contributing a new example

1. Add the script under the appropriate `examples/<category>/` directory with
   a 1-2 sentence module docstring.
2. Run `python tools/finalize_examples.py` (or hand-edit a notebook) so a
   paired `.ipynb` exists with the Open-in-Colab badge + install cell pattern.
3. Add a bullet to the section above (or rerun the regeneration script).
4. If the example needs significant RAM or external data, mention it in the
   docstring so the auto-generated notebook can surface a warning.
"""
    Path("examples/README.md").write_text(output)
    print("wrote examples/README.md")


if __name__ == "__main__":
    main()
