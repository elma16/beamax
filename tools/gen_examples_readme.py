#!/usr/bin/env python3
"""
Regenerate examples/README.md as the public example index.

Private examples under examples/private are preserved as unsupported archived
material and are intentionally omitted from the gallery.
"""

from __future__ import annotations

import re
import ast
from pathlib import Path

GITHUB_REPO = "elma16/beamax"
GITHUB_BRANCH = "main"
PUBLIC_EXAMPLES_ROOT = Path("examples")

# Friendly section titles, ordered
GROUPS = [
    ("examples/decomposition", "Frequency decomposition & MSWPT"),
    ("examples/forward", "Forward propagation"),
    ("examples/reconstruction/time-reversal", "Time-reversal reconstruction"),
    ("examples/rays", "Rays and autofocus"),
    ("examples/single-gaussian-beam", "Single Gaussian beam diagnostics"),
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


def example_metadata(py_path: Path) -> dict[str, str]:
    """Read ``Example key: value`` metadata from a module docstring."""
    try:
        docstring = ast.get_docstring(ast.parse(py_path.read_text()))
    except SyntaxError:
        return {}
    if not docstring:
        return {}
    metadata: dict[str, str] = {}
    for line in docstring.splitlines():
        if not line.startswith("Example "):
            continue
        key, sep, value = line.partition(":")
        if sep:
            metadata[key.removeprefix("Example ").strip().lower()] = value.strip()
    return metadata


def is_default_smoke_example(py_path: Path) -> bool:
    smoke = example_metadata(py_path).get("smoke", "true")
    return smoke.lower() not in {"0", "false", "no", "off"}


def optional_note(py_path: Path) -> str:
    metadata = example_metadata(py_path)
    if is_default_smoke_example(py_path):
        return ""
    extras = metadata.get("extras", "").strip()
    install = f"`beamax[{extras}]`" if extras else "extra dependencies"
    return f" _(optional; requires {install}; skipped by default smoke)_"


def gallery_sort_key(py_path: Path) -> tuple[bool, str]:
    """Sort base smoke examples before optional examples, then by filename."""
    return (not is_default_smoke_example(py_path), py_path.name)


def colab_url(rel_nb: str) -> str:
    return f"https://colab.research.google.com/github/{GITHUB_REPO}/blob/{GITHUB_BRANCH}/{rel_nb}"


def gallery_entry(py_path: Path) -> str:
    nb_path = py_path.with_suffix(".ipynb")
    desc = short_desc(py_path)
    rel_py = str(py_path.relative_to(Path("examples")))
    rel_nb = str(nb_path) if nb_path.exists() else None
    line = f"- [`{py_path.name}`]({rel_py})"
    if desc:
        line += f" — {desc}{optional_note(py_path)}"
    if rel_nb:
        line += f" [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_url(rel_nb)})"
    return line


def is_public_example(py_path: Path) -> bool:
    try:
        rel = py_path.relative_to(PUBLIC_EXAMPLES_ROOT)
    except ValueError:
        return False
    return "private" not in rel.parts


def main() -> None:
    root = Path("examples")
    seen: set[Path] = set()
    sections: list[str] = []

    for group_dir, group_title in GROUPS:
        gd = Path(group_dir)
        if not gd.is_dir():
            continue
        py_files = []
        for p in sorted(gd.iterdir(), key=lambda path: path.name):
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
        for p in sorted(py_files, key=gallery_sort_key):
            section.append(gallery_entry(p))
        section.append("")
        sections.append("\n".join(section))

    # Catch-all for any .py we missed
    leftovers = sorted(
        (
            p
            for p in root.rglob("*.py")
            if "__pycache__" not in p.parts and p not in seen and is_public_example(p)
        ),
        key=gallery_sort_key,
    )
    if leftovers:
        section = ["### Uncategorised", ""]
        for p in leftovers:
            section.append(gallery_entry(p))
        section.append("")
        sections.append("\n".join(section))

    body = "\n".join(sections)
    output = f"""# Examples

This directory holds the supported beamax example gallery. Base examples are
small, documented, paired with notebooks, linted, and smoke-tested in CI.
Examples marked optional require extra dependencies and are skipped by default
smoke runs.

`private/` preserves research, profiling, comparison, and diagnostic scripts.
They may require extra data, optional solver backends, large memory, or local
hardware assumptions, and are not part of the public docs or CI smoke suite.

Every public script has a matching notebook with an **Open in Colab** badge.
The public examples are small enough to run on a standard CPU Colab runtime.

Each notebook installs beamax from this repository in its first code cell:

```
%pip install --quiet "beamax[viz-mpl] @ git+https://github.com/{GITHUB_REPO}.git"
```

When running locally from a checkout, that cell can be skipped.

## Style

Examples that customise matplotlib import `use_beamax_style` from
`beamax.plotter` — the style file is bundled inside the installed package, so
it resolves identically in a checkout, an installed wheel, or on Colab.

## Gallery

{body}

## Contributing a new example

1. Add the script under the appropriate `examples/<category>/` directory with
   a 1-2 sentence module docstring.
   Use `Example extras: ...` and `Example smoke: false` for optional-runtime
   examples.
2. Run `python tools/finalize_examples.py` (or hand-edit a notebook) so a
   paired `.ipynb` exists with the Open-in-Colab badge + install cell pattern.
3. Add a bullet to the section above (or rerun the regeneration script).
4. Keep public examples self-contained and fast. Move research/profiling/data-
   dependent material to `examples/private`.
"""
    Path("examples/README.md").write_text(output)
    print("wrote examples/README.md")


if __name__ == "__main__":
    main()
