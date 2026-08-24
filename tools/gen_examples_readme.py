#!/usr/bin/env python3
"""
Regenerate examples/README.md as the public example index.

The script index is generated from the repository's example tree.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from example_metadata import ExampleInfo, read_example_info

GITHUB_REPO = "elma16/beamax"
GITHUB_BRANCH = "main"
PUBLIC_EXAMPLES_ROOT = Path("examples")

GROUPS = [
    ("forward", "Forward propagation"),
    ("reconstruction", "Reconstruction"),
    ("rays", "Rays and autodiff"),
    ("single-gaussian-beam", "Single Gaussian beam diagnostics"),
    ("diagnostics", "Diagnostics"),
]


def optional_note(info: ExampleInfo) -> str:
    if info.smoke:
        return ""
    extras = ",".join(info.extras)
    install = f"`beamax[{extras}]`" if extras else "extra dependencies"
    return f" _(requires {install}; excluded from default smoke)_"


def gallery_sort_key(entry: tuple[Path, ExampleInfo]) -> tuple[bool, str]:
    """Sort base smoke examples before optional examples, then by filename."""
    path, info = entry
    return (not info.smoke, path.name)


def colab_url(rel_nb: str) -> str:
    return f"https://colab.research.google.com/github/{GITHUB_REPO}/blob/{GITHUB_BRANCH}/{rel_nb}"


def gallery_entry(py_path: Path, info: ExampleInfo) -> str:
    nb_path = py_path.with_suffix(".ipynb")
    rel_py = str(py_path.relative_to(Path("examples")))
    rel_nb = str(nb_path) if nb_path.exists() else None
    line = f"- [`{py_path.name}`]({rel_py})"
    if info.description:
        line += f" — {info.description}{optional_note(info)}"
    if rel_nb:
        line += f" [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_url(rel_nb)})"
    return line


def render_readme() -> str:
    """Render the public example index from example metadata."""
    sections: list[str] = []
    has_optional_examples = False

    for group_name, group_title in GROUPS:
        group_dir = PUBLIC_EXAMPLES_ROOT / group_name
        entries = [
            (path, read_example_info(path))
            for path in group_dir.glob("*.py")
            if "__pycache__" not in path.parts
        ]
        if not entries:
            continue
        section = [f"### {group_title}", ""]
        for path, info in sorted(entries, key=gallery_sort_key):
            section.append(gallery_entry(path, info))
            has_optional_examples |= not info.smoke
        section.append("")
        sections.append("\n".join(section))

    body = "\n".join(sections)
    optional_section = (
        """## Smoke testing

Examples marked `Example smoke: false` are skipped by default. To include them:

```bash
python tools/run_examples.py --directory examples --include-optional --silent-figures
```

"""
        if has_optional_examples
        else ""
    )
    intro = """# Examples

Run scripts from the repository root. Selected examples include Colab notebooks;
optional requirements are listed below.

## Gallery
"""
    output_parts = [intro, body]
    if optional_section:
        output_parts.append(optional_section)
    return "\n\n".join(part.strip() for part in output_parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if examples/README.md is stale without changing it",
    )
    args = parser.parse_args()

    path = Path("examples/README.md")
    output = render_readme()
    if args.check:
        if not path.exists() or path.read_text() != output:
            raise SystemExit(
                "examples/README.md is stale; run tools/gen_examples_readme.py"
            )
        print("examples/README.md is up to date")
        return

    path.write_text(output)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
