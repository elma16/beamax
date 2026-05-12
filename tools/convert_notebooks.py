#!/usr/bin/env python3
"""
convert_notebooks.py

Convert Jupyter notebooks to cleaned Python scripts.

- Uses `jupyter nbconvert --to python`.
- Removes Jupyter cell headers like "# In[1]:" / "# Out[ ]:".
- Strips lines containing "get_ipython()".
- Collapses multiple blank lines; trims trailing whitespace.
- Skips .ipynb_checkpoints.
- Optional: `--git-add` to stage converted .py files.
- Default root is "<repo>/examples" if beamax.utils.detect_root() is available,
  otherwise CWD / "examples" when present. `examples/private` is skipped.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

try:
    # Prefer repo-relative detection if available
    from beamax import utils as _beamax_utils  # type: ignore
except Exception:
    _beamax_utils = None  # fallback


CELL_HDR_RE = re.compile(r"^\s*#\s*(In|Out)\s*\[\s*\d*\s*\]\s*:.*$")
IPYNB_CHECKPOINTS = ".ipynb_checkpoints"


def _have_nbconvert() -> bool:
    try:
        subprocess.run(
            ["jupyter", "nbconvert", "--version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False


def _clean_nbconvert_output(py_path: Path) -> None:
    """Rewrite file in place: drop cell headers + get_ipython, collapse blanks."""
    try:
        text = py_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"    ✗ read failed: {py_path} ({e})")
        return

    lines_in = text.splitlines()
    out_lines: list[str] = []
    prev_blank = False

    for raw in lines_in:
        ln = raw.rstrip()  # trim trailing whitespace
        if "get_ipynb" in ln:  # stray typos some exporters add
            pass  # fall through, we still filter below
        if "get_ipython()" in ln:
            continue
        if CELL_HDR_RE.match(ln):
            continue

        if ln == "":
            if prev_blank:
                continue
            prev_blank = True
            out_lines.append("")  # single blank
        else:
            prev_blank = False
            out_lines.append(ln)

    try:
        py_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    except Exception as e:
        print(f"    ✗ write failed: {py_path} ({e})")


def _convert_one(
    nb_path: Path, overwrite: bool = True, quiet: bool = True
) -> Path | None:
    """Run nbconvert for a single notebook. Return output .py path or None on failure."""
    out_dir = nb_path.parent
    py_path = out_dir / f"{nb_path.stem}.py"
    if py_path.exists() and not overwrite:
        print(f"    ↷ skipping (exists): {py_path.name}")
        return py_path

    cmd = [
        "jupyter",
        "nbconvert",
        "--to",
        "python",
        "--output",
        nb_path.stem,
        "--output-dir",
        str(out_dir),
        str(nb_path),
    ]
    try:
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        if not quiet and res.stdout:
            print(res.stdout)
    except subprocess.CalledProcessError as e:
        msg = e.stderr.strip() or e.stdout.strip() or str(e)
        print(f"    ✗ nbconvert failed: {nb_path.name}\n      {msg}")
        return None

    if not py_path.exists():
        print(f"    ✗ nbconvert reported success but {py_path.name} not found")
        return None

    _clean_nbconvert_output(py_path)
    return py_path


def _git_add(paths: list[Path]) -> None:
    if not paths:
        return
    try:
        subprocess.run(["git", "add", "--"] + [str(p) for p in paths], check=True)
    except Exception as e:
        print(f"    ✗ git add failed: {e}")


def find_default_root() -> Path:
    if _beamax_utils is not None:
        try:
            root = _beamax_utils.detect_root()
            ex = Path(root) / "examples"
            return ex if ex.exists() else Path(root)
        except Exception:
            pass
    # fallback: CWD/examples if exists, else CWD
    cwd = Path.cwd()
    ex = cwd / "examples"
    return ex if ex.exists() else cwd


def convert_notebooks(
    root: Path,
    include_glob: str = "**/*.ipynb",
    overwrite: bool = True,
    git_add: bool = False,
    quiet: bool = True,
) -> tuple[int, int, list[Path]]:
    """Return (ok_count, total, converted_paths)."""
    root = root.resolve()
    if not root.exists():
        print(f"Error: root does not exist: {root}")
        return 0, 0, []

    nbs = [
        p
        for p in root.rglob("*")
        if p.suffix == ".ipynb" and IPYNB_CHECKPOINTS not in p.parts
        and "private" not in p.relative_to(root).parts
    ]
    total = len(nbs)
    if total == 0:
        print(f"No notebooks under {root}")
        return 0, 0, []

    print(f"Found {total} notebook(s) under '{root}':")
    converted: list[Path] = []
    ok = 0
    for nb in sorted(nbs):
        rel = nb.relative_to(root)
        print(f"  → {rel}")
        out = _convert_one(nb, overwrite=overwrite, quiet=quiet)
        if out is None:
            continue
        print(f"    ✓ wrote {out.name}")
        converted.append(out)
        ok += 1

    if git_add:
        _git_add(converted)

    print(f"\nSummary: {ok}/{total} converted.")
    return ok, total, converted


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Convert Jupyter notebooks to cleaned Python scripts."
    )
    ap.add_argument(
        "root",
        nargs="?",
        default=None,
        help="Root directory to search (default: <repo>/examples or CWD/examples).",
    )
    ap.add_argument(
        "--git-add",
        action="store_true",
        help="git add the converted .py files after conversion.",
    )
    ap.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Do not overwrite existing .py outputs; skip those notebooks.",
    )
    ap.add_argument(
        "--verbose", "-v", action="store_true", help="Print nbconvert stdout."
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    if not _have_nbconvert():
        print("Error: jupyter nbconvert not found. Install with: pip install nbconvert")
        return 2

    root = Path(args.root) if args.root is not None else find_default_root()
    overwrite = not args.no_overwrite
    ok, total, _ = convert_notebooks(
        root=root,
        overwrite=overwrite,
        git_add=args.git_add,
        quiet=not args.verbose,
    )
    return 0 if ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
