from pathlib import Path
import subprocess
import sys
import argparse
import os
import ast

PRIVATE_EXAMPLE_DIRS = {"private", "thesis", "learned", "benchmarks"}


def example_metadata(file_path: Path) -> dict[str, str]:
    """Read ``Example key: value`` metadata from a module docstring."""
    try:
        docstring = ast.get_docstring(ast.parse(file_path.read_text()))
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


def is_default_smoke_example(file_path: Path) -> bool:
    """Return whether this example should run in the default smoke suite."""
    smoke = example_metadata(file_path).get("smoke", "true")
    return smoke.lower() not in {"0", "false", "no", "off"}


def run_python_files(
    directory: str | Path,
    fail_fast: bool = False,
    silent_figures: bool = False,
    include_optional: bool = False,
) -> None:
    total_failures = 0

    # Use pathlib to recursively find all Python files.
    for file_path in sorted(Path(directory).rglob("*.py")):
        if "__pycache__" in file_path.parts or (
            PRIVATE_EXAMPLE_DIRS & set(file_path.parts)
        ):
            continue
        if not include_optional and not is_default_smoke_example(file_path):
            continue
        print(f"Running: {file_path}")

        # Set up environment for subprocess
        env = os.environ.copy()
        if silent_figures:
            env["MPLBACKEND"] = "Agg"  # Use non-interactive matplotlib backend
            if "MPLCONFIGDIR" not in env:
                mpl_config_dir = (
                    Path(os.environ.get("TMPDIR", "/tmp")) / "beamax-mplconfig"
                )
                mpl_config_dir.mkdir(parents=True, exist_ok=True)
                env["MPLCONFIGDIR"] = str(mpl_config_dir)

        try:
            result = subprocess.run(
                [sys.executable, str(file_path)],
                capture_output=True,
                text=True,
                check=True,
                env=env,
            )
            print(f"Output:\n{result.stdout}")
        except subprocess.CalledProcessError as e:
            print(f"Errors:\n{e.stderr}")
            total_failures += 1
            if fail_fast:
                print("Terminating on first failure (--fail-fast enabled)")
                sys.exit(1)
        except Exception as e:
            print(f"Failed to run {file_path}: {e}")
            total_failures += 1
            if fail_fast:
                print("Terminating on first failure (--fail-fast enabled)")
                sys.exit(1)

    if total_failures > 0:
        print(f"Total failures: {total_failures}")
        sys.exit(1)
    else:
        print("All Python files ran successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Python files in a directory")
    parser.add_argument(
        "--fail-fast",
        "-f",
        action="store_true",
        help="Exit on first failure instead of running all files",
    )
    parser.add_argument(
        "--silent-figures",
        "-s",
        action="store_true",
        help="Prevent matplotlib figures from popping up",
    )
    parser.add_argument(
        "--directory",
        "-d",
        default="examples",
        help="Directory to search for Python files (default: examples)",
    )
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Also run examples marked with `Example smoke: false`",
    )

    args = parser.parse_args()
    run_python_files(
        args.directory,
        args.fail_fast,
        args.silent_figures,
        args.include_optional,
    )
