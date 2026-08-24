import argparse
import os
import subprocess
import sys
from pathlib import Path

from example_metadata import ExampleInfo, read_example_info


def optional_skip_reason(info: ExampleInfo) -> str:
    """Return the reason an optional example is skipped by the default suite."""
    if info.extras:
        return f"requires beamax[{','.join(info.extras)}]"
    return "marked optional by Example smoke: false"


def run_python_files(
    directory: str | Path,
    fail_fast: bool = False,
    silent_figures: bool = False,
    include_optional: bool = False,
) -> None:
    total_failures = 0
    skipped_optional: list[tuple[Path, str]] = []

    for file_path in sorted(Path(directory).rglob("*.py")):
        if "__pycache__" in file_path.parts:
            continue
        info = read_example_info(file_path)
        if not include_optional and not info.smoke:
            skipped_optional.append((file_path, optional_skip_reason(info)))
            continue
        print(f"Running: {file_path}")

        env = os.environ.copy()
        if silent_figures:
            env["MPLBACKEND"] = "Agg"
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
        if skipped_optional:
            print("\nSkipped optional examples:")
            for path, reason in skipped_optional:
                print(f"  - {path} ({reason})")
            print("Run again with --include-optional to execute them.")
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
