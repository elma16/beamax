from pathlib import Path
import subprocess
import sys
import argparse
import os


def run_python_files(directory, fail_fast=False, silent_figures=False):
    total_failures = 0

    # Use pathlib to recursively find all Python files
    for file_path in Path(directory).rglob("*.py"):
        print(f"Running: {file_path}")

        # Set up environment for subprocess
        env = os.environ.copy()
        if silent_figures:
            env["MPLBACKEND"] = "Agg"  # Use non-interactive matplotlib backend

        try:
            result = subprocess.run(
                ["python", str(file_path)],
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
        default="examples/",
        help="Directory to search for Python files (default: examples/)",
    )

    args = parser.parse_args()
    run_python_files(args.directory, args.fail_fast, args.silent_figures)
