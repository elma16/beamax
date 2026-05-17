#!/usr/bin/env python3
"""
Minimal reproducer for two suspected k-wave-python upstream issues.

It deliberately imports `kwave` directly and does not import Beamax.

Expected use:

    python tools/repro_kwave_upstream_binary_absorption.py

Exit codes:
    0  all checks passed; the claims were not reproduced
    1  at least one claim was reproduced
    2  a required dependency or executable was unavailable
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


TMP_ROOT = Path(tempfile.gettempdir())
os.environ.setdefault("MPLCONFIGDIR", str(TMP_ROOT / "kwave_repro_mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(TMP_ROOT / "kwave_repro_cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)


@dataclass
class CheckResult:
    name: str
    status: str
    details: str


def _sha256_short(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _fake_omp_binary(path: Path, label: str) -> None:
    path.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' '" + label + ':\'"$0" >> "$KWAVE_REPRO_MARKER"\n'
        "exit 91\n",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def _make_problem(alpha_coeff: float, alpha_power: float):
    import numpy as np
    from kwave.kgrid import kWaveGrid
    from kwave.kmedium import kWaveMedium
    from kwave.ksensor import kSensor
    from kwave.ksource import kSource

    n = (32, 32)
    kgrid = kWaveGrid(n, (1e-4, 1e-4))
    kgrid.setTime(30, 2e-8)

    x = np.arange(n[0])[:, None]
    y = np.arange(n[1])[None, :]
    source = kSource()
    source.p0 = np.exp(-((x - 16) ** 2 + (y - 16) ** 2) / (2 * 3**2)).astype(np.float32)

    sensor_mask = np.zeros(n, dtype=bool)
    sensor_mask[4, :] = True

    medium = kWaveMedium(
        sound_speed=1500.0,
        density=1000.0,
        alpha_coeff=alpha_coeff,
        alpha_power=alpha_power,
    )
    return kgrid, medium, source, kSensor(mask=sensor_mask)


def _run_problem(*, backend: str, alpha_coeff: float, alpha_power: float):
    import numpy as np
    from kwave.kspaceFirstOrder import kspaceFirstOrder

    kgrid, medium, source, sensor = _make_problem(alpha_coeff, alpha_power)
    out = kspaceFirstOrder(
        kgrid,
        medium,
        source,
        sensor,
        backend=backend,
        device="cpu",
        pml_size=4,
        pml_inside=True,
        smooth_p0=False,
        quiet=True,
        num_threads=1,
    )
    return np.asarray(out["p"], dtype=float)


def _install_darwin_hdf5_compat(binary_path: Path) -> str | None:
    """Allow a libhdf5.310-linked Darwin binary to load against libhdf5.320."""
    if sys.platform != "darwin":
        return None

    try:
        out = subprocess.run(
            ["otool", "-L", str(binary_path)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None

    refs = re.findall(
        r"^\s+(/.*libhdf5(?:_hl)?\.310\.dylib)\s",
        out,
        flags=re.MULTILINE,
    )
    if not refs:
        return None

    compat_dir = TMP_ROOT / "kwave_repro_hdf5_compat"
    compat_dir.mkdir(parents=True, exist_ok=True)

    linked = []
    for ref in refs:
        ref_path = Path(ref)
        if ref_path.exists():
            continue
        for replacement_name in (
            ref_path.name.replace(".310.", ".320."),
            ref_path.name.replace(".310", ""),
        ):
            replacement = ref_path.with_name(replacement_name)
            if replacement.exists():
                link_path = compat_dir / ref_path.name
                if link_path.exists() or link_path.is_symlink():
                    link_path.unlink()
                link_path.symlink_to(replacement)
                linked.append(f"{link_path.name}->{replacement.name}")
                break

    if not linked:
        return None

    current = os.environ.get("DYLD_LIBRARY_PATH", "")
    entries = [str(compat_dir)]
    if current:
        entries.append(current)
    os.environ["DYLD_LIBRARY_PATH"] = ":".join(entries)
    return ", ".join(linked)


def _stage_binary_dir(kwave, binary_dir: Path | None) -> tuple[Path, Path, str]:
    binary_name = (
        "kspaceFirstOrder-OMP.exe"
        if sys.platform == "win32"
        else "kspaceFirstOrder-OMP"
    )

    if binary_dir is not None:
        staged_dir = binary_dir
        binary_path = staged_dir / binary_name
        if not binary_path.exists():
            raise FileNotFoundError(f"{binary_path} does not exist")
        return staged_dir, binary_path, "provided"

    source = Path(kwave.BINARY_PATH) / binary_name
    if not source.exists():
        raise FileNotFoundError(f"{source} does not exist")

    staged_dir = Path(tempfile.mkdtemp(prefix="kwave_repro_bin_"))
    binary_path = staged_dir / binary_name
    shutil.copy2(source, binary_path)
    binary_path.chmod(binary_path.stat().st_mode | stat.S_IEXEC)
    return staged_dir, binary_path, "bundled-copy"


def check_binary_path_is_honored() -> CheckResult:
    import kwave
    from kwave.compat import options_to_kwargs
    from kwave.kspaceFirstOrder import kspaceFirstOrder
    from kwave.options import SimulationExecutionOptions

    with tempfile.TemporaryDirectory(prefix="kwave_repro_binary_path_") as tmp:
        tmp_path = Path(tmp)
        bundled_dir = tmp_path / "bundled"
        custom_dir = tmp_path / "custom"
        bundled_dir.mkdir()
        custom_dir.mkdir()

        binary_name = (
            "kspaceFirstOrder-OMP.exe"
            if sys.platform == "win32"
            else "kspaceFirstOrder-OMP"
        )
        bundled_bin = bundled_dir / binary_name
        custom_bin = custom_dir / binary_name
        marker = tmp_path / "marker.txt"
        _fake_omp_binary(bundled_bin, "bundled")
        _fake_omp_binary(custom_bin, "custom")

        converted = options_to_kwargs(
            None,
            SimulationExecutionOptions(
                is_gpu_simulation=False,
                backend="OMP",
                binary_path=str(custom_bin),
                show_sim_log=False,
                num_threads=1,
            ),
        )
        legacy_drops_path = "binary_path" not in converted

        signature_has_binary_path = (
            "binary_path" in inspect.signature(kspaceFirstOrder).parameters
        )
        if not signature_has_binary_path:
            detail = (
                "kspaceFirstOrder() has no binary_path parameter; "
                f"options_to_kwargs keys are {sorted(converted)}"
            )
            return CheckResult("binary_path", "fail", detail)

        original_binary_path = kwave.BINARY_PATH
        original_marker = os.environ.get("KWAVE_REPRO_MARKER")
        try:
            kwave.BINARY_PATH = bundled_dir
            os.environ["KWAVE_REPRO_MARKER"] = str(marker)
            kgrid, medium, source, sensor = _make_problem(0.0, 1.5)
            try:
                kspaceFirstOrder(
                    kgrid,
                    medium,
                    source,
                    sensor,
                    backend="cpp",
                    device="cpu",
                    binary_path=str(custom_bin),
                    pml_size=4,
                    pml_inside=True,
                    smooth_p0=False,
                    quiet=True,
                    num_threads=1,
                )
            except subprocess.CalledProcessError:
                pass

            ran = (
                marker.read_text(encoding="utf-8").strip().splitlines()
                if marker.exists()
                else []
            )
        finally:
            kwave.BINARY_PATH = original_binary_path
            if original_marker is None:
                os.environ.pop("KWAVE_REPRO_MARKER", None)
            else:
                os.environ["KWAVE_REPRO_MARKER"] = original_marker

        if any(line.startswith("custom:") for line in ran):
            detail = "custom binary_path was executed"
            if legacy_drops_path:
                detail += "; legacy options_to_kwargs still drops binary_path"
                return CheckResult("binary_path", "fail", detail)
            return CheckResult("binary_path", "pass", detail)

        if any(line.startswith("bundled:") for line in ran):
            return CheckResult(
                "binary_path",
                "fail",
                "binary_path was provided, but kwave.BINARY_PATH binary was executed",
            )

        return CheckResult(
            "binary_path",
            "fail",
            "binary_path was provided, but no fake binary execution was observed",
        )


def check_power_law_absorption(
    binary_dir: Path | None, drop_tol: float, python_min: float
) -> CheckResult:
    import numpy as np
    import kwave

    original_binary_path = kwave.BINARY_PATH
    try:
        staged_dir, binary_path, source_kind = _stage_binary_dir(kwave, binary_dir)
        compat = _install_darwin_hdf5_compat(binary_path)
        kwave.BINARY_PATH = staged_dir

        cpp_lossless = _run_problem(backend="cpp", alpha_coeff=0.0, alpha_power=1.5)
        cpp_absorbing = _run_problem(backend="cpp", alpha_coeff=50.0, alpha_power=1.5)
        py_lossless = _run_problem(backend="python", alpha_coeff=0.0, alpha_power=1.5)
        py_absorbing = _run_problem(backend="python", alpha_coeff=50.0, alpha_power=1.5)
    except FileNotFoundError as exc:
        return CheckResult("power_law_absorption", "skip", str(exc))
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        details = f"C++ binary failed with return code {exc.returncode}"
        if stderr:
            details += f": {stderr.splitlines()[-1]}"
        return CheckResult("power_law_absorption", "skip", details)
    finally:
        kwave.BINARY_PATH = original_binary_path

    details = f"{source_kind} {binary_path} sha256={_sha256_short(binary_path)}"
    if compat:
        details += f"; hdf5 compat={compat}"

    nonfinite = {
        "cpp_lossless": int(cpp_lossless.size - np.isfinite(cpp_lossless).sum()),
        "cpp_absorbing": int(cpp_absorbing.size - np.isfinite(cpp_absorbing).sum()),
        "python_lossless": int(py_lossless.size - np.isfinite(py_lossless).sum()),
        "python_absorbing": int(py_absorbing.size - np.isfinite(py_absorbing).sum()),
    }
    if any(nonfinite.values()):
        return CheckResult(
            "power_law_absorption",
            "fail",
            details + f"; non-finite samples={nonfinite}",
        )

    cpp_rel = float(
        np.linalg.norm(cpp_absorbing - cpp_lossless)
        / max(np.linalg.norm(cpp_lossless), np.finfo(float).eps)
    )
    py_rel = float(
        np.linalg.norm(py_absorbing - py_lossless)
        / max(np.linalg.norm(py_lossless), np.finfo(float).eps)
    )

    details += f"; cpp rel diff={cpp_rel:.3e}; python rel diff={py_rel:.3e}"

    if py_rel < python_min:
        return CheckResult(
            "power_law_absorption",
            "skip",
            details
            + f"; python backend did not show the expected >= {python_min:g} sensitivity",
        )

    if cpp_rel <= drop_tol:
        return CheckResult(
            "power_law_absorption",
            "fail",
            details + f"; cpp response is <= drop_tol={drop_tol:g}",
        )

    return CheckResult("power_law_absorption", "pass", details)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fail when k-wave-python ignores binary_path or when the OMP binary "
            "does not respond to power-law absorption."
        )
    )
    parser.add_argument(
        "--binary-dir",
        type=Path,
        default=None,
        help="Directory containing kspaceFirstOrder-OMP to test instead of kwave.BINARY_PATH.",
    )
    parser.add_argument(
        "--skip-absorption",
        action="store_true",
        help="Only run the binary_path check.",
    )
    parser.add_argument(
        "--drop-tol",
        type=float,
        default=1e-4,
        help="Relative C++ lossless-vs-absorbing difference at or below this is treated as dropped absorption.",
    )
    parser.add_argument(
        "--python-min",
        type=float,
        default=1e-2,
        help="Required Python-backend relative difference proving the setup is absorption-sensitive.",
    )
    args = parser.parse_args()

    try:
        import kwave
    except Exception as exc:
        print(f"SKIP dependency: cannot import kwave: {exc}", file=sys.stderr)
        return 2

    print(f"k-wave-python version: {getattr(kwave, '__version__', 'unknown')}")
    print(f"platform: {sys.platform}")
    print(f"kwave.BINARY_PATH: {getattr(kwave, 'BINARY_PATH', 'unknown')}")

    results = [check_binary_path_is_honored()]
    if not args.skip_absorption:
        results.append(
            check_power_law_absorption(args.binary_dir, args.drop_tol, args.python_min)
        )

    print()
    for result in results:
        print(f"{result.status.upper():4} {result.name}: {result.details}")

    if any(result.status == "fail" for result in results):
        return 1
    if any(result.status == "skip" for result in results):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
