"""
Device/memory helpers for JAX environments.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import Union
from pathlib import Path
import os


def get_devices():
    """
    Inspect available JAX devices.

    Returns
    -------
    Tuple[bool, bool, bool]
        Presence flags for (CPU, GPU, TPU). Prints the detected devices.
    """
    devices = jax.devices()
    print("Available devices:", devices)

    has_gpu = any(d.platform == "gpu" for d in devices)
    has_tpu = any(d.platform == "tpu" for d in devices)
    has_cpu = any(d.platform == "cpu" for d in devices)

    return has_cpu, has_gpu, has_tpu


def _format_bytes(bytes: int) -> str:
    """
    Human-readable formatting for byte counts.

    Parameters
    ----------
    bytes : int

    Returns
    -------
    str
        Formatted string with units (KB/MB/GB).
    """
    if bytes < 1024**2:
        return f"{bytes / 1024:.2f} Kb"
    elif bytes < 1024**3:
        return f"{bytes / 1024**2:.2f} Mb"
    else:
        return f"{bytes / 1024**3:.2f} Gb"


def memory_estimate(dims: jnp.ndarray, dtype: jnp.dtype) -> str:
    """
    Estimate memory footprint for an array shape/dtype.

    Parameters
    ----------
    dims : jnp.ndarray
        Shape tuple or array of dimensions.
    dtype : jnp.dtype

    Returns
    -------
    str
        Estimated memory usage.
    """
    mem = jnp.prod(dims) * jnp.dtype(dtype).itemsize
    return _format_bytes(int(mem))


def memory_str(x: jnp.ndarray) -> str:
    """
    Memory usage of an existing array.

    Parameters
    ----------
    x : jnp.ndarray

    Returns
    -------
    str
        Human-friendly memory string.
    """
    return _format_bytes(x.nbytes)


def array_str(x: Union[jnp.ndarray, None]) -> Union[str, None]:
    """
    Short descriptor for an array (shape/dtype/memory).

    Parameters
    ----------
    x : jnp.ndarray | None

    Returns
    -------
    str | None
    """
    return None if x is None else f"Array {x.dtype} {tuple(x.shape)} | {memory_str(x)}"


def find_repo_root(start: Path, marker: str = "pyproject.toml") -> Path:
    """
    Climb parent dirs until a directory containing `marker` exists.

    Parameters
    ----------
    start : Path
    marker : str
        Relative path to test for existence. Defaults to ``pyproject.toml``
        which lives at the beamax repo root and is absent from site-packages
        installs, so this doubles as a "are we in a source checkout?" probe.

    Returns
    -------
    Path
        Found root or the nearest directory containing ``start`` if not found.
    """
    start = Path(start)
    base = start if start.is_dir() else start.parent
    for parent in [base] + list(base.parents):
        if (parent / marker).exists():
            return parent
    return base


def detect_root() -> Path:
    """
    Locate the repository root used by examples for output files.

    Priority
    --------
    1) `BEAMAX_ROOT` environment variable
    2) Current working directory upward search
    3) Package source location upward search
    4) Current working directory

    Returns
    -------
    Path
    """
    if "BEAMAX_ROOT" in os.environ:
        return Path(os.environ["BEAMAX_ROOT"]).expanduser()

    cwd_root = find_repo_root(Path.cwd())
    if (cwd_root / "pyproject.toml").exists():
        return cwd_root

    if "__file__" in globals():
        package_root = find_repo_root(Path(__file__).resolve())
        if (package_root / "pyproject.toml").exists():
            return package_root

    return Path.cwd()


def example_plot_dir(example_file: str | os.PathLike[str]) -> Path:
    """
    Return the plot output directory for a public example script.

    Examples mirror their first directory under ``examples/``:

    - ``examples/forward/2d_forward.py`` -> ``<root>/plots/forward``
    - ``examples/rays/2d_ray_bending.py`` -> ``<root>/plots/rays``

    If ``example_file`` is outside the detected checkout's ``examples`` tree,
    the file's immediate parent directory name is used as the category.

    Parameters
    ----------
    example_file : str or os.PathLike
        Usually ``__file__`` from an example script.

    Returns
    -------
    Path
        Existing output directory for plots from that example category.
    """
    root = detect_root()
    example_path = Path(example_file).resolve()

    try:
        rel = example_path.relative_to((root / "examples").resolve())
        category = rel.parts[0] if len(rel.parts) > 1 else example_path.parent.name
    except ValueError:
        category = example_path.parent.name

    out = root / "plots" / category
    out.mkdir(parents=True, exist_ok=True)
    return out
