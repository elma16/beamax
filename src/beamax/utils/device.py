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
    return _format_bytes(mem)


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


def find_repo_root(start: Path, marker: str = "src/beamax") -> Path:
    """
    Climb parent dirs until a directory containing `marker` exists.

    Parameters
    ----------
    start : Path
    marker : str
        Relative path to test for existence.

    Returns
    -------
    Path
        Found root or `start` if not found.
    """
    for parent in [start] + list(start.parents):
        if (parent / marker).exists():
            return parent
    return start  # fallback if marker not found


def detect_root() -> Path:
    """
    Heuristics for locating the repository root.

    Priority
    --------
    1) `BEAMAX_ROOT` environment variable
    2) Colab default path
    3) Script location (`__file__`)
    4) Current working directory upward search

    Returns
    -------
    Path
    """
    # 1. Env var override
    if "BEAMAX_ROOT" in os.environ:
        return Path(os.environ["BEAMAX_ROOT"]).expanduser()

    # 2. Colab
    if "COLAB_GPU" in os.environ:
        return Path("/content/drive/MyDrive/beamax")

    # 3. Script
    if "__file__" in globals():
        return find_repo_root(Path(__file__).resolve())

    # 4. Notebook / REPL → start from CWD
    return find_repo_root(Path.cwd())
