"""
Profiling utilities for MSGB solver performance analysis.

Usage:
    Set environment variable BEAMAX_PROFILE=1 to enable profiling output.

    from beamax.utils.profiling import profile_section, get_memory_mb

    with profile_section("my_operation"):
        # ... code ...
"""

from __future__ import annotations

import os
import time
import functools
from contextlib import contextmanager, suppress
from typing import Optional, Dict, Any, Iterable

import jax
import jax.numpy as jnp


# Check if profiling is enabled
PROFILING_ENABLED = os.environ.get("BEAMAX_PROFILE", "0") == "1"


# ---- Memory helpers (explicit exceptions; no bare excepts) ----------------- #


def _gpu_memory_used_mb() -> Optional[float]:
    """Return used GPU memory (MB) via NVML, or None if unavailable."""
    try:
        devices = jax.devices()
    except RuntimeError:
        return None

    if not devices or getattr(devices[0], "platform", None) != "gpu":
        return None

    try:
        import pynvml  # type: ignore
        from pynvml import NVMLError  # type: ignore
    except ImportError:
        return None

    try:
        pynvml.nvmlInit()
    except NVMLError:
        return None

    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return float(info.used) / (1024**2)
    except NVMLError:
        return None
    finally:
        # Some drivers raise on shutdown; suppress only NVML-specific errors.
        with suppress(NVMLError):
            pynvml.nvmlShutdown()


def _rss_memory_used_mb() -> Optional[float]:
    """Return process RSS (MB) via psutil, or None if unavailable."""
    try:
        import psutil  # type: ignore
    except ImportError:
        return None

    try:
        process = psutil.Process()
        return float(process.memory_info().rss) / (1024**2)
    except (psutil.Error, OSError):
        return None


def _rusage_maxrss_mb() -> Optional[float]:
    """Return max resident set size (MB) via resource.getrusage, or None."""
    try:
        import resource  # Unix-only
        import sys
    except ImportError:
        return None

    try:
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # On macOS ru_maxrss is bytes; on Linux it's kilobytes.
        if sys.platform == "darwin":
            return float(ru) / (1024**2)
        else:
            return float(ru) / 1024.0
    except (ValueError, OSError):
        return None


def get_memory_mb() -> float:
    """
    Get current device/process memory usage in MB.

    Returns
    -------
    float
        Memory in MB, or ``-1.0`` if no method succeeded.

    Notes
    -----
    The lookup order is GPU memory via NVML, process RSS via ``psutil``, then
    ``resource.getrusage`` maximum RSS as a platform-dependent fallback.
    """
    for f in (_gpu_memory_used_mb, _rss_memory_used_mb, _rusage_maxrss_mb):
        val = f()
        if val is not None:
            return val
    return -1.0


# ---- Formatting and array inspection -------------------------------------- #


def format_bytes(nbytes: int) -> str:
    """
    Format bytes into a human-readable string.

    Parameters
    ----------
    nbytes : int
        Byte count.

    Returns
    -------
    str
        Formatted size in KB, MB, or GB.
    """
    if nbytes < 1024**2:
        return f"{nbytes / 1024:.2f} KB"
    elif nbytes < 1024**3:
        return f"{nbytes / (1024**2):.2f} MB"
    else:
        return f"{nbytes / (1024**3):.2f} GB"


def array_info(arr: jnp.ndarray, name: str = "array") -> Dict[str, Any]:
    """
    Get detailed information about a JAX array.

    Parameters
    ----------
    arr : jnp.ndarray
        Array to inspect.
    name : str, default="array"
        Display name for the array.

    Returns
    -------
    Dict[str, Any]
        Shape, dtype, size, and memory metadata.
    """
    if arr is None:
        return {name: "None"}

    # jnp.ndarray has .nbytes; if not, compute from size * dtype.itemsize.
    try:
        nbytes = int(arr.nbytes)  # type: ignore[attr-defined]
    except (AttributeError, TypeError):
        nbytes = int(arr.size) * int(arr.dtype.itemsize)

    return {
        "name": name,
        "shape": tuple(arr.shape),
        "dtype": str(arr.dtype),
        "size": int(arr.size),
        "memory": format_bytes(nbytes),
        "memory_bytes": nbytes,
    }


# ---- Profiling primitives -------------------------------------------------- #


@contextmanager
def profile_section(
    name: str,
    enabled: Optional[bool] = None,
    print_arrays: Optional[Dict[str, jnp.ndarray]] = None,
    sync: Optional[Iterable[jnp.ndarray] | jnp.ndarray] = None,
):
    """
    Context manager for profiling a code section.

    Parameters
    ----------
    name : str
        Section name for logging.
    enabled : bool, optional
        Override global :data:`PROFILING_ENABLED`.
    print_arrays : Dict[str, jnp.ndarray], optional
        Arrays whose metadata should be printed at the start.
    sync : Iterable[jnp.ndarray] or jnp.ndarray, optional
        Array or iterable of arrays to ``block_until_ready`` at exit.

    Examples
    --------
    ::

        with profile_section("forward_pass",
                             print_arrays={"p0": p0, "dpdt": dpdt},
                             sync=[p0, dpdt]):
            result = expensive_computation()
    """
    is_enabled = PROFILING_ENABLED if enabled is None else bool(enabled)

    if not is_enabled:
        yield
        return

    mem_start = get_memory_mb()
    time_start = time.perf_counter()

    print(f"\n{'=' * 60}")
    print(f"PROFILE START: {name}")
    print(f"{'=' * 60}")
    if mem_start >= 0.0:
        print(f"Memory at start: {mem_start:.2f} MB")
    else:
        print("Memory at start: unavailable")

    if print_arrays:
        print("\nInput arrays:")
        for arr_name, arr in print_arrays.items():
            if arr is not None:
                info = array_info(arr, arr_name)
                print(
                    f"  {arr_name}: {info['shape']} {info['dtype']} ({info['memory']})"
                )

    try:
        yield
    finally:
        # If caller provided arrays to sync on, block until ready.
        if sync is not None:
            try:
                items = sync if isinstance(sync, (list, tuple)) else (sync,)
                for x in items:
                    # Only block on JAX arrays that support the method.
                    if hasattr(x, "block_until_ready"):
                        x.block_until_ready()
            except (AttributeError, RuntimeError):
                # Do not crash profiling if user passes a bad object.
                pass

        time_end = time.perf_counter()
        mem_end = get_memory_mb()
        elapsed = time_end - time_start

        print(f"\n{'-' * 60}")
        print(f"PROFILE END: {name}")
        print(f"  Elapsed time: {elapsed:.4f} s")
        if mem_end >= 0.0:
            print(f"  Memory at end: {mem_end:.2f} MB")
            if mem_start >= 0.0:
                mem_delta = mem_end - mem_start
                print(f"  Memory delta: {mem_delta:+.2f} MB")
            else:
                print("  Memory delta: unavailable (start unknown)")
        else:
            print("  Memory at end: unavailable")
        print(f"{'=' * 60}\n")


def profile_function(name: Optional[str] = None, print_args: bool = False):
    """
    Decorator for profiling functions.

    Parameters
    ----------
    name : str, optional
        Custom profile section name. Defaults to the wrapped function name.
    print_args : bool, default=False
        Whether to print shapes for JAX-array arguments.

    Returns
    -------
    Callable
        Decorator that wraps a function in :func:`profile_section`.

    Examples
    --------
    ::

        @profile_function(print_args=True)
        def my_function(x, y):
            return x + y
    """

    def decorator(func):
        """
        Wrap a function with optional profiling instrumentation.

        Parameters
        ----------
        func : Callable
            Function to wrap.

        Returns
        -------
        Callable
            Wrapped function.
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """
            Execute the wrapped function with optional profiling output.

            Parameters
            ----------
            *args
                Positional arguments forwarded to the wrapped function.
            **kwargs
                Keyword arguments forwarded to the wrapped function.

            Returns
            -------
            Any
                Result returned by the wrapped function.
            """
            fname = name or func.__name__

            if not PROFILING_ENABLED:
                return func(*args, **kwargs)

            arrays: Dict[str, jnp.ndarray] = {}
            if print_args:
                for i, arg in enumerate(args):
                    if isinstance(arg, jnp.ndarray):
                        arrays[f"arg{i}"] = arg
                for k, v in kwargs.items():
                    if isinstance(v, jnp.ndarray):
                        arrays[k] = v

            # Execute function within profiled section.
            # If the function returns JAX arrays, we’ll sync on them.
            result = None
            with profile_section(fname, print_arrays=arrays):
                result = func(*args, **kwargs)

            # Print result info and (if possible) block on output arrays.
            outputs_to_sync: list[jnp.ndarray] = []
            if isinstance(result, (tuple, list)):
                print(f"Result ({fname}):")
                for i, r in enumerate(result):
                    if isinstance(r, jnp.ndarray):
                        info = array_info(r, f"output{i}")
                        print(
                            f"  {info['name']}: {info['shape']} {info['dtype']} ({info['memory']})"
                        )
                        outputs_to_sync.append(r)
            elif isinstance(result, jnp.ndarray):
                info = array_info(result, "output")
                print(
                    f"Result ({fname}): {info['shape']} {info['dtype']} ({info['memory']})"
                )
                outputs_to_sync.append(result)

            # Best-effort sync after printing results.
            for r in outputs_to_sync:
                with suppress(Exception):
                    r.block_until_ready()

            return result

        return wrapper

    return decorator


def print_memory_summary(label: str = ""):
    """
    Print the current memory usage summary when profiling is enabled.

    Parameters
    ----------
    label : str, default=""
        Optional label included in the printed message.
    """
    if not PROFILING_ENABLED:
        return
    mem = get_memory_mb()
    if mem >= 0.0:
        print(f"[MEMORY {label}] {mem:.2f} MB")
    else:
        print(f"[MEMORY {label}] unavailable")
