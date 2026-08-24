"""Keep runtime and distribution version metadata in sync."""

from pathlib import Path
import subprocess
import sys
import tomllib

import beamax
import pytest


def test_runtime_version_matches_project_metadata():
    root = Path(__file__).resolve().parents[1]
    with (root / "pyproject.toml").open("rb") as handle:
        metadata = tomllib.load(handle)

    version = metadata["project"]["version"]
    citation = (root / "CITATION.cff").read_text()

    assert beamax.__version__ == version == "0.3.0"
    assert f"version: {version}" in citation
    assert "date-released: 2026-08-24" in citation


def test_import_beamax_does_not_eagerly_import_scipy():
    code = """
import sys
import beamax
assert "scipy" not in sys.modules
from beamax import utils
assert "scipy" not in sys.modules
assert callable(utils.unitary_fft)
assert "scipy" not in sys.modules
assert utils.Interpolator is not None
assert "scipy" in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_public_modules_and_solver_config_are_exposed():
    from beamax import coefficients
    from beamax.gb import SolverConfig, solve_hom_TR
    from beamax.solvers import HybridSolverConfig

    assert callable(coefficients.streamed_top_n_coefficients)
    assert callable(solve_hom_TR)
    assert SolverConfig.__module__ == "beamax.gb.gb_solvers"
    assert HybridSolverConfig.__module__ == "beamax.solvers.hybrid_solver"


def test_optional_solver_loader_does_not_mask_internal_errors(monkeypatch):
    import beamax.solvers as solvers

    monkeypatch.delattr(solvers, "KWaveSolver", raising=False)

    def fail_import(name):
        raise RuntimeError(f"broken module: {name}")

    monkeypatch.setattr(solvers, "import_module", fail_import)
    with pytest.raises(RuntimeError, match="broken module"):
        solvers.__getattr__("KWaveSolver")


def test_optional_solver_loader_only_relabels_missing_kwave(monkeypatch):
    import beamax.solvers as solvers

    monkeypatch.delattr(solvers, "KWaveSolver", raising=False)

    def missing_internal_dependency(name):
        raise ModuleNotFoundError(
            f"{name} needs internal_dependency", name="internal_dependency"
        )

    monkeypatch.setattr(solvers, "import_module", missing_internal_dependency)
    with pytest.raises(ModuleNotFoundError, match="internal_dependency"):
        solvers.__getattr__("KWaveSolver")

    def missing_kwave(name):
        raise ModuleNotFoundError(f"{name} needs kwave", name="kwave")

    monkeypatch.setattr(solvers, "import_module", missing_kwave)
    with pytest.raises(ImportError, match=r"install beamax\[kwave\]"):
        solvers.__getattr__("KWaveSolver")
