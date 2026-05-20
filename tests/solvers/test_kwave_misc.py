"""
Unit tests for thin validation / dispatch paths in ``kwave_solver`` that the
existing integration tests don't reach. Pure-Python, no k-Wave required.
"""

import json
import types

import numpy as np
import pytest

from beamax.solvers import kwave_solver as kwave_solver_module
from beamax.solvers.kwave_solver import KWaveSolver


# ---------------------------------------------------------------------------
# _normalize_kwave_binary_path
# ---------------------------------------------------------------------------


def test_normalize_binary_path_missing_directory_raises(tmp_path):
    """An empty directory (no binary inside) must raise FileNotFoundError."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="does not exist"):
        kwave_solver_module._normalize_kwave_binary_path(empty_dir, device="cpu")


def test_normalize_binary_path_nonexistent_file_raises(tmp_path):
    """A nonexistent explicit file path must raise FileNotFoundError."""
    bogus = tmp_path / "nope"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        kwave_solver_module._normalize_kwave_binary_path(bogus, device="cpu")


# ---------------------------------------------------------------------------
# _metadata_marks_bad_darwin_omp
# ---------------------------------------------------------------------------


def test_metadata_marks_bad_omp_returns_false_when_no_metadata(tmp_path):
    """No metadata file -> returns False (don't reject the binary)."""
    binary = tmp_path / "kspaceFirstOrder-OMP"
    binary.write_text("", encoding="utf-8")
    assert kwave_solver_module._metadata_marks_bad_darwin_omp(binary) is False


def test_metadata_marks_bad_omp_returns_false_on_bad_json(tmp_path):
    """Malformed metadata JSON -> returns False (don't crash)."""
    binary = tmp_path / "kspaceFirstOrder-OMP"
    binary.write_text("", encoding="utf-8")
    binary.with_name("kspaceFirstOrder-OMP_metadata.json").write_text(
        "{not: valid json",
        encoding="utf-8",
    )
    assert kwave_solver_module._metadata_marks_bad_darwin_omp(binary) is False


def test_metadata_marks_bad_omp_returns_false_for_good_version(tmp_path):
    """A metadata file pointing at a known-good version is not flagged."""
    binary = tmp_path / "kspaceFirstOrder-OMP"
    binary.write_text("", encoding="utf-8")
    binary.with_name("kspaceFirstOrder-OMP_metadata.json").write_text(
        json.dumps({"version": "v1.4.1", "url": "https://example/x"}),
        encoding="utf-8",
    )
    assert kwave_solver_module._metadata_marks_bad_darwin_omp(binary) is False


# ---------------------------------------------------------------------------
# _domain_has_nonzero_absorption
# ---------------------------------------------------------------------------


def test_domain_has_nonzero_absorption_none():
    """alpha_coeff = None → no absorption."""
    domain = types.SimpleNamespace(alpha_coeff=None)
    assert kwave_solver_module._domain_has_nonzero_absorption(domain) is False


def test_domain_has_nonzero_absorption_scalar_zero():
    """alpha_coeff = 0.0 (scalar) → no absorption."""
    domain = types.SimpleNamespace(alpha_coeff=0.0)
    assert kwave_solver_module._domain_has_nonzero_absorption(domain) is False


def test_domain_has_nonzero_absorption_scalar_nonzero():
    """alpha_coeff = 0.5 (scalar) → absorption present."""
    domain = types.SimpleNamespace(alpha_coeff=0.5)
    assert kwave_solver_module._domain_has_nonzero_absorption(domain) is True


def test_domain_has_nonzero_absorption_callable():
    """A callable alpha_coeff is evaluated on the domain grid."""

    def alpha(x):
        return 0.5

    def _eval(_):
        return np.array([0.5])

    domain = types.SimpleNamespace(alpha_coeff=alpha, _eval=_eval)
    assert kwave_solver_module._domain_has_nonzero_absorption(domain) is True


# ---------------------------------------------------------------------------
# _coerce_sensor_data_layout — full error-path matrix
# ---------------------------------------------------------------------------


class TestCoerceSensorDataLayout:
    def _mask(self, n: int) -> np.ndarray:
        m = np.zeros((4, 4))
        m.flat[:n] = 1
        return m

    def test_invalid_data_layout_string_raises(self):
        with pytest.raises(ValueError, match="Invalid data_layout"):
            KWaveSolver._coerce_sensor_data_layout(
                np.zeros((2, 5)),
                self._mask(2),
                data_layout="bogus",
                op_name="test",
            )

    def test_empty_source_mask_raises(self):
        with pytest.raises(ValueError, match="at least one active"):
            KWaveSolver._coerce_sensor_data_layout(
                np.zeros((2, 5)),
                np.zeros((4, 4)),
                data_layout="auto",
                op_name="test",
            )

    def test_1d_data_with_multiple_sources_raises(self):
        with pytest.raises(ValueError, match="1D data"):
            KWaveSolver._coerce_sensor_data_layout(
                np.zeros((5,)),
                self._mask(2),
                data_layout="auto",
                op_name="test",
            )

    def test_1d_data_with_single_source_is_expanded(self):
        """A 1D record with Ns=1 is reshaped to (1, Nt)."""
        out = KWaveSolver._coerce_sensor_data_layout(
            np.arange(5.0),
            self._mask(1),
            data_layout="auto",
            op_name="test",
        )
        assert out.shape == (1, 5)

    def test_3d_data_raises(self):
        with pytest.raises(ValueError, match="2D"):
            KWaveSolver._coerce_sensor_data_layout(
                np.zeros((2, 3, 4)),
                self._mask(2),
                data_layout="auto",
                op_name="test",
            )

    def test_explicit_ns_nt_with_wrong_rows_raises(self):
        with pytest.raises(ValueError, match="Ns="):
            KWaveSolver._coerce_sensor_data_layout(
                np.zeros((5, 2)),
                self._mask(2),  # rows=5, ns=2
                data_layout="ns_nt",
                op_name="test",
            )

    def test_explicit_nt_ns_with_wrong_cols_raises(self):
        with pytest.raises(ValueError, match="Ns="):
            KWaveSolver._coerce_sensor_data_layout(
                np.zeros((5, 3)),
                self._mask(2),  # cols=3, ns=2
                data_layout="nt_ns",
                op_name="test",
            )

    def test_square_auto_layout_keeps_input(self):
        """When Ns == Nt the layout is ambiguous; auto returns the input as-is."""
        data = np.arange(4.0).reshape(2, 2)
        out = KWaveSolver._coerce_sensor_data_layout(
            data,
            self._mask(2),
            data_layout="auto",
            op_name="test",
        )
        assert np.array_equal(out, data)

    def test_auto_layout_indeterminate_raises(self):
        """Auto-inference must fail for shapes that match neither orientation."""
        with pytest.raises(ValueError, match="could not infer"):
            KWaveSolver._coerce_sensor_data_layout(
                np.zeros((7, 9)),
                self._mask(2),
                data_layout="auto",
                op_name="test",
            )


# ---------------------------------------------------------------------------
# _build_adjoint_source / _binary_name small helpers
# ---------------------------------------------------------------------------


def test_binary_name_cpu_vs_gpu():
    """The OMP binary is for CPU and the CUDA binary is for GPU."""
    assert kwave_solver_module._binary_name("cpu") == "kspaceFirstOrder-OMP"
    assert kwave_solver_module._binary_name("gpu") == "kspaceFirstOrder-CUDA"


def test_default_kwave_binary_path_returns_path_with_known_name(monkeypatch):
    """The default-path helper returns a path with the expected binary file name."""
    import types as _types

    fake_kwave = _types.SimpleNamespace(BINARY_PATH="/tmp/kwave_bin")
    monkeypatch.setitem(__import__("sys").modules, "kwave", fake_kwave)
    out = kwave_solver_module._default_kwave_binary_path("cpu")
    assert out.name == "kspaceFirstOrder-OMP"
    assert str(out).startswith("/tmp/kwave_bin")


if __name__ == "__main__":
    pytest.main([__file__])
