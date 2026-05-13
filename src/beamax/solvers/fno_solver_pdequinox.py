from typing import Union
import json
import numpy as np
from pathlib import Path

import jax
import jax.numpy as jnp
import equinox as eqx

from beamax.solvers.solverbase import Solver

try:
    from pdequinox.arch._classic_fno import ClassicFNO
except ImportError as e:
    raise ImportError(
        "beamax.solvers.fno_solver_pdequinox requires the 'fno' extra. "
        "Install with: pip install 'beamax[fno]'"
    ) from e


class FNOpdequinoxSolver(Solver):
    """
    Fourier Neural Operator solver wrapping :mod:`pdequinox`'s ``ClassicFNO``.

    Loads a pre-trained FNO model from disk and exposes it via the
    :class:`Solver` interface. Only :meth:`forward` is implemented;
    :meth:`time_reversal` and :meth:`adjoint` raise
    :class:`NotImplementedError` because there is no direct inverse for a
    trained operator.

    Parameters
    ----------
    folder : str
        Path to a directory containing:

        - ``hparams.json`` — JSON dict with keys ``num_modes``,
          ``num_spatial_dims``, ``hidden_channels``, ``in_channels``,
          ``out_channels``, ``num_blocks``.
        - ``model.eqx`` — Equinox tree of learned weights
          (``eqx.tree_serialise_leaves``).
    key : jax.random.PRNGKey, optional
        PRNG key used to build the skeleton model before weights are
        deserialised into it. Default is ``jax.random.PRNGKey(0)``.

    Notes
    -----
    Requires the ``[fno]`` extra (``pdequinox``, ``equinox``). JSON is
    lossy for tuples, so ``num_modes`` is re-coerced to a tuple on load.
    """

    def __init__(self, folder: str, key=jax.random.PRNGKey(0)):
        """
        Load a pdequinox FNO checkpoint and build a batched predictor.

        Parameters
        ----------
        folder : str
            Directory containing ``hparams.json`` and ``model.eqx``.
        key : jax.random.PRNGKey, optional
            PRNG key used to instantiate the model skeleton before
            deserializing weights.

        Raises
        ------
        FileNotFoundError
            If checkpoint files are missing when opened.
        """
        # paths to hyperparams and checkpoint
        folder = Path(folder)
        hparam_path = folder / "hparams.json"
        eqx_path = folder / "model.eqx"

        # 1) load hyperparameters
        with open(hparam_path, "r") as f:
            hp = json.load(f)
        # map JSON keys → FNO kwargs. JSON does not preserve tuples, so we
        # coerce num_modes back to a tuple (ClassicFNO concatenates it with a
        # tuple internally).
        fno_kwargs = dict(
            num_modes=tuple(hp["num_modes"]),
            num_spatial_dims=hp["num_spatial_dims"],
            hidden_channels=hp["hidden_channels"],
            in_channels=hp["in_channels"],
            out_channels=hp["out_channels"],
            num_blocks=hp["num_blocks"],
            key=key,
        )

        # 2) build skeleton (no large buffers allocated)
        skeleton = eqx.filter_eval_shape(ClassicFNO, **fno_kwargs)

        # 3) deserialize weights into skeleton
        with open(eqx_path, "rb") as f:
            model = eqx.tree_deserialise_leaves(f, skeleton)

        # 4) put into inference mode
        model = eqx.nn.inference_mode(model)

        # 5) jit‑compile a batched predictor
        self.predict = jax.jit(jax.vmap(model))

    def forward(
        self,
        p0: Union[jnp.ndarray, np.ndarray],
        domain=None,
        sensors=None,
        ts=None,
        **kwargs,
    ) -> jnp.ndarray:
        """
        Run the FNO on ``p0`` and return its prediction.

        The operator is fully learned: ``domain``, ``sensors``, and ``ts``
        are ignored (they're accepted for :class:`Solver`-interface
        compatibility).

        Parameters
        ----------
        p0 : jnp.ndarray or np.ndarray, shape (*N,)
            Input field with spatial shape matching what the model was
            trained for. A leading batch and channel axis are prepended
            internally, then squeezed out of the output.

        Returns
        -------
        jnp.ndarray, shape (*N,)
            FNO prediction.
        """
        p0 = p0[None, None, ...]
        return self.predict(p0).squeeze()

    def time_reversal(
        self, data: Union[jnp.ndarray, np.ndarray]
    ) -> Union[jnp.ndarray, np.ndarray]:
        """
        Raise because pdequinox FNO time reversal is not implemented.

        Parameters
        ----------
        data : jnp.ndarray or np.ndarray
            Sensor data or model output that would be inverted.

        Returns
        -------
        jnp.ndarray or np.ndarray
            This method never returns.

        Raises
        ------
        NotImplementedError
            Always raised for this solver.
        """
        raise NotImplementedError("Time reversal is not implemented for FNOSolver.")

    def adjoint(
        self, data: Union[jnp.ndarray, np.ndarray]
    ) -> Union[jnp.ndarray, np.ndarray]:
        """
        Raise because pdequinox FNO adjoint is not implemented.

        Parameters
        ----------
        data : jnp.ndarray or np.ndarray
            Sensor data or model output that would be used by an adjoint solve.

        Returns
        -------
        jnp.ndarray or np.ndarray
            This method never returns.

        Raises
        ------
        NotImplementedError
            Always raised for this solver.
        """
        raise NotImplementedError("Adjoint is not implemented for FNOSolver.")
