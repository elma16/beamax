import logging
from typing import Union
import numpy as np
import jax.numpy as jnp
from pathlib import Path
from beamax.solvers.solverbase import Solver

try:
    from neuralop.models import FNO
    from neuralop.layers.spectral_convolution import SpectralConv
    from neuralop.layers.embeddings import GridEmbeddingND
    import torch
except ImportError as e:
    raise ImportError(
        "beamax.solvers.fno_solver_neurops requires the 'fno' extra. "
        "Install with: pip install 'beamax[fno]'"
    ) from e

logger = logging.getLogger(__name__)


class FNONeuralOpsSolver(Solver):
    """
    A solver that uses a pre-trained FNO model from the neuralop library.

    Parameters
    ----------
    save_folder : str
        Directory containing model checkpoint files.
    save_name : str
        Base name of the model checkpoint files.
    device : torch.device or str, optional
        Device used for inference.

    Notes
    -----
    This class handles loading a model checkpoint from a GPU-trained state
    onto any device (CPU or GPU) and running inference.
    """

    def __init__(
        self, save_folder: str, save_name: str, device: Union[torch.device, str] = None
    ):
        """
        Initializes the solver by loading the FNO model from a checkpoint.

        Parameters
        ----------
        save_folder : str
            The directory where the model checkpoint and metadata are stored.
        save_name : str
            The base name of the model files (e.g., 'fno_final').
        device : torch.device or str, optional
            The device to run the model on. If None, it automatically
            detects if a CUDA-enabled GPU is available, otherwise uses CPU.
        """
        # 1. Set the device for computation
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("FNONeuralOpsSolver using device: %s", self.device)

        # 2. Add custom classes to torch's safe globals for unpickling
        # This is required for loading checkpoints with custom layers.
        torch.serialization.add_safe_globals(
            [torch.nn.functional.gelu, SpectralConv, GridEmbeddingND]
        )

        # --- Manual Model Loading (Bypassing FNO.from_checkpoint) ---
        save_path = Path(save_folder)
        metadata_path = save_path / f"{save_name}_metadata.pkl"
        model_path = save_path / f"{save_name}.pt"

        if not metadata_path.exists() or not model_path.exists():
            raise FileNotFoundError(
                f"Checkpoint files not found in {save_path} with name {save_name}"
            )

        # 3. Load model initialization arguments, mapping them to the correct device
        init_kwargs = torch.load(metadata_path, map_location=self.device)

        # 4. Instantiate the model with the loaded arguments
        self.model = FNO(**init_kwargs)

        # 5. Load the model's learned weights (state dictionary)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

        # 6. Move model to the target device and set to evaluation mode
        self.model.to(self.device)
        self.model.eval()
        logger.info("FNO model loaded and configured.")

    def forward(self, p0, domain=None, sensors=None, ts=None, **kwargs):
        """
        Performs a forward pass (inference) with the loaded FNO model.

        Parameters
        ----------
        p0 : jnp.ndarray or np.ndarray
            The initial condition or input field for the model. The shape should
            be the spatial dimensions of the problem (e.g., (H, W) or (D, H, W)).

        Returns
        -------
        np.ndarray
            The model's prediction as a numpy array, with batch and channel
            dimensions removed.
        """
        p0_tensor = torch.from_numpy(p0[None, None, ...]).to(self.device)
        with torch.no_grad():
            output = self.model(p0_tensor)
        return output.cpu().numpy().squeeze()

    def time_reversal(
        self, data: Union[jnp.ndarray, np.ndarray]
    ) -> Union[jnp.ndarray, np.ndarray]:
        """
        Raise because neuralop FNO time reversal is not implemented.

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
        Raise because neuralop FNO adjoint is not implemented.

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
