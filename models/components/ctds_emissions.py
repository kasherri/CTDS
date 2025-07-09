import jax
import jax.numpy as jnp
from typing import Tuple, Optional
from models.components.constraints import clip_matrix, apply_dale_constraint, apply_block_sparsity


class CTDSEmissions:
    """
    Encapsulates CTDS-specific emission structure:
      - Block-sparsity by region and cell type
      - Positive weights constraint (clip)
      - Optional normalization 
    """
    def __init__(
        self,
        cell_identity: jnp.ndarray,
        list_of_dimensions: jnp.ndarray,
        region_identity: Optional[jnp.ndarray] = None,
        base_strength: float = 1.0,
        noise_scale: float = 0.5,
        normalize: bool = False
    ):
        """
        Args:
            cell_identity: (D,) values in 0 (unknown), >=1 (cell-type index)
            list_of_dimensions: (R, T) latent dims per region×cell-type
            region_identity: (D,) values in 0..R-1, default all zeros
            base_strength: initial emission weight magnitude
            noise_scale: observation noise variance
            normalize: whether to normalize each neuron
        """
        self.cell_identity = cell_identity.astype(int)
        self.list_of_dimensions = list_of_dimensions.astype(int)
        self.region_identity = (
            region_identity.astype(int)
            if region_identity is not None
            else jnp.zeros_like(cell_identity, dtype=int)
        )
        self.base_strength = base_strength
        self.noise_scale = noise_scale
        self.normalize = normalize



    def build(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Build emission matrix C and noise matrix R using per-region block structure.

        Returns:
            C: (D, K) emission weight matrix
            R: (D, D) diagonal observation noise covariance
        """
        num_regions = self.list_of_dimensions.shape[0]
        num_neurons_per_region = jnp.bincount(self.region_identity).max()
        neurons_per_type = num_neurons_per_region // 2
        total_latents = int(jnp.sum(self.list_of_dimensions))
        total_neurons = len(self.cell_identity)

        C = jnp.zeros((total_neurons, total_latents))
        latent_offset = 0

        #TO DO BUILD
        #

        
        # Diagonal observation noise
        D = total_neurons
        R = jnp.eye(D) * self.noise_scale
        return C, R

    def m_step(self, posterior, Y: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        M-step for emissions: estimate readout matrix C and noise covariance R.

        Args:
            posterior: dict of smoothed moments:
                - "Ex": (T, D) posterior mean of latents
                - "Exx": (T, D, D) posterior 2nd moment of latents
            Y: (T, N) observed neural activity

        Returns:
            C: (N, D) emission matrix
            R: (N, N) noise covariance (diagonal)
        """
        Ex = posterior["Ex"]            # (T, D)
        Exx = posterior["Exx"]          # (T, D, D)
        T, N = Y.shape
        D = Ex.shape[1]

        #TO DO REST

        return C, R