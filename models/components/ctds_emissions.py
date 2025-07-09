import jax.numpy as jnp
from typing import Tuple, Optional
from models.components.constraints import (
    clip_matrix,
    apply_block_sparsity,
    project_to_unit_norm,
)


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

    def _latent_metadata(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Return arrays of latent regions and cell types."""
        latent_regions = []
        latent_types = []
        for r, dims in enumerate(self.list_of_dimensions):
            # assume ordering [excitatory, inhibitory]
            d_e = int(dims[0]) if dims.shape[0] > 0 else 0
            d_i = int(dims[1]) if dims.shape[0] > 1 else 0
            latent_regions += [r] * (d_e + d_i)
            latent_types += [1] * d_e
            latent_types += [2] * d_i
        return jnp.array(latent_regions), jnp.array(latent_types)

    def _build_mask(self) -> jnp.ndarray:
        """Construct block-sparse mask for emissions."""
        total_latents = int(jnp.sum(self.list_of_dimensions))
        total_neurons = len(self.cell_identity)
        latent_regions, latent_types = self._latent_metadata()
        mask = jnp.zeros((total_neurons, total_latents))
        for n in range(total_neurons):
            r = int(self.region_identity[n])
            ct = int(self.cell_identity[n])
            allow = (latent_regions == r) & ((latent_types == ct) | (ct == 0))
            mask = mask.at[n, allow].set(1.0)
        return mask



    def build(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Build emission matrix C and noise matrix R using per-region block structure.

        Returns:
            C: (D, K) emission weight matrix
            R: (D, D) diagonal observation noise covariance
        """
        num_regions = self.list_of_dimensions.shape[0]
        total_latents = int(jnp.sum(self.list_of_dimensions))
        total_neurons = len(self.cell_identity)

        mask = self._build_mask()
        C = mask * self.base_strength
        if self.normalize:
            C = project_to_unit_norm(C, axis=1)
        C = clip_matrix(C, 0.0, jnp.inf)

        
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

        Exx_sum = jnp.sum(Exx, axis=0)
        C_hat = (Y.T @ Ex) @ jnp.linalg.inv(Exx_sum)

        mask = self._build_mask()
        C_hat = apply_block_sparsity(C_hat, mask)
        C_hat = clip_matrix(C_hat, 0.0, jnp.inf)
        if self.normalize:
            C_hat = project_to_unit_norm(C_hat, axis=1)

        pred = Ex @ C_hat.T
        resid = Y - pred
        R_diag = jnp.mean(resid ** 2, axis=0)
        R = jnp.diag(R_diag + self.noise_scale)
        return C_hat, R
