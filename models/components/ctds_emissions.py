# Placeholder for ctds_emissions.py
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

    def compute_region_boundaries(self) -> jnp.ndarray:
        """
        Compute cumulative latent-dimension boundaries for each region.
        """
        latents_per_region = jnp.sum(self.list_of_dimensions, axis=1)
        return jnp.concatenate([jnp.array([0]), jnp.cumsum(latents_per_region)])

    def build_block_mask(self) -> jnp.ndarray:
        """
        Build a binary mask of shape (D, K) where mask[d, k] = 1 if neuron d
        loads on latent dimension k, based on its region and cell type.
        """
        D = self.cell_identity.shape[0]
        boundaries = self.compute_region_boundaries()
        K = int(boundaries[-1])
        mask = jnp.zeros((D, K))
        num_regions, num_types = self.list_of_dimensions.shape
        # Precompute cumulative dims within each region
        cum_dims = jnp.cumsum(jnp.concatenate([jnp.array([[0]]), self.list_of_dimensions], axis=0), axis=1)

        for d in range(D):
            region = self.region_identity[d]
            ct = self.cell_identity[d]
            start = int(boundaries[region])
            if ct > 0:
                # Known cell type indexing starts at 1
                ct_idx = ct - 1
                type_start = int(cum_dims[region, ct_idx])
                length = int(self.list_of_dimensions[region, ct_idx])
                global_start = start + type_start
                mask = mask.at[d, global_start:global_start+length].set(1)
            else:
                # Unknown: allow all dims in region
                region_size = int(boundaries[region+1] - boundaries[region])
                mask = mask.at[d, start:start+region_size].set(1)
        return mask

    def build(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        build emission parameters (C, R).

        Returns:
            C: (D, K) emission weight matrix
            R: (D, D) observation noise covariance
        """
        mask = self.build_block_mask()
        C = self.base_strength * mask
        if self.normalize:
            # normalize each row
            row_norms = jnp.linalg.norm(C, axis=1, keepdims=True) + 1e-6
            C = C / row_norms
        # enforce nonnegative weights
        C = clip_matrix(C, min_val=0.0, max_val=jnp.inf)
        D = C.shape[0]
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

        # Closed-form solution: C = (Yᵀ Ex) (∑ Exx)^-1
        S_yx = Y.T @ Ex                 # (N, D)
        S_xx = jnp.sum(Exx, axis=0)     # (D, D)
        C = S_yx @ jnp.linalg.inv(S_xx) # (N, D)

        # Apply optional constraints
        if self.sparse:
            mask = jnp.ones_like(C)  # TODO: load actual mask if needed
            C = apply_block_sparsity(C, mask)

        if self.dale:
            C = apply_dale_constraint(C, self.cell_type_identity)

        if self.unit_norm:
            C = project_to_unit_norm(C, axis=1)

        # Estimate R: empirical residual covariance
        Y_hat = Ex @ C.T                # (T, N)
        residuals = Y - Y_hat
        R_diag = jnp.mean(residuals ** 2, axis=0)  # (N,)
        R = jnp.diag(R_diag + self.noise_scale)    # (N, N)

        return C, R