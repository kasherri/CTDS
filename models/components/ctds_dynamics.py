import jax.numpy as jnp
from typing import Tuple
from models.components.constraints import apply_dale_constraint, apply_block_sparsity
from jaxopt import OSQP


class CTDSDynamics:
    """
    Encapsulates CTDS-specific dynamics:
      - Block-diagonal structure by region
      - Dale's law within regions
      - Cross-region sparsity
    
    Args:
        list_of_dimensions: (num_regions, num_cell_types) jnp.ndarray of latent dims per region×cell-type
        within_region: enforce Dale's law within-region if True
        across_region: enforce cross-region sparsity if True
        base_strength: diagonal strength for each block
        noise_scale: variance for process noise Q    

    """
    def __init__(
        self,
        list_of_dimensions: jnp.ndarray,
        within_region: bool = True,
        across_region: bool = True,
        base_strength: float = 0.99,
        noise_scale: float = 0.1
    ):
    
        self.list_of_dimensions = list_of_dimensions.astype(int)
        self.within_region = within_region
        self.across_region = across_region
        self.base_strength = base_strength
        self.noise_scale = noise_scale

    def compute_region_boundaries(self) -> jnp.ndarray:
        """
        Compute cumulative boundaries of latent dims per region.
        """
        latents_per_region = jnp.sum(self.list_of_dimensions, axis=1)
        return jnp.concatenate([jnp.array([0]), jnp.cumsum(latents_per_region)])

    def build_block_diag_A(self) -> jnp.ndarray:
        """
        Construct a block-diagonal transition matrix A.
        """
        boundaries = self.compute_region_boundaries()
        K = int(boundaries[-1])
        A = jnp.zeros((K, K))
        for r in range(self.list_of_dimensions.shape[0]):
            start, end = int(boundaries[r]), int(boundaries[r+1])
            size = end - start
            A = A.at[start:end, start:end].set(self.base_strength * jnp.eye(size))
        return A

    def build_latent_types(self) -> jnp.ndarray:
        """
        Create a length-K array of latent cell-type labels:
          1 = excitatory, 2 = inhibitory, 0 = unknown
        """
        types = []
        # assume first cell type dim is excitatory, second inhibitory
        for dims in self.list_of_dimensions:
            if dims.shape[0] < 2:
                raise ValueError("Expected two cell-type dims per region")
            D_e, D_i = int(dims[0]), int(dims[1])
            types += [1] * D_e
            types += [2] * D_i
        return jnp.array(types)

    def build_cross_region_mask(self) -> jnp.ndarray:
        """
        Construct a mask that zeroes out region2->region1 connections and
        keeps all others (for two-region case). For more regions, default to identity.
        """
        boundaries = self.compute_region_boundaries()
        K = int(boundaries[-1])
        mask = jnp.ones((K, K))
        num_regions = self.list_of_dimensions.shape[0]
        if num_regions == 2:
            start1, end1 = int(boundaries[0]), int(boundaries[1])
            start2, end2 = int(boundaries[1]), int(boundaries[2])
            # Zero region2 -> region1
            mask = mask.at[start1:end1, start2:end2].set(0)
        return mask

    def build(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        build the CTDS dynamics parameters A and Q.

        Returns:
            A: (K, K) transition matrix
            Q: (K, K) process noise covariance
        """
        A = self.build_block_diag_A()
        if self.across_region:
            mask = self.build_cross_region_mask()
            A = apply_block_sparsity(A, mask)
        if self.within_region:
            latent_types = self.build_latent_types()
            A = apply_dale_constraint(A, latent_types)
        # Process noise covariance
        Q = jnp.eye(A.shape[0]) * self.noise_scale
        return A, Q
    
   
    def m_step(self, posterior_dict) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        to do
        """
        Ex = posterior_dict["Ex"]
        Exx = posterior_dict["Exx"]
        Exnx = posterior_dict["Exnx"]
        T, D = Ex.shape

        # setup qp problem
        X_t = Ex[:-1]        # (T-1, D)
        X_tp1 = Ex[1:]       # (T-1, D)
        XtX = X_t.T @ X_t    # (D, D)
        XtY = X_t.T @ X_tp1  # (D, D)

        P = jnp.kron(XtX, jnp.eye(D))        # (D^2, D^2)
        q = -XtY.T.flatten()                 # (D^2,)

        # set inequality constraints
        cell_types = self._build_latent_types()  # (D,)
        sparsity_mask = self._build_cross_region_mask() if self.across_region else jnp.ones((D, D))
        G, h = self._build_inequality_constraints(D, cell_types, sparsity_mask)

        # solve 
        solver = OSQP()
        sol = solver.run(params_obj=(P, q, G, h, None)).params
        A = sol.primal.reshape((D, D))  # reshape flat vector to matrix

        pred = A @ X_t.T             # (D, T-1)
        err = X_tp1.T - pred         # (D, T-1)
        Q = (err @ err.T) / (T - 1)  # empirical residual covariance
        Q += self.noise_scale * jnp.eye(D)  # regularization

        return A, Q


    


    def _build_inequality_constraints(self, D, cell_types, sparsity_mask):
        G_list, h_list = [], []
        for i in range(D):
            for j in range(D):
                index = i * D + j
                if sparsity_mask[i, j] == 0:
                    row = jnp.zeros(D*D).at[index].set(1.0)
                    G_list.append(row); h_list.append(0.0)
                    row = jnp.zeros(D*D).at[index].set(-1.0)
                    G_list.append(row); h_list.append(0.0)
                elif cell_types[j] == 1:
                    row = jnp.zeros(D*D).at[index].set(-1.0)
                    G_list.append(row); h_list.append(0.0)
                elif cell_types[j] == 2:
                    row = jnp.zeros(D*D).at[index].set(1.0)
                    G_list.append(row); h_list.append(0.0)
        G = jnp.stack(G_list, axis=0)
        h = jnp.array(h_list)
        return G, h
    
    
    
   