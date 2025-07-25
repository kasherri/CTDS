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


    


    
    
   