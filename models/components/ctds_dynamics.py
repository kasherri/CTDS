import jax.numpy as jnp
from typing import Tuple
from models.components.constraints import apply_dale_constraint


class CTDSDynamics:
    """
    Encapsulates CTDS-specific dynamics: block-diagonal structure, Dale's law, and cross-region constraints.
    """
    def __init__(
        self,
        list_of_dimensions: jnp.ndarray,
        cell_types: jnp.ndarray,
        base_strength: float = 0.99,
        within_region: bool = True,
        across_region: bool = True,
        noise_scale: float = 0.1
    ):
        self.list_of_dimensions = list_of_dimensions.astype(int)
        self.cell_types = cell_types
        self.base_strength = base_strength
        self.within_region = within_region
        self.across_region = across_region
        self.noise_scale = noise_scale

    def compute_region_boundaries(self) -> jnp.ndarray:
        latents_per_region = jnp.sum(self.list_of_dimensions, axis=1)
        boundaries = jnp.concatenate([jnp.array([0]), jnp.cumsum(latents_per_region)])
        return boundaries

    def build_block_diag_A(self) -> jnp.ndarray:
        boundaries = self.compute_region_boundaries()
        K = int(boundaries[-1])
        A = jnp.zeros((K, K))
        num_regions = self.list_of_dimensions.shape[0]
        for r in range(num_regions):
            start, end = int(boundaries[r]), int(boundaries[r+1])
            size = end - start
            A = A.at[start:end, start:end].set(self.base_strength * jnp.eye(size))
        return A

    def apply_across_region_constraints(self, A: jnp.ndarray) -> jnp.ndarray:
        boundaries = self.compute_region_boundaries()
        if self.list_of_dimensions.shape[0] != 2:
            return A
        start1, end1 = int(boundaries[0]), int(boundaries[1])
        start2, end2 = int(boundaries[1]), int(boundaries[2])
        D_e1 = int(self.list_of_dimensions[0, 0])
        for i in range(start2, end2):
            for j in range(start1, start1 + D_e1):
                A = A.at[i, j].set(jnp.abs(A[i, j]))
        for i in range(start1, end1):
            for j in range(start2, end2):
                A = A.at[i, j].set(0.0)
        return A

    def build(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        A = self.build_block_diag_A()
        if self.within_region:
            A = apply_dale_constraint(A, self.cell_types)
        if self.across_region:
            A = self.apply_across_region_constraints(A)
        K = A.shape[0]
        Q = jnp.eye(K) * self.noise_scale
        return A, Q




import jax.numpy as jnp
from typing import Tuple
from models.components.constraints import apply_dale_constraint, apply_block_sparsity


class CTDSDynamics:
    """
    Encapsulates CTDS-specific dynamics:
      - Block-diagonal structure by region
      - Dale's law within regions
      - Cross-region sparsity
    """
    def __init__(
        self,
        list_of_dimensions: jnp.ndarray,
        within_region: bool = True,
        across_region: bool = True,
        base_strength: float = 0.99,
        noise_scale: float = 0.1
    ):
        """
        Args:
            list_of_dimensions: (num_regions, num_cell_types) array of latent dims per region×cell-type
            within_region: enforce Dale's law within-region if True
            across_region: enforce cross-region sparsity if True
            base_strength: diagonal strength for each block
            noise_scale: variance for process noise Q
        """
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

    def _build_latent_types(self) -> jnp.ndarray:
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

    def _build_cross_region_mask(self) -> jnp.ndarray:
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
        Build the CTDS dynamics parameters.

        Returns:
            A: (K, K) transition matrix
            Q: (K, K) process noise covariance
        """
        A = self.build_block_diag_A()
        if self.across_region:
            mask = self._build_cross_region_mask()
            A = apply_block_sparsity(A, mask)
        if self.within_region:
            latent_types = self._build_latent_types()
            A = apply_dale_constraint(A, latent_types)
        # Process noise covariance
        Q = jnp.eye(A.shape[0]) * self.noise_scale
        return A, Q




class AutoRegressiveCellTypeObservations(AutoRegressiveObservations):
    """ AutoRegressive observation model with Gaussian noise, where the weights are constrained by cell type"""

    def __init__(self, K, D, M=0, lags=1, within_region_constrains=None, across_region_constraints=None, list_of_dimensions=None, **kwargs):
        super(AutoRegressiveCellTypeObservations, self).__init__(K, D, M, lags=lags)
        self.within_region_constraints = within_region_constrains or self._default_within_region_constraints # list of constraints to be enforced on the weights within regions
        self.across_region_constraints = across_region_constraints or self._default_across_region_constraints # list of constraints to be enforced on the weights across regions
        self.list_of_dimensions = list_of_dimensions.astype(int) # list of dimensions of each region
        if within_region_constrains is None:
            print("Assuming Dale's constraints for the weights within regions")
        if across_region_constraints is None:
            print("Assuming FOF-ADS cross-region constraints if 2 regions, otherwise no cross-region constraints")

    def _default_within_region_constraints(self, W):
        """ setup Dale's constraints for the weights W assuming E and I cell classes """
        M, lags = self.M, self.lags
        assert lags == 1, "Only lags==1 is supported for now"
        list_of_dimensions = self.list_of_dimensions
        within_region_constraints = []

        num_regions = list_of_dimensions.shape[0]
        for region in range(num_regions):
            dims_this_region = list_of_dimensions[region]
            assert len(dims_this_region)==2, "assuming E and I cell-types by default, but this list has either less than or more than 2 elements"
            D_e, D_i = dims_this_region[0], dims_this_region[1]
            dims_prev_regions = np.sum(list_of_dimensions[:region]) if region > 0 else 0
            # all excitatory columns should be positive and all inhibitory columns should be negative, except the diagonal elements
            for i in range(D_e + D_i):
                for j in range(D_e + D_i):
                    if i != j:
                        if i < D_e:
                            within_region_constraints.append(W[j + dims_prev_regions, i + dims_prev_regions] >= 0)
                        elif i >= D_e:
                            within_region_constraints.append(W[j + dims_prev_regions, i + dims_prev_regions] <= 0)
        return within_region_constraints
    

    def _default_across_region_constraints(self, W):
        """ Setup Dale's constraints for the weights W assuming E and I cell classes, and cross-region constraints """
            
        list_of_dimensions = self.list_of_dimensions
        num_regions = len(list_of_dimensions)
        total_latents = np.sum(list_of_dimensions)
        across_region_constraints = []
        
        # cross-region constraints, assuming there are only two regions, where one is in the cortex and the other is in the striatum (FOF and ADS, as in the paper)
        if num_regions == 2:
            latents_region_1 = np.sum(list_of_dimensions[0])
            d_e_region_1, _ = list_of_dimensions[0]
            across_region_constraints = [W[latents_region_1:total_latents, 0:d_e_region_1]>=0, W[latents_region_1:total_latents, d_e_region_1:latents_region_1]==0] # only constraining FOF->ADS connections to be excitatory

        return across_region_constraints    

    def _solve_constrained_A(self, k, ExuxuTs_k, ExuyTs_k, Sigmas_k):
        """ to solve for a constrained A matrix using cvxpy"""
       
        D, M, lags = self.D, self.M, self.lags
        assert lags==1, "Only lags==1 is supported for now"
   

        Q_inv = np.linalg.inv(Sigmas_k) # get the inverse of Sigma
        Q_inv = Q_inv/np.max(np.abs(Q_inv)) # for numerical stability
        L = np.linalg.cholesky(Q_inv) # perform a cholesky decomposition
        kron_ExuxuTs = np.kron((ExuxuTs_k+self.J0_k).T, np.eye(D))

        W = cp.Variable((D, D*lags+M))

        # define the constraints
        constraints = self.within_region_constraints(W) + self.across_region_constraints(W)

        # define the objective function
        objective = cp.Minimize(cp.quad_form((L.T@W).flatten(), kron_ExuxuTs) - 2*cp.trace(Q_inv@W@(ExuyTs_k+self.h0_k)))
        prob = cp.Problem(objective, constraints)
        prob.solve(solver = cp.MOSEK, verbose = False, warm_start = True,) # solve the problem
        if prob.status != 'optimal': # check if the problem is solved
            print("Warning: M step for A failed to converge!")
        Wk = W.value
        return Wk
    
    def m_step(self, expectations, datas, inputs, masks, tags,
               continuous_expectations=None, **kwargs):
        """Compute M-step for Gaussian Auto Regressive Observations.

        If continuous_expectations is not None, this function will
        compute an exact M-step using the expected sufficient statistics for the
        continuous states. In this case, we ignore the prior provided by (J0, h0),
        because the calculation is exact. continuous_expectations should be a tuple of
        (Ex, Ey, ExxT, ExyT, EyyT).

        If continuous_expectations is None, we use datas and expectations,
        and (optionally) the prior given by (J0, h0). In this case, we estimate the sufficient
        statistics using datas, which is typically a single sample of the continuous
        states from the posterior distribution.
        """
        K, D, M, lags = self.K, self.D, self.M, self.lags

        # Collect sufficient statistics
        if continuous_expectations is None:
            ExuxuTs, ExuyTs, EyyTs, Ens = self._get_sufficient_statistics(expectations, datas, inputs)
        else:
            ExuxuTs, ExuyTs, EyyTs, Ens = \
                self._extend_given_sufficient_statistics(expectations, continuous_expectations, inputs)

        # Solve the linear regressions
        As = np.zeros((K, D, D * lags))
        Vs = np.zeros((K, D, M))
        bs = np.zeros((K, D))
        Sigmas = np.zeros((K, D, D))
        
        for k in range(K):
            ExuxuTs_k = ExuxuTs[k][:D*lags+M, :D*lags+M]
            ExuyTs_k = ExuyTs[k][:D*lags+M]
            self.J0_k = self.J0[k][:D*lags+M, :D*lags+M]
            self.h0_k = self.h0[k][:D*lags+M]

            Wk = self._solve_constrained_A(k, ExuxuTs_k, \
                                            ExuyTs_k, self.Sigmas[k])
            As[k] = Wk[:, :D * lags]
            Vs[k] = Wk[:, D * lags:D*lags+M]
            bs[k] = np.zeros(D) 

            # Solve for the MAP estimate of the covariance
            EWxyT =  Wk @ ExuyTs_k
            sqerr = EyyTs[k] - EWxyT.T - EWxyT + Wk @ ExuxuTs_k @ Wk.T
            nu = self.nu0 + Ens[k]
            Sigmas[k] = (sqerr + self.Psi0) / (nu + D + 1) 

        # If any states are unused, set their parameters to a perturbation of a used state
        unused = np.where(Ens < 1)[0]
        used = np.where(Ens > 1)[0]
        if len(unused) > 0:
            for k in unused:
                i = npr.choice(used)
                As[k] = As[i] + 0.01 * npr.randn(*As[i].shape)
                Vs[k] = Vs[i] + 0.01 * npr.randn(*Vs[i].shape)
                bs[k] = bs[i] + 0.01 * npr.randn(*bs[i].shape)
                Sigmas[k] = Sigmas[i]

        # Update parameters via their setter
        self.As = As
        self.Vs = Vs
        self.bs = bs
        self._sqrt_Sigmas[0] = np.linalg.cholesky(Sigmas[0])
        self.Sigmas = Sigmas
