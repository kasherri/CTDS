import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Any
from models.components.constraints import clip_matrix, apply_dale_constraint, apply_block_sparsity




class CTDSEmissions:
    """
    JAX implementation of cell-type dynamical system (CTDS) Gaussian emissions.
    - Block-sparse, non-negative emission matrix C (block per cell type/region)
    - Dale's law: each block is non-negative
    - Compatible with Dynamax
    """

    def __init__(
        self,
        N: int,
        D: int,
        cell_identity: jnp.ndarray,
        region_identity: jnp.ndarray,
        list_of_dimensions: jnp.ndarray,
        key: jax.random.PRNGKey,
    ):
        """
        Args:
            N: Number of observed neurons
            D: Total latent dimension
            cell_identity: (N,) array, 0 for unknown, >=1 for known cell type
            region_identity: (N,) array, region index for each neuron 
            list_of_dimensions: (num_regions, num_cell_types) array, latent dims per region/cell type
            key: JAX PRNGKeys
        """
        self.N = N 
        self.D = D
        self.cell_identity = cell_identity 
        self.region_identity = region_identity
        self.list_of_dimensions = list_of_dimensions
        self.num_regions, self.num_cell_types = list_of_dimensions.shape 

        # Initialize parameters
        self.key = key
        self.params = self.initialize()

    def initialize(self, Y, inputs=None, masks=None, tags=None):
        """
        JAX-compatible initialization using Scikit-JAX PCA.
        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: (C, R)
        """
        # Optionally handle missing data imputation here if needed
        # datas = [interpolate_data_jax(data, mask) for data, mask in zip(datas, masks)]

        # Run JAX PCA-based initialization
        pca = self._initialize_with_pca_jax(Y, inputs=inputs)

        # Emission matrix (C) and noise covariance (R)
        C = jnp.array(pca.components_).T   # shape (N, D) 
        R = jnp.diag(jnp.array(pca.noise_variance_))  # shape (N, N)

        return C, R

      

    def apply(self, x: jnp.ndarray, params: Dict[str, Any]) -> jnp.ndarray:
        """
        Forward pass: y = Cx + d
        Args:
            x: (T, D) latent states
            params: dict with "C", "d"
        Returns:
            y: (T, N) predicted observations
        """
        return jnp.dot(x, params["C"].T) + params["d"]

    def log_likelihood(self, y: jnp.ndarray, x: jnp.ndarray, params: Dict[str, Any]) -> float:
        """
        Compute log-likelihood of y given x and emission params.
        Args:
            y: (T, N) observed data
            x: (T, D) latent states
            params: dict with "C", "d", "Sigma"
        Returns:
            loglik: scalar
        """
        pred = self.apply(x, params)
        Sigma = params["Sigma"]
        # Diagonal Gaussian log-likelihood
        ll = -0.5 * jnp.sum(jnp.log(2 * jnp.pi * Sigma))
        ll -= 0.5 * jnp.sum(((y - pred) ** 2) / Sigma)
        return ll

    def get_params(self) -> Dict[str, Any]:
        return self.params
    
    def _initialize_with_pca_jax(self, Y, inputs=None):
        Keff=1 #keff is effective number of emission subspaces so since we are not doing switching we use 1
        # Convert all data to jax arrays
        Y = [jnp.array(data) for data in Y]
        if inputs is not None:
            inputs = [jnp.array(inp) for inp in inputs]

        # If there are inputs, regress them out first (optional, can be skipped if not needed)
        if self.M > 0 and inputs is not None:
            # Simple least squares regression using jax
            X = jnp.vstack(inputs)
            Y = jnp.vstack(Y)
            coef = jnp.linalg.lstsq(X, Y, rcond=None)[0]
            Fs = jnp.tile(coef[None, :, :], (Keff, 1, 1))
            self.Fs = Fs
            resids = [data - inp @ Fs[0].T for data, inp in zip(Y, inputs)]
        else:
            resids = Y

        # Stack all residuals for PCA
        all_resids = jnp.vstack(resids)
        n_components = min(self.D * Keff, self.N)

        # Run JAX PCA
        pca = JaxPCA(n_components=n_components)
        pca.fit(all_resids)

        # Assign each state a random projection of these dimensions
        Cs, ds = [], []
        for k in range(Keff):
            # Random orthogonal weights
            weights = jnp.linalg.qr(jnp.array(npr.randn(self.D, self.D * Keff)))[0]
            Cs.append((weights @ pca.components_.T).T)
            ds.append(pca.mean_)

        self.Cs = jnp.stack(Cs)
        self.ds = jnp.stack(ds)

        # Optionally, set noise variance (diagonal covariance)
        self.inv_etas = pca.noise_variance_ + 1e-4 * jnp.eye(self.N)

        return pca


    #posterior is continous right now but will later extend for switching models and mixture models
    def m_step(self,
           posterior: dict,
           Y: jnp.ndarray,
           list_of_dims: jnp.ndarray,
           region_identity: jnp.ndarray,
           cell_identity: jnp.ndarray,
           fit_intercept: bool = True,
           initial_C: jnp.ndarray = None,
           num_iters: int = 100) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        M-step for CTDS emissions: estimates emission matrix C and diagonal noise covariance R.

        Args:
            posterior: dict with keys:
                - "Ex": (T, D) posterior means of latent states
                - "Exx": (T, D, D) posterior second moments (E[x x^T])
            Y: (T, N) observed neural activity
            list_of_dims: (num_regions, num_cell_types), latent dims per region/type
            region_identity: (N,), maps each neuron to its region
            cell_identity: (N,), maps each neuron to its cell type (1-indexed, 0=unknown)
            fit_intercept: whether to estimate a bias term
            initial_C: optional initial guess for C (unused in this implementation)
            num_iters: ignored (used inside OSQP, not needed with jaxopt default)

        Returns:
            C: (N, D) emission matrix
            R: (N, N) diagonal noise covariance matrix
        """
        Ex = posterior["Ex"]       # (T, D)
        Exx = posterior["Exx"]     # (T, D, D)
        T, D = Ex.shape
        N = Y.shape[1]

        # Augment Ex for intercept if needed
        if fit_intercept:
            Ex_aug = jnp.hstack([Ex, jnp.ones((T, 1))])  # (T, D+1)
            ExxT = Ex_aug.T @ Ex_aug                     # (D+1, D+1)
            ExyT = Ex_aug.T @ Y                          # (D+1, N)
        else:
            ExxT = jnp.sum(Exx, axis=0) + Ex.T @ Ex      # (D, D)
            ExyT = Ex.T @ Y                              # (D, N)

        # Fit emission matrix C using batched constrained regression
        C, bias = fit_constrained_linear_regression_batched(
            ExxT=ExxT,
            ExyT=ExyT,
            list_of_dims=list_of_dims,
            region_identity=region_identity,
            cell_identity=cell_identity,
            fit_intercept=fit_intercept
        )

        # Compute predicted emissions
        Y_hat = Ex @ C.T + (bias[None, :] if fit_intercept else 0.0)  # (T, N)

        # Residual variance → diagonal noise covariance R
        residuals = Y - Y_hat
        R_diag = jnp.mean(residuals ** 2, axis=0)
        R = jnp.diag(R_diag)  # (N, N)

        return C, R



