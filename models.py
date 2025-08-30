"""
Cell-Type Dynamical Systems (CTDS) implementation.

This module provides the core CTDS model class for analyzing neural population
dynamics with cell-type-specific constraints and Dale's law enforcement.
"""
import jax
import jax.numpy as jnp
import chex
from fastprogress.fastprogress import progress_bar
from typing import Optional, Any, Callable, Tuple, Union
from jaxtyping import Array, Float, Real
from functools import partial
import logging

from params import (
    ParamsCTDS, SufficientStats, ParamsCTDSDynamics, 
    ParamsCTDSEmissions, ParamsCTDSInitial, ParamsCTDSConstraints, 
    M_Step_State
)
from abc import ABC, abstractmethod
from inference import InferenceBackend, DynamaxLGSSMBackend
from tensorflow_probability.substrates.jax import distributions as tfd
from utlis import (
    estimate_J, blockwise_NMF, solve_constrained_QP, 
    blockwise_NNLS, jaxOpt_NNLS
)
from dynamax.ssm import SSM

jax.config.update("jax_enable_x64", True) 

class BaseCTDS(ABC):
    """
    Abstract base class for Cell-Type Dynamical Systems.
    
    This class defines the interface for CTDS models with different
    inference backends and constraint handling approaches.

    Parameters
    ----------
    backend : InferenceBackend
        Inference backend (e.g., DynamaxLGSSMBackend).
    dynamics_fn : Callable
        Function computing transition dynamics (may include nonlinearities).
    emissions_fn : Callable
        Function computing observation model.
    constraints : ParamsCTDSConstraints, optional
        Cell-type and Dale's law constraints.
    """
    def __init__(self,
                 backend: InferenceBackend,
                 dynamics_fn: Callable,
                 emissions_fn: Callable,
                 constraints: Optional[ParamsCTDSConstraints] = None):
        self.backend = backend
        self.dynamics_fn = dynamics_fn
        self.emissions_fn = emissions_fn
        self.constraints = constraints

    @abstractmethod
    def initialize(self, Y: chex.Array, mask: chex.Array, **kwargs) -> ParamsCTDS:
        """
        Initialize model parameters from observed data.
        
        Parameters
        ----------
        Y : Array, shape (N, T)
            Observed emission sequence.
        mask : Array, shape (N,)
            Boolean mask for valid observations.
        **kwargs
            Additional initialization arguments.
            
        Returns
        -------
        ParamsCTDS
            Initialized model parameters.
        """
        ...

    def infer(self, params: ParamsCTDS, emissions: chex.Array, 
              inputs: Optional[chex.Array] = None):
        """
        Run inference using the attached backend.
        
        Parameters
        ----------
        params : ParamsCTDS
            Model parameters.
        emissions : Array, shape (T, N)
            Observed emission sequence.
        inputs : Array, shape (T, U), optional
            Exogenous input sequence.
            
        Returns
        -------
        Posterior statistics from the inference backend.
        """
        return self.backend.infer(params, emissions, inputs)
    
    @abstractmethod
    def m_step(self, params: ParamsCTDS, stats: SufficientStats) -> ParamsCTDS:
        """
        Update model parameters given sufficient statistics.

        Parameters
        ----------
        params : ParamsCTDS
            Current model parameters.
        stats : SufficientStats
            Sufficient statistics from E-step.

        Returns
        -------
        ParamsCTDS
            Updated model parameters.
        """
        ...


def _psd_project(M: chex.Array, floor: float) -> chex.Array:
    """
    Project matrix to positive semi-definite with eigenvalue floor.
    
    Parameters
    ----------
    M : Array, shape (D, D)
        Input matrix to project.
    floor : float
        Minimum eigenvalue threshold.
        
    Returns
    -------
    Array, shape (D, D)
        Projected positive semi-definite matrix.
        
    Notes
    -----
    Ensures numerical stability by flooring eigenvalues and
    symmetrizing the matrix before eigendecomposition.
    """
    M = 0.5 * (M + M.T)
    w, V = jnp.linalg.eigh(M)
    w = jnp.maximum(w, floor)
    return (V * w) @ V.T


def _gauge_fix_clamped(A: chex.Array, C: chex.Array, Q: chex.Array, 
                       smin: float = 0.3, smax: float = 3.0, 
                       q_floor: float = 1e-6, eps: float = 1e-8) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """
    Apply column-norm gauge fixing with clamped scales.
    
    Parameters
    ----------
    A : Array, shape (D, D)
        Dynamics matrix.
    C : Array, shape (N, D) 
        Emission matrix.
    Q : Array, shape (D, D)
        Process noise covariance.
    smin, smax : float
        Min/max scale factors for gauge fixing.
    q_floor : float
        Eigenvalue floor for Q after scaling.
    eps : float
        Numerical stability constant.
        
    Returns
    -------
    A_scaled : Array, shape (D, D)
        Gauge-fixed dynamics matrix.
    C_scaled : Array, shape (N, D)
        Gauge-fixed emission matrix.
    Q_scaled : Array, shape (D, D)
        Gauge-fixed process covariance.
        
    Notes
    -----
    Applies similarity transform to normalize column scales while
    preserving Dale's law signs in the emission matrix.
    """
    # Column-norm gauge on C with clamped scales; positive diagonal S preserves Dale signs
    s = jnp.maximum(jnp.linalg.norm(C, axis=0), eps)        # (D,)
    s = jnp.clip(s, smin, smax)
    S = jnp.diag(s)
    S_inv = jnp.diag(1.0 / s)
    A = S @ A @ S_inv
    C = C @ S_inv
    Q = S @ Q @ S
    Q = _psd_project(Q, q_floor)                            # re-floor after similarity
    return A, C, Q

class CTDS(SSM):
    """
    Cell-Type Dynamical System with Dale's law constraints.
    
    A linear state-space model for neural population dynamics that enforces
    biologically plausible sign constraints based on cell type identity.

    Parameters
    ----------
    emission_dim : int
        Number of observed neurons (N).
    cell_types : Array, shape (K,)
        Cell type labels as contiguous integers [0, 1, ..., K-1].
    cell_sign : Array, shape (K,)
        Sign constraints: +1 for excitatory, -1 for inhibitory cell types.
    cell_type_dimensions : Array, shape (K,)
        Latent dimensions allocated to each cell type.
    cell_type_mask : Array, shape (N,)
        Cell type assignment for each observed neuron.
    region_identity : Array, shape (N,), optional
        Brain region identity for each neuron (for future multi-region support).
    inputs_dim : int, optional
        Dimension of exogenous inputs.
    state_dim : int, optional
        Total latent state dimension (computed from cell_type_dimensions if None).

    Notes
    -----
    The CTDS model implements the linear state-space equations:
    
    **State evolution:**
    $$x_{t+1} = A x_t + B u_t + \\varepsilon_t, \\quad \\varepsilon_t \\sim \\mathcal{N}(0, Q)$$
    
    **Observations:**
    $$y_t = C x_t + D u_t + \\eta_t, \\quad \\eta_t \\sim \\mathcal{N}(0, R)$$
    
    where Dale's law constraints enforce that:
    - Excitatory neurons (cell_sign[k] = +1) have non-negative connection weights
    - Inhibitory neurons (cell_sign[k] = -1) have non-positive connection weights
    
    The model supports block-structured connectivity patterns reflecting
    cell-type-specific dynamics and connectivity.

    Raises
    ------
    AssertionError
        If cell_types is not a contiguous range [0, 1, ..., K-1].
    ValueError
        If cell_type_mask contains invalid cell type indices.
        
    Examples
    --------
    >>> # Create CTDS model with 2 cell types
    >>> cell_types = jnp.array([0, 1])
    >>> cell_sign = jnp.array([1, -1])  # excitatory, inhibitory
    >>> cell_type_dimensions = jnp.array([3, 2])  # 3 + 2 = 5 total latent dims
    >>> cell_type_mask = jnp.array([0, 0, 0, 1, 1])  # 3 exc + 2 inh neurons
    >>> 
    >>> ctds = CTDS(
    ...     emission_dim=5,
    ...     cell_types=cell_types,
    ...     cell_sign=cell_sign, 
    ...     cell_type_dimensions=cell_type_dimensions,
    ...     cell_type_mask=cell_type_mask
    ... )
    """
    def __init__(self, 
                 emission_dim: int,
                 cell_types: chex.Array,
                 cell_sign: chex.Array,
                 cell_type_dimensions: chex.Array,
                 cell_type_mask: chex.Array,
                 region_identity: Optional[chex.Array] = None,
                 inputs_dim: Optional[int] = None,
                 state_dim: Optional[int] = None):
        # Validate cell type consistency
        assert jnp.all(cell_types == jnp.arange(cell_types.shape[0])), \
            "cell_types must be a contiguous range of integers from 0 to K-1 for correct indexing."      
        self.state_dim = state_dim
        self.emission_dim = emission_dim
        self.region_identity = region_identity
        self.inputs_dim = inputs_dim
        self.constraints = ParamsCTDSConstraints(
            cell_types=cell_types,
            cell_sign=cell_sign,
            cell_type_dimensions=cell_type_dimensions,
            cell_type_mask=cell_type_mask
        )
    # ---------- Shapes ----------
    @property
    def emission_shape(self) -> Tuple[int]:
        """Shape of emission observations."""
        return (self.emission_dim,)
    
    @property
    def _emission_dim(self) -> Tuple[int]:
        """Internal emission dimension property."""
        return (self.emission_dim,)
    
    @property
    def inputs_shape(self) -> Optional[Tuple[int]]:
        """Shape of exogenous inputs, if any."""
        return None if self.inputs_dim is None else (self.inputs_dim,)

    # ---------- Distributions ----------
    def initial_distribution(
        self,
        params: ParamsCTDS,
        inputs: Optional[jnp.ndarray]
    ) -> tfd.Distribution:
        """p(z1) = N(mu0, Sigma0)."""
        mu0 = params.initial.mean              # shape (K,)
        Sigma0 = params.initial.cov            # shape (K,K) or (K,)
        return tfd.MultivariateNormalFullCovariance(mu0, jnp.atleast_2d(Sigma0))

    def transition_distribution(
        self,
        params: ParamsCTDS,
        state: jnp.ndarray,                     # shape (K,)
        inputs: Optional[jnp.ndarray]
    ) -> tfd.Distribution:
        """
        p(z_{t+1} | z_t) = N(A z_t + B u_t, Q).
        If you do not use inputs, set B u_t = 0.
        """
        A = params.dynamics.weights             # shape (K,K)
        Q = params.dynamics.cov                 # shape (K,K) or (K,)
        if inputs is not None and params.dynamics.input_weights is not None:
            B = params.dynamics.input_weights   # shape (K,U) or time-varying
            mean = A @ state + B @ inputs
        else:
            mean = A @ state
        return tfd.MultivariateNormalFullCovariance(mean, jnp.atleast_2d(Q))

    def emission_distribution(
        self,
        params: ParamsCTDS,
        state: jnp.ndarray,                     # shape (K,)
        inputs: Optional[jnp.ndarray]=None
    ) -> tfd.Distribution:
        """p(y_t | z_t) = N(C z_t, R)."""
        C = params.emissions.weights            # shape (N,K)
        R = params.emissions.cov                # shape (N,N) or (N,)
        mean = C @ state
        return tfd.MultivariateNormalFullCovariance(mean, jnp.atleast_2d(R))
    #TODO: make intialize dynamics and blockwise nmf jittable
    def initialize_dynamics(self, V_list: list, U: chex.Array) -> ParamsCTDSDynamics:
        """
        Initialize dynamics parameters from block factorization results.
        
        Parameters
        ----------
        V_list : list of Arrays
            List of (N, D_k) factor matrices for each cell type k.
        U : Array, shape (N, D)
            Concatenated emission weight matrix.
            
        Returns
        -------
        ParamsCTDSDynamics
            Initialized dynamics parameters with Dale's law sign corrections.
            
        Notes
        -----
        Constructs the dynamics matrix A as A = V_dale^T @ U where V_dale
        applies sign corrections: positive for excitatory cell types,
        negative for inhibitory cell types to respect Dale's law.
        
        Raises
        ------
        AssertionError
            If V_list length doesn't match number of cell types or if
            dimensions are inconsistent across matrices.
        """
        # Validate input consistency
        assert len(V_list) == len(self.constraints.cell_types), \
            "Number of V matrices must match number of cell types."
        
        V_dale_list = []
        dynamics_mask_list = []
        
        for i in range(len(V_list)):
            # Check dimension consistency
            assert V_list[i].shape[1] == self.constraints.cell_type_dimensions[i], \
                "entries in cell_type_dimensions should correspond to entries in cell_types and cell_sign."
            
            if self.constraints.cell_sign[i] == -1:
                # Inhibitory: negative weights
                V_dale_list.append(-V_list[i])
                dynamics_mask_list.append(jnp.full((self.constraints.cell_type_dimensions[i],), -1))
            else:
                # Excitatory: positive weights  
                V_dale_list.append(V_list[i])
                dynamics_mask_list.append(jnp.full((self.constraints.cell_type_dimensions[i],), 1))

        # Validate consistent number of rows
        assert all(V.shape[0] == V_list[0].shape[0] for V in V_dale_list), \
            "All V matrices must have the same number of rows (neurons)."
        
        # Concatenate along columns to form full emission matrix
        V_dale = jnp.concatenate(V_dale_list, axis=1)  # shape (N, D) 
        A = V_dale.T @ U
        D = V_dale.shape[1]
        Q = 1e-6 * jnp.eye(D)  # Small process noise initialization

        return ParamsCTDSDynamics(
            weights=A,
            cov=Q, 
            dynamics_mask=jnp.concatenate(dynamics_mask_list, axis=0)
        )

    def initialize_emissions(self, Y: chex.Array, U_list: list, state_dim: int) -> ParamsCTDSEmissions:
        """
        Initialize emission parameters from block factorization results.
        
        Parameters
        ----------
        Y : Array, shape (N, T)
            Observed emission sequence.
        U_list : list of Arrays
            List of (N_k, D_k) emission factor matrices for each cell type k.
        state_dim : int
            Total latent state dimension D.
            
        Returns
        -------
        ParamsCTDSEmissions
            Initialized emission parameters with block-diagonal structure.
            
        Notes
        -----
        Constructs a block-diagonal emission matrix C by concatenating
        cell-type-specific blocks with appropriate zero padding. The 
        observation noise covariance R is initialized as diagonal with
        empirical variances from the data.
        
        Raises
        ------
        AssertionError
            If the sum of latent dimensions doesn't match state_dim.
        """
        # Validate dimension consistency
        assert sum(U.shape[1] for U in U_list) == state_dim, \
            "Latent dimensions do not match state dimension."
        
        # Build block-diagonal emission matrix
        C_blocks = []
        col_start = 0
        emission_dims = []
        left_padding_dims = []
        right_padding_dims = []
        emiss_start = 0
        
        for U in U_list:
            N_type, K_type = U.shape
            
            # Create padded block: [zeros_left, U, zeros_right]
            left_pad = jnp.zeros((N_type, col_start))
            right_pad = jnp.zeros((N_type, state_dim - col_start - K_type))
            
            left_padding_dims.append(left_pad.shape)
            right_padding_dims.append(right_pad.shape)

            C_block = jnp.concatenate([left_pad, U, right_pad], axis=1)
            C_blocks.append(C_block)

            emission_dims.append((emiss_start, emiss_start + N_type))

            emiss_start += N_type
            col_start += K_type
        
        # Concatenate vertically to form complete emission matrix
        C = jnp.concatenate(C_blocks, axis=0)  # Shape: (N, D)
        
        # Initialize observation noise from empirical variances
        mean_Y = jnp.mean(Y, axis=1, keepdims=True)
        var_Y = jnp.mean((Y - mean_Y)**2, axis=1)
        var_Y = jnp.maximum(var_Y, 1e-6)  # Floor for numerical stability
        
        R = jnp.diag(var_Y)  # Shape: (N, N)

        return ParamsCTDSEmissions(
            weights=C, 
            cov=R, 
            emission_dims=jnp.array(emission_dims), 
            left_padding_dims=jnp.array(left_padding_dims), 
            right_padding_dims=jnp.array(right_padding_dims)
        )


    def initialize(self, observations: chex.Array) -> ParamsCTDS:
        """
        Initialize CTDS parameters from observed emission data.
        
        Parameters
        ----------
        observations : Array, shape (N, T)
            Observed emission sequence with N neurons over T timesteps.
            
        Returns
        -------
        ParamsCTDS
            Initialized model parameters.
            
        Notes
        -----
        Initialization proceeds via:
        1. Estimate linear recurrent connectivity matrix J using constrained regression
        2. Apply blockwise NMF to J respecting cell-type structure  
        3. Build block-diagonal emission and dynamics matrices
        4. Set small initial state variance and process noise
        
        The method respects Dale's law constraints throughout the initialization.
        
        Raises
        ------
        AssertionError
            If observations shape doesn't match expected emission_dim.
            
        Examples
        --------
        >>> observations = jnp.ones((10, 100))  # 10 neurons, 100 timesteps
        >>> params = ctds.initialize(observations)
        """
        # Validate input shape
        assert observations.shape[0] == self.emission_dim, \
            "Observations should have shape (N, T) where N is number of neurons and T is number of time steps."
        
        Y = observations  # (N, T)
        
        # Create Dale's law mask for observed neurons
        # Uses fancy indexing: for each neuron, lookup the sign for that neuron's cell type
        dale_mask = jnp.where(
            self.constraints.cell_sign[self.constraints.cell_type_mask] == 1, 
            True, False
        )  
        
        # Step 1: Estimate linear recurrent matrix J with Dale's law constraints
        J = estimate_J(Y, dale_mask)

        # Step 2: Apply blockwise NMF to decompose J
        block_factors = blockwise_NMF(J, self.constraints)
        state_dim = int(jnp.sum(self.constraints.cell_type_dimensions))
        
        # TODO: Make this JAX-compatible (currently uses list comprehension)
        U_list = [tup[0] for tup in block_factors]  # Emission factors for each cell type
        V_list = [tup[1] for tup in block_factors]  # Dynamics factors for each cell type
        
        # Step 3: Initialize parameter components
        initial = ParamsCTDSInitial(
            mean=jnp.zeros(state_dim), 
            cov=1e-2 * jnp.eye(state_dim)
        )

        emissions = self.initialize_emissions(Y, U_list, state_dim)
        dynamics = self.initialize_dynamics(V_list, emissions.weights)
        
        return ParamsCTDS(
            initial=initial,
            dynamics=dynamics, 
            emissions=emissions, 
            constraints=self.constraints, 
            observations=observations
        )
    
    @partial(jax.jit, static_argnums=0)
    def e_step(self,
        params: ParamsCTDS,
        emissions: chex.Array,
        inputs: Optional[chex.Array] = None,
    ) -> SufficientStats:
        """
        Expectation step: compute sufficient statistics via RTS smoothing.

        Parameters
        ----------
        params : ParamsCTDS
            Current model parameters.
        emissions : Array, shape (T, N)
            Observed emission sequence.
        inputs : Array, shape (T, U), optional
            Exogenous input sequence.

        Returns
        -------
        SufficientStats
            Sufficient statistics for M-step parameter updates:
            - latent_mean: E[x_t], shape (T, D)
            - latent_second_moment: E[x_t x_t^T], shape (T, D, D)
            - cross_time_moment: E[x_{t+1} x_t^T], shape (T-1, D, D)
            - loglik: marginal log-likelihood
            - T: number of timesteps
            
        Notes
        -----
        Uses the RTS (Rauch-Tung-Striebel) smoother to compute posterior
        moments of the latent states given all observations. These moments
        are sufficient for updating model parameters in the M-step.
        """
        return DynamaxLGSSMBackend.e_step(params, emissions, inputs)

    def initialize_m_step_state(self, params: ParamsCTDS, 
                               props: Optional[ParamsCTDS] = None) -> M_Step_State:
        """Initialize M-step state tracking for convergence diagnostics."""
        return M_Step_State(0, jnp.array([0.0]), jnp.array([0.0]), jnp.array([0.0]), jnp.array([0.0]))

    def m_step(self,
               params: ParamsCTDS,
               props: Optional[ParamsCTDS], 
               batch_stats: SufficientStats,
               m_step_state: Optional[M_Step_State] 
               ) -> Tuple[ParamsCTDS, M_Step_State]:
        """
        Maximization step: update parameters using sufficient statistics.

        Parameters
        ----------
        params : ParamsCTDS
            Current model parameters.
        props : ParamsCTDS, optional
            Unused parameter for compatibility with base class.
        batch_stats : SufficientStats
            Sufficient statistics averaged across batch sequences.
        m_step_state : M_Step_State, optional
            Convergence tracking state.

        Returns
        -------
        params_updated : ParamsCTDS
            Updated model parameters after M-step.
        m_step_state_updated : M_Step_State
            Updated convergence tracking state.

        Notes
        -----
        The M-step updates parameters by solving constrained optimization problems:
        
        **Dynamics matrix A**: Solved via constrained QP respecting Dale's law:
        $$\\min_A \\|A\\|_F^2 \\text{ s.t. } A_{ij} \\geq 0 \\text{ if neuron } j \\text{ is excitatory}$$
        
        **Process noise Q**: Updated via residual covariance after A update.
        
        **Emission matrix C**: Block-wise non-negative least squares (NNLS) 
        maintaining cell-type structure and Dale's law sign constraints.
        
        **Observation noise R**: Diagonal residual variance estimation.
        
        **Initial state**: Set to first timestep posterior moments.
        
        A gauge-fixing step normalizes column scales to improve numerical stability.

        Raises
        ------
        AssertionError
            If batch_stats arrays don't have proper batch dimensions.
        """
        # Validate batch dimensions
        assert batch_stats.latent_mean.ndim == 3, \
            "latent_mean should have shape (batch_size, T, D). For single sequence, use shape (1, T, D)"
        assert batch_stats.latent_second_moment.ndim == 4, \
            "latent_second_moment should have shape (batch_size, T, D, D). For single sequence, use shape (1, T, D, D)"
        assert batch_stats.cross_time_moment.ndim == 4, \
            "cross_time_moment should have shape (batch_size, T-1, D, D). For single sequence, use shape (1, T-1, D, D)"

        # Average sufficient statistics across batch sequences
        latent_mean = jnp.mean(batch_stats.latent_mean, axis=0)                     
        latent_second_moment = jnp.mean(batch_stats.latent_second_moment, axis=0)  
        cross_time_moment = jnp.mean(batch_stats.cross_time_moment, axis=0)        
        
        stats = SufficientStats(
            latent_mean=latent_mean,
            latent_second_moment=latent_second_moment,
            cross_time_moment=cross_time_moment,
            loglik=0.0,  # Unused in M-step
            T=latent_mean.shape[0]
        )
        
        T = stats.T

        # Update dynamics matrix A via constrained QP
        S11 = jnp.sum(stats.latent_second_moment[1:], axis=0)   # Sum over time: (D, D)
        S10 = jnp.sum(stats.cross_time_moment, axis=0).T       # Cross-time: (D, D) 
        S00 = jnp.sum(stats.latent_second_moment[:-1], axis=0)  # Lagged: (D, D)
        
        # Set up constraint masks for Dale's law
        masks = jnp.logical_not(jnp.eye(S10.shape[0], dtype=jnp.bool_))  # Off-diagonal mask
        cell_type_mask = jnp.where(params.dynamics.dynamics_mask == -1, False, True)  # Excitatory mask
        
        # Solve constrained QP for each row of A 
        vmap_solver1 = jax.vmap(solve_constrained_QP, in_axes=(None, 0, 1, 0, 1))
        H = 2.0 * S00 + 1e-6 * jnp.eye(S00.shape[0])  # Regularized Gram matrix
        A_t = vmap_solver1(H, -2.0 * S10, masks, cell_type_mask, params.dynamics.weights)
        A = jnp.transpose(A_t)
        delta_A = jnp.concatenate([m_step_state.delta_A, jnp.array([jnp.linalg.norm(A - params.dynamics.weights)])])

        # Update process noise Q via residual covariance
        AS00 = A @ S00
        AS00AT = AS00 @ A.T
        Q = (S11 - A @ S10.T - S10 @ A.T + AS00AT) / (T - 1.0)      
        delta_Q = jnp.concatenate([m_step_state.delta_Q, jnp.array([jnp.linalg.norm(Q - params.dynamics.cov)])])

        # Update emission matrix C via blockwise NNLS
        Y_obs = params.observations.T  # (T, N)
        X = stats.latent_mean          # (T, D)
        C = blockwise_NNLS(
            Y_obs, X, 
            params.emissions.left_padding_dims, 
            params.emissions.right_padding_dims, 
            params.emissions.emission_dims, 
            params.constraints.cell_type_dimensions, 
            params.emissions.weights
        )
        delta_C = jnp.concatenate([m_step_state.delta_C, jnp.array([jnp.linalg.norm(C - params.emissions.weights)])])

        # Update observation noise R (diagonal)
        Y = params.observations                          # (N, T)
        S_yx = Y @ stats.latent_mean                     # (N, D)
        YY_diag = jnp.sum(Y * Y, axis=1)                 # (N,)
        cross = jnp.sum(C * S_yx, axis=1)                # (N,)
        quad = jnp.einsum('nd,tdk,nk->n', C, stats.latent_second_moment, C)  # (N,)

        R_diag = (YY_diag - 2.0 * cross + quad) / stats.T
        R = jnp.diag(R_diag)
        delta_R = jnp.concatenate([m_step_state.delta_R, jnp.array([jnp.linalg.norm(R - params.emissions.cov)])])

        # Apply gauge fixing for numerical stability
        A, C, Q = _gauge_fix_clamped(A, C, Q, 0.3, 3.0, 1e-6)
        
        # Build updated parameter structures
        dynamics = ParamsCTDSDynamics(
            weights=A, 
            cov=Q, 
            dynamics_mask=params.dynamics.dynamics_mask
        )
        emissions = ParamsCTDSEmissions(
            weights=C, 
            cov=R, 
            emission_dims=params.emissions.emission_dims, 
            left_padding_dims=params.emissions.left_padding_dims, 
            right_padding_dims=params.emissions.right_padding_dims
        )

        # Update initial state to first timestep posterior
        initial_mean = stats.latent_mean[0]  
        initial_cov = stats.latent_second_moment[0]  
        initial = ParamsCTDSInitial(mean=initial_mean, cov=initial_cov)
        
        # Update convergence tracking
        iter_count = m_step_state.iteration + 1
        updated_m_step_state = M_Step_State(iter_count, delta_A, delta_C, delta_Q, delta_R)
        
        return (
            ParamsCTDS(initial, dynamics, emissions, 
                      constraints=params.constraints, 
                      observations=params.observations),  
            updated_m_step_state
        )

    
    def fit_em(self, 
               params: ParamsCTDS,
               batch_emissions: Union[Real[Array, "T N"], Real[Array, "B T N"]],
               batch_inputs: Optional[chex.Array] = None,
               num_iters: int = 100,
               verbose: bool = True) -> Tuple[ParamsCTDS, chex.Array]:
        """
        Expectation–Maximization training loop for the Cell-Type Dynamical System.

        Parameters
        ----------
        params : ParamsCTDS
            Initial model parameters.
        batch_emissions : Array, shape (T, N) or (B, T, N)
            Observation sequence(s). Single sequence (T, N) or batch (B, T, N).
        batch_inputs : Array, shape (T, U) or (B, T, U), optional
            Exogenous input sequence(s) aligned to emissions.
        num_iters : int, default=100
            Maximum number of EM iterations.
        verbose : bool, default=True
            Whether to display progress and convergence information.

        Returns
        -------
        params_fitted : ParamsCTDS
            Updated parameters after EM convergence.
        log_probs : Array, shape (num_iters_run,)
            Marginal log-likelihood trace across iterations.

        Notes
        -----
        The E-step computes smoothed moments $E[x_t]$, $E[x_t x_t^T]$, $E[x_{t+1}x_t^T]$
        using the RTS smoother. The M-step updates $(A,B)$ and $(C,D)$ via 
        sign-constrained least squares; $(Q,R)$ from residual covariances.
        
        Convergence is detected when the relative change in log-likelihood
        falls below 1e-6.

        Raises
        ------
        ValueError
            If input shapes are inconsistent or contain NaNs/Infs.

        Examples
        --------
        >>> params_fitted, lls = ctds.fit_em(
        ...     params0, emissions, num_iters=50, verbose=True
        ... )
        >>> print(f"Final log-likelihood: {lls[-1]:.2f}")
        """
        def em_step(params, m_step_state):
            """Perform one EM step: E-step + M-step."""
            # E-step: compute sufficient statistics (vectorized over batch)
            vmap_solver = jax.vmap(DynamaxLGSSMBackend.e_step, in_axes=(None, 0, 0))
            batch_stats, lls = vmap_solver(params, batch_emissions, batch_inputs)
            
            # Compute total log-likelihood including priors
            lp = self.log_prior(params) + lls.sum()
            
            # M-step: update parameters
            params, m_step_state = self.m_step(params, None, batch_stats, m_step_state)

            return params, m_step_state, lp

        # Initialize and run first EM iteration
        log_probs = []
        m_step_state = self.initialize_m_step_state(params, None)
        params, m_step_state, marginal_logprob = em_step(params, m_step_state)
        log_probs.append(marginal_logprob)

        # Main EM loop with optional progress bar
        pbar = progress_bar(range(1, num_iters)) if verbose else range(1, num_iters)
        for i in pbar:
            params, m_step_state, marginal_logprob = em_step(params, m_step_state)
            
            if verbose:
                logging.info(f"Iteration {i}: log-likelihood = {marginal_logprob:.6f}")
            
            # Check for convergence
            convergence_criteria = jnp.abs(marginal_logprob - log_probs[-1]) / jnp.abs(log_probs[-1])
            log_probs.append(marginal_logprob)
            
            if convergence_criteria < 1e-6:
                if verbose:
                    logging.info(f"Converged at iteration {i}")
                break
                
        return params, jnp.array(log_probs)

    def sample(self, params: ParamsCTDS, key: jax.random.PRNGKey, 
               num_timesteps: int, inputs: Optional[chex.Array] = None) -> Tuple[Float[Array, "T D"], Float[Array, "T N"]]:
        """
        Generate samples from the CTDS model.

        Parameters
        ----------
        params : ParamsCTDS
            Model parameters for sampling.
        key : jax.random.PRNGKey
            Random key for stochastic operations.
        num_timesteps : int
            Number of timesteps to sample.
        inputs : Array, shape (T, U), optional
            Optional exogenous input sequence.

        Returns
        -------
        latent_states : Array, shape (T, D)
            Sampled latent state trajectories.
        emissions : Array, shape (T, N)
            Sampled emission observations.

        Notes
        -----
        Generates a forward sample by simulating the state-space model:
        
        $$x_{t+1} = A x_t + B u_t + \\varepsilon_t$$
        $$y_t = C x_t + D u_t + \\eta_t$$
        
        where noise terms are drawn from their respective Gaussian distributions.

        Examples
        --------
        >>> key = jax.random.PRNGKey(0)
        >>> states, obs = ctds.sample(params, key, num_timesteps=100)
        >>> print(f"States shape: {states.shape}, Obs shape: {obs.shape}")
        """
        return super().sample(params, key, num_timesteps, inputs)

    def forecast(self, params: ParamsCTDS, emissions: chex.Array, 
                 num_steps: int, inputs: Optional[chex.Array] = None,
                 key: Optional[jax.random.PRNGKey] = None) -> chex.Array:
        """
        Forecast future emissions given observed history.

        Parameters
        ----------
        params : ParamsCTDS
            Fitted model parameters.
        emissions : Array, shape (T_obs, N)
            Observed emission history for conditioning.
        num_steps : int
            Number of future timesteps to forecast.
        inputs : Array, shape (T_obs + num_steps, U), optional
            Exogenous inputs for observed + forecast periods.
        key : jax.random.PRNGKey, optional
            Random key for stochastic forecasts. If None, returns mean predictions.

        Returns
        -------
        forecasts : Array, shape (num_steps, N)
            Predicted future emissions.

        Notes
        -----
        Computes forecast by:
        1. Running the smoother on observed data to get final posterior state
        2. Forward-simulating the dynamics for `num_steps` timesteps
        3. Generating emission predictions via the observation model
        
        If `key` is provided, includes process and observation noise.
        Otherwise returns the deterministic mean trajectory.

        Examples
        --------
        >>> # Deterministic forecast
        >>> y_pred = ctds.forecast(params, emissions[-50:], num_steps=20)
        >>> 
        >>> # Stochastic forecast with uncertainty
        >>> key = jax.random.PRNGKey(42)
        >>> y_pred = ctds.forecast(params, emissions[-50:], num_steps=20, key=key)
        """
        # Get final posterior state from observed data
        posterior = DynamaxLGSSMBackend.smoother(params, emissions, 
                                                inputs[:emissions.shape[0]] if inputs is not None else None)
        
        # Extract final state estimate
        final_mean = posterior[0][-1]  # Last smoothed mean
        final_cov = posterior[1][-1]   # Last smoothed covariance
        
        # Initialize forecast arrays
        A = params.dynamics.weights
        C = params.emissions.weights
        Q = params.dynamics.cov
        R = params.emissions.cov
        
        forecasts = []
        current_mean = final_mean
        
        for t in range(num_steps):
            # Predict next state
            if inputs is not None and params.dynamics.input_weights is not None:
                B = params.dynamics.input_weights
                input_idx = emissions.shape[0] + t
                current_mean = A @ current_mean + B @ inputs[input_idx]
            else:
                current_mean = A @ current_mean
                
            # Add process noise if stochastic
            if key is not None:
                key, subkey = jax.random.split(key)
                if Q.ndim == 1:  # Diagonal covariance
                    noise = jax.random.normal(subkey, current_mean.shape) * jnp.sqrt(Q)
                else:  # Full covariance
                    noise = jax.random.multivariate_normal(subkey, jnp.zeros_like(current_mean), Q)
                current_mean = current_mean + noise
            
            # Generate emission prediction
            # Generate emission prediction
            emission_mean = C @ current_mean
            # TODO: Add emission input weights (D matrix) if needed in future
                
            # Add observation noise if stochastic
            if key is not None:
                key, subkey = jax.random.split(key)
                if R.ndim == 1:  # Diagonal covariance
                    emission_noise = jax.random.normal(subkey, emission_mean.shape) * jnp.sqrt(R)
                else:  # Full covariance
                    emission_noise = jax.random.multivariate_normal(subkey, jnp.zeros_like(emission_mean), R)
                emission_mean = emission_mean + emission_noise
            
            forecasts.append(emission_mean)
            
        return jnp.stack(forecasts, axis=0)

    def log_prob(self, params: ParamsCTDS, states: chex.Array, 
                 emissions: chex.Array, inputs: Optional[chex.Array] = None) -> float:
        """
        Compute log-probability of state-emission trajectory.

        Parameters
        ----------
        params : ParamsCTDS
            Model parameters.
        states : chex.Array, shape (T, D)
            Latent state trajectory.
        emissions : chex.Array, shape (T, N)
            Observation sequence.
        inputs : chex.Array, optional, shape (T, M)
            Exogenous input sequence.

        Returns
        -------
        log_prob : float
            Joint log-probability of states and emissions.
        """
        return super().log_prob(params, states, emissions, inputs)

    def filter(self, params: ParamsCTDS, emissions: chex.Array, 
               inputs: Optional[chex.Array] = None):
        """
        Run forward Kalman filter.

        Parameters
        ----------
        params : ParamsCTDS
            Model parameters.
        emissions : chex.Array, shape (T, N)
            Observation sequence.
        inputs : chex.Array, optional, shape (T, M)
            Exogenous input sequence.

        Returns
        -------
        FilterResults
            Filtered means, covariances, and log-likelihood.
        """
        return DynamaxLGSSMBackend.filter(params, emissions, inputs)

    @staticmethod
    def smoother(params: ParamsCTDS, emissions: chex.Array, 
                 inputs: Optional[chex.Array] = None):
        """
        Run RTS smoother for posterior inference.

        Parameters
        ----------
        params : ParamsCTDS
            Model parameters.
        emissions : chex.Array, shape (T, N)
            Observation sequence.
        inputs : chex.Array, optional, shape (T, M)
            Exogenous input sequence.

        Returns
        -------
        SmootherResults
            Smoothed means, covariances, and log-likelihood.
        """
        return DynamaxLGSSMBackend.smoother(params, emissions, inputs)

    def marginal_log_prob(self, params: ParamsCTDS, emissions: chex.Array, 
                         inputs: Optional[chex.Array] = None) -> float:
        """
        Compute marginal log-likelihood of emissions.

        Parameters
        ----------
        params : ParamsCTDS
            Model parameters.
        emissions : chex.Array, shape (T, N)
            Observation sequence.
        inputs : chex.Array, optional, shape (T, M)
            Exogenous input sequence.

        Returns
        -------
        log_prob : float
            Marginal log-likelihood p(y_{1:T}).
        """
        return super().marginal_log_prob(params, emissions, inputs)












