#from ast import Tuple
import jax
from fastprogress.fastprogress import progress_bar
from typing import Optional, Any, Callable, Tuple, Union
from jaxtyping import Array, Float, Real
import jax.numpy as jnp
from params import ParamsCTDS, SufficientStats, ParamsCTDSDynamics, ParamsCTDSEmissions, ParamsCTDSInitial, ParamsCTDSConstraints, M_Step_State
from abc import ABC, abstractmethod
from inference import InferenceBackend, DynamaxLGSSMBackend
from tensorflow_probability.substrates.jax import distributions as tfd

from utlis import estimate_J,blockwise_NMF, solve_constrained_QP, blockwise_NNLS
from functools import partial
from dynamax.ssm import SSM
jax.config.update("jax_enable_x64", True)
from jax.experimental.layout import DeviceLocalLayout, Layout 

#maybe later implement as subclass of dynamax SSM. right now dont see a reason to do so
class BaseCTDS(ABC):
    def __init__(self,
                 backend: InferenceBackend,
                 dynamics_fn: Callable,
                 emissions_fn: Callable,
                 constraints: Optional[ParamsCTDSConstraints]=None,
                 #m_step: Optional[Callable] = None
                 ):
        """
        Abstract base class for Cell-Type Dynamical Systems (CTDS).

        Args:
            backend: An inference backend (e.g., DynamaxLGSSMInference)
            dynamics_fn: Function to compute transition dynamics (possibly nonlinear or cell-type-dependent)
            emissions_fn: Function to compute observation model (usually linear Gaussian)
        """
        self.backend = backend
        self.dynamics_fn = dynamics_fn
        self.emissions_fn = emissions_fn
        self.constraints = constraints

    @abstractmethod
    def initialize(self, Y: Array, mask: Array, **kwargs) -> ParamsCTDS:
        """Model-specific initialization."""
        ...

    def infer(self, params: ParamsCTDS, emissions: Array, inputs: Optional[Array] = None):
        """Run inference using the attached backend."""
        return self.backend.infer(params, emissions, inputs)
    
    @abstractmethod
    def m_step(self, params: ParamsCTDS, stats: SufficientStats) -> ParamsCTDS:
        """
        Update model parameters given sufficient statistics.

        Args:
            params: current parameters
            stats: sufficient statistics computed from inference backend

        Returns:
            Updated ParamsCTDS
        """
    ...



class CTDS(SSM):
    """
    #TODO: add region identity 
    """
    def __init__(self, 
                 emission_dim: int,
                 cell_types:  Array, #(k,) where k is number of cell types. contains cell type labels
                 cell_sign: Array, #(k,) where k is number of cell types. values are 1 for excitatory, -1 for inhibitory
                 cell_type_dimensions: Array, #(k,) where k is number of cell types. contains latent dimensions for each cell type
                 cell_type_mask: Array, #(N,) where N is number of neurons. contains cell type label for each neuron
                 region_identity: Optional[Array] = None,
                 inputs_dim: Optional[int] = None,  # dimension of control inputs
                 state_dim: Optional[int] = None,  # total state dimension across all cell types


                 ):
        #First check that cell_types is a contiguous range
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
        return (self.emission_dim,)
    @property
    def _emission_dim(self) -> Tuple[int]:
        return (self.emission_dim,)
    @property
    def inputs_shape(self) -> Optional[Tuple[int]]:
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
    def initialize_dynamics(self, V_list, U) -> ParamsCTDSDynamics:
        #build V_dale 
        #V_list should have have the same length and order as cell_types and cell_sign
        assert len(V_list) == len(self.constraints.cell_types), "Number of V matrices must match number of cell types."
        V_dale_list = []
        dynamics_mask_list = []
        for i in range(len(V_list)):
            if self.constraints.cell_sign[i] == -1:
                V_dale_list.append(-V_list[i])  # Invert sign for inhibitory cell types

                #first check that number of elements in V_list[i] is equal to number of corresponding cell type dimensions
                assert V_list[i].shape[1] == self.constraints.cell_type_dimensions[i], "entries in cell_type_dimensions should correspond to entries in cell_types and cell_sign."
                dynamics_mask_list.append(jnp.full((self.constraints.cell_type_dimensions[i],), -1))  # -1 for inhibitory
            else:
                V_dale_list.append(V_list[i])
                #first check that number of elements in V_list[i] is equal to number of corresponding cell type dimensions
                assert V_list[i].shape[1] == self.constraints.cell_type_dimensions[i], "entries in cell_type_dimensions should correspond to entries in cell_types and cell_sign."
                dynamics_mask_list.append(jnp.full((self.constraints.cell_type_dimensions[i],), 1))  # 1 for excitatory

        #first check that all matrices in V_dale_list have the same number of rows
        assert all(V.shape[0] == V_list[0].shape[0] for V in V_dale_list), "All V matrices must have the same number of rows (neurons)."
        
        #then concatenate them along columns
        V_dale = jnp.concatenate(V_dale_list, axis=1) #shape (N, D) 
        A=V_dale.T @ U
        D=V_dale.shape[1]
        Q=1e-2 * jnp.eye(D)

        return ParamsCTDSDynamics(weights=A,cov=Q, dynamics_mask=jnp.concatenate(dynamics_mask_list, axis=0))

    def initialize_emissions(self, Y, U_list, state_dim) -> ParamsCTDSEmissions:
        #the number of columns K of each U in U_list should add up to state_dim D
        assert sum(U.shape[1] for U in U_list) == state_dim, "Latent dimensions do not match state dimension."
        
        #------------Compute C---------------------
        # create block-diagonal emission matrix
        C_blocks = []
        col_start = 0
        emission_dims = []
        left_padding_dims = []
        right_padding_dims = []
        emiss_start=0
        for U in U_list:
            N_type, K_type = U.shape
            
            # create padded block: [zeros_left, U, zeros_right]
            left_pad = jnp.zeros((N_type, col_start))
            right_pad = jnp.zeros((N_type, state_dim - col_start - K_type))
            left_padding_dims.append(left_pad.shape)
            right_padding_dims.append(right_pad.shape)

            C_block = jnp.concatenate([left_pad, U, right_pad], axis=1)
            C_blocks.append(C_block)

            emission_dims.append((emiss_start, emiss_start + N_type))

            emiss_start += N_type
            col_start += K_type
        
        # concatenate vertically to form complete emission matrix
        C = jnp.concatenate(C_blocks, axis=0)  # Shape: (total_neurons, state_dim)
        
        #------------Compute R------------------
        mean_Y = jnp.mean(Y, axis=1, keepdims=True)
        var_Y = jnp.mean((Y - mean_Y)**2, axis=1)
        var_Y = jnp.maximum(var_Y, 1e-6)  # Floor for numerical stability
        
        R = jnp.diag(var_Y)  # Shape: (total_neurons, total_neurons)

        return ParamsCTDSEmissions(weights=C, cov=R, emission_dims=jnp.array(emission_dims), left_padding_dims=jnp.array(left_padding_dims), right_padding_dims=jnp.array(right_padding_dims))


    def initialize(self, observations) -> ParamsCTDS:
        """
        Estimate initial parameters from observed data Y and cell-type mask.
        """
        #first check that observation has shape (N, T)
        assert observations.shape[0] == self.emission_dim, "Observations should have shape (N, T) where N is number of neurons and T is number of time steps."
        Y=observations  # (N, T) where N is number of neurons, T is number of time steps
        
        # Create Dale mask with True for Excitatory neurons and False for Inhibitory neurons
        # Uses fancy indexing: for each neuron, it looks up the sign for that neuron's cell type.
        dale_mask=jnp.where(self.constraints.cell_sign[self.constraints.cell_type_mask] == 1, True, False)  
        
        # Estimate linear recurrent matrix J 
        J = estimate_J(Y, dale_mask)

        block_factors = blockwise_NMF(J, self.constraints)
        state_dim=int(jnp.sum(self.constraints.cell_type_dimensions)) # total state dimension across all cell types
        #TODO:MAKE JAX COMPATIBLE
        U_list=[tup[0] for tup in block_factors]  # list of (N, K) matrices for each cell type
        V_list=[tup[1] for tup in block_factors]  # list of (N, K) matrices for each cell type
        #initial param for CTDSParams
        initial=ParamsCTDSInitial(mean = jnp.zeros(state_dim), cov = 1e-2 * jnp.eye(state_dim))

        #Initalize emissions
        emissions = self.initialize_emissions(Y, U_list, state_dim)
        #Initalize dynamics
        dynamics = self.initialize_dynamics(V_list, emissions.weights)

        return ParamsCTDS(initial,dynamics, emissions, self.constraints, observations=observations)
    
    @partial(jax.jit, static_argnums=0)
    def e_step(self,
        params: ParamsCTDS,
        emissions: Array,
        inputs: Optional[Array] = None,
    ) -> SufficientStats:
        """
        Perform the E-step of the EM algorithm: compute sufficient statistics.

        Args:
            params: CTDS model parameters.
            emissions: Emission data (T, D).
            inputs: Optional control inputs (T, U).

        Returns:
            SufficientStats object containing:
                - latent_mean: Posterior means of latent states (T, D)
                - latent_second_moment: Posterior second moments (T, D, D)
                - cross_time_moment: Cross-covariances between consecutive latent states (T-1, D, D)
        """
        return DynamaxLGSSMBackend.e_step(params, emissions, inputs)
    def initialize_m_step_state(self, params, props: Optional[ParamsCTDS]=None):
        return M_Step_State(0, jnp.array([0.0]), jnp.array([0.0]), jnp.array([0.0]), jnp.array([0.0]))

    #jit does not work on cpu
    #@partial(jax.jit, static_argnums=0
    def m_step(self,
               params: ParamsCTDS,
               props: Optional[ParamsCTDS], # unused here just for compatibility
               batch_stats: SufficientStats, #batch of sufficient statistics
               m_step_state: Optional[Any] 
               ) -> Tuple[ParamsCTDS, Optional[Any]]:
        """
        M-step that supports batches of sufficient statistics from multiple sequences.
        Averages sufficient statistics across the batch, then calls single-sequence M-step.
        Args:
            params: Current CTDS parameters.
            props: Optional parameter properties (unused here).
            batch_stats: Sufficient statistics from multiple sequences.
            m_step_state: Optional state for M-step (unused here).
        Returns:
            Updated CTDS parameters and (unchanged) M-step state.   
        """
        #TODO: add conditional to check if batch_stats has a batch dimension
        
        #first we make sure every parameter in batch_stats has a batch dimension even if size 1
        assert batch_stats.latent_mean.ndim == 3, "latent_mean should have shape (batch_size, T, D). For single sequence, use shape (1, T, D)"
        assert batch_stats.latent_second_moment.ndim == 4, "latent_second_moment should have shape (batch_size, T, D, D). For single sequence, use shape (1, T, D, D)"
        assert batch_stats.cross_time_moment.ndim == 4, "cross_time_moment should have shape (batch_size, T-1, D, D). For single sequence, use shape (1, T-1, D, D)"

        # Average sufficient statistics across the batch
        # batch_stats is a SufficientStats object with arrays of shape (batch_size, T, ...)
        # where batch_size is the number of sequences e.g number of trials
        latent_mean = jnp.mean(batch_stats.latent_mean, axis=0)                     # (T, K)
        latent_second_moment = jnp.mean(batch_stats.latent_second_moment, axis=0)  # (T, K, K)
        cross_time_moment = jnp.mean(batch_stats.cross_time_moment, axis=0)        # (T−1, K, K)
        stats = SufficientStats(
            latent_mean=latent_mean,
            latent_second_moment=latent_second_moment,
            cross_time_moment=cross_time_moment,
            loglik=0.0,  # optional, unused in M-step
            T=latent_mean.shape[0]
        )
        T=stats.T
        #------------Update A---------------------
        
        # latent_mean: (T, D) ⇒ Ex: (D, T)
        Ex = stats.latent_mean.T
        X  = Ex[:, :-1]     # (D, T-1)   past
        Y = Ex[:,  1:]     # (D, T-1)   future
        X_T=stats.latent_mean[:-1]  # (T-1, D)
        
        #H1=jax.lax.dot_general(X, X_T, dimension_numbers=(((1,), (0,)),  ((), ()))) # (D, D) Gram matrix of past states
        #F1=jax.lax.dot_general(Y, X_T, dimension_numbers=(((1,), (0,)), ((), ())))  # (D, D) cross-covariance
        H1=X @ X_T + 1e-3 * jnp.eye(X.shape[0]) # (D, D) Gram matrix of past states
        F1=Y @ X_T  # (D, D) cross-covariance
        
        #check that H1 is well conditioned
        #assert check_qp_condition(H1), "H1 is not well conditioned. Consider regularizing the model or checking the data."

        masks = jnp.logical_not(jnp.eye(H1.shape[0], dtype=jnp.bool_))#(D,D) array where all entries are true except diagonals
        cell_type_mask = jnp.where(params.dynamics.dynamics_mask==-1, False, True) # shape (D,) 

        vmap_solver1 = jax.vmap(solve_constrained_QP, in_axes=(None, 1,1,0))

        A_t= vmap_solver1(H1, F1, masks, cell_type_mask)  # shape (D, D)
        A=jnp.transpose(A_t)
        delta_A =jnp.concatenate([m_step_state.delta_A, jnp.array([jnp.linalg.norm(A - params.dynamics.weights)] )])

        #---------Update Q------------------------
        # Q = (1/(T-1)) * sum_{t=2}^T [E[x_t x_t^T] - A E[x_{t-1} x_t^T]^T - E[x_t x_{t-1}^T] A^T + A E[x_{t-1} x_{t-1}^T] A^T]
        #   =(1/(T-1)) * sum_{t>=2}^T [S11 - A @ S10.T - S10 @ A.T + A @ S00 @ A.T]
        S11 = jnp.sum(stats.latent_second_moment[1:], axis=0)  #sums T-1 matrices returns (D, D) matrix
        S10 = jnp.sum(stats.cross_time_moment, axis=0)  #sums T-1 matrices returns  shape (D, D) 
        S00 = jnp.sum(stats.latent_second_moment[:-1], axis=0)  # shape (D, D) 

        # Optimizing Gemm calls by reusing intermidiaries
        AS00   = A @ S00
        AS00AT = AS00 @ A.T
        Q = (S11 - A @ S10.T - S10 @ A.T + AS00AT) / (T - 1.0)      # (D, D) 
        delta_Q = jnp.concatenate([m_step_state.delta_Q, jnp.array([jnp.linalg.norm(Q - params.dynamics.cov)])])

        dynamics = ParamsCTDSDynamics(weights=A, cov=Q, dynamics_mask=params.dynamics.dynamics_mask)
        
        #---------Update C---------------------------
        Y_obs=params.observations.T # shape (T,N)
        X=stats.latent_mean #shape(T, D)
        C=blockwise_NNLS(Y_obs, X, params.emissions.left_padding_dims, params.emissions.right_padding_dims, params.emissions.emission_dims, params.constraints.cell_type_dimensions)
        delta_C = jnp.concatenate([m_step_state.delta_C, jnp.array([jnp.linalg.norm(C - params.emissions.weights)])])

        #xs=jnp.arange(params.constraints.cell_sign.shape[0]-1, -1, -1) 
        #init=(Y_obs, X, params.emissions.left_padding_dims, params.emissions.right_padding_dims, params.emissions.emission_dims, params.constraints.cell_type_dimensions)   
        #carry, C=jax.lax.scan(blockwise_NNLS, init,xs)    
        #vmap_solver2=jax.vmap(blockwise_NNLS, (None, None, 0, 0,0,0))
        #C=vmap_solver2(Y_obs, X, params.emissions.left_padding_dims, params.emissions.right_padding_dims, params.emissions.emission_dims, params.constraints.cell_type_dimensions) #shape (N, D)


        #---------Update R----------------------------  
        """
        # R = (1/T) * sum_{t=1}^T [ y_t y_t^T - C E[x_t] y_t^T - y_t E[x_t]^T C^T + C E[x_t x_t^T] C^T ]
        # Efficient reuse + batched quadratic form over time:
        CEx  = C @ Ex                                                 # (N, T)
        CExY = CEx @ Y_obs.T                                          # (N, N)
        YY   = Y_obs @ Y_obs.T                                        # (N, N)
        # quad = sum_t C E[x_t x_t^T] C^T
        quad = jnp.einsum('nd,tdk,nk->nn', C, stats.latent_second_moment, C)  # (N, N)

        R = (YY - CExY - CExY.T + quad) / T                           # (N, N)
        emissions = ParamsCTDSEmissions(weights=C, cov=R)
        # R= (1 / T) * jnp.sum(Y_obs @ Y_obs.T - C @ Ex @ Y_obs.T - Y_obs @ Ex.T @ C.T + C @ stats.latent_second_moment @ C.T, axis=0) #shape (N, N)

        """
        # Alternatively, estimate R as diagonal matrix with average residual variance per neuron
        Y_pred = C @ Ex  # shape (N, T)
        resid = params.observations - Y_pred  # shape (N, T)
        R_diag = jnp.mean(resid ** 2, axis=1)  # shape (N,)
        R = jnp.diag(R_diag) # shape (N, N)
        delta_R = jnp.concatenate([m_step_state.delta_R, jnp.array([jnp.linalg.norm(R - params.emissions.cov)])])

        emissions = ParamsCTDSEmissions(weights=C, cov=R, emission_dims=params.emissions.emission_dims, left_padding_dims=params.emissions.left_padding_dims, right_padding_dims=params.emissions.right_padding_dims)

        #---------Update initial state-----------------
        # Initial state mean is the first latent mean
        initial_mean = stats.latent_mean[0]  # shape (D,)
        # Initial state covariance is the first latent covariance
        initial_cov = stats.latent_second_moment[0]  # shape (D,)

        initial = ParamsCTDSInitial(mean=initial_mean, cov=initial_cov)
        iter=m_step_state.iteration + 1
        return ParamsCTDS(initial, dynamics, emissions, constraints=params.constraints, observations=params.observations),  M_Step_State(iter, delta_A, delta_C, delta_Q, delta_R)

    
    def fit_em(self, 
               params: ParamsCTDS,
               batch_emissions: Union[Real[Array, "num_timesteps emission_dim"],
                                Real[Array, "num_batches num_timesteps emission_dim"]],
               batch_inputs: Optional[jnp.ndarray] = None,
               num_iters: int = 100,
               verbose: bool = True) -> Tuple[ParamsCTDS, jnp.ndarray]:

        #@jax.jit
        def em_step(params, m_step_state):
            """Perform one EM step."""
            vmap_solver=jax.vmap(DynamaxLGSSMBackend.e_step, in_axes=(None, 0, 0))  # Vectorize over batch dimension
            batch_stats, lls = vmap_solver(params,batch_emissions, batch_inputs)
            lp = self.log_prior(params) + lls.sum()
            params, m_step_state = self.m_step(params, None, batch_stats, m_step_state)

            return params, m_step_state, lp

        log_probs = []
        m_step_state = self.initialize_m_step_state(params, None)
        params, m_step_state, marginal_logprob = em_step(params, m_step_state)
        log_probs.append(marginal_logprob)

        pbar = progress_bar(range(1, num_iters)) if verbose else range(num_iters)
        for i in pbar:
            params, m_step_state, marginal_logprob = em_step(params, m_step_state)
            print(f"Iteration {i}: log-likelihood = {marginal_logprob}")
            convergence_criteria = jnp.abs(marginal_logprob - log_probs[-1]) / jnp.abs(log_probs[-1])
            log_probs.append(marginal_logprob)
            if convergence_criteria < 1e-4:
                print(f"Converged at iteration {i}")
                break

        return params, jnp.array(log_probs[1:])

    def sample(self, params: ParamsCTDS, key: jax.random.PRNGKey, num_timesteps: int, inputs: Optional[jnp.ndarray] = None) -> Tuple[Float[Array, "num_timesteps state_dim"],Float[Array, "num_timesteps emission_dim"]]:
        """
        Generates samples from the model using the provided parameters and random key.
        Args:
            params (ParamsCTDS): Model parameters for sampling.
            key (jax.random.PRNGKey): Random key for stochastic operations.
            num_timesteps (int): Number of timesteps to sample.
            inputs (Optional[jnp.ndarray], optional): Optional input data for conditional sampling. Defaults to None.
        Returns:
            Tuple[latent_states, emissions] where:
                - latent_states: (T, D) The sampled latent states.
                - emissions: (T, N) The sampled emissions.
        """
        return super().sample(params, key, num_timesteps, inputs)
    
    def log_prob(self, params, states, emissions, inputs = None):
        return super().log_prob(params, states, emissions, inputs)

    def filter(self, params: ParamsCTDS, emissions, inputs=None):
        return DynamaxLGSSMBackend.filter(params, emissions, inputs)

    @staticmethod
    def smoother(params: ParamsCTDS, emissions, inputs=None):
        return DynamaxLGSSMBackend.smoother(params, emissions, inputs)

    def marginal_log_prob(self, params, emissions, inputs=None):
        return super().marginal_log_prob(params, emissions, inputs)
    
    
    #def sample
    #def forecast
    #def plot latent trajectory
    #def fit EM
    


#FUTURE TO DO: SLDS CTDS 
class SwitchingCTDS(BaseCTDS):
    """
    Future implementation for HMM-style discrete switching between CTDS modes.
    """
    def __init__(self,
                 params: Optional[ParamsCTDS] = None,
                 inference_engine: Optional[Any] = None):
        super().__init__(params=params, inference_engine=inference_engine)



#FUTURE TO DO: NonlinearCTDS 
class NonlinearCTDS(BaseCTDS):
    """
    Future implementation with nonlinear dynamics or emissions.
    """
    def __init__(self,
                 params: Optional[ParamsCTDS] = None,
                 inference_engine: Optional[Any] = None):
        super().__init__(params=params, inference_engine=inference_engine)












