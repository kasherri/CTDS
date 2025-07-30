import jax
from typing import Optional, Any, Callable
from jaxtyping import Array
import jax.numpy as jnp
from params import ParamsCTDS, SufficientStats, ParamsCTDSDynamics, ParamsCTDSEmissions, ParamsCTDSInitial
from abc import ABC, abstractmethod
from inference import InferenceBackend, DynamaxLGSSMBackend
from utlis import estimate_J,blockwise_nmf, build_v_dale, solve_constrained_QP, NNLS

#maybe later implement as subclass of dynamax SSM. right now dont see a reason to do so
class BaseCTDS(ABC):
    def __init__(self,
                 backend: InferenceBackend,
                 dynamics_fn: Callable,
                 emissions_fn: Callable,
                 cell_types: Optional[Array] = None,
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
    

    @abstractmethod
    def initialize_params(self, Y: Array, mask: Array, **kwargs) -> ParamsCTDS:
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




class CTDS(BaseCTDS):
    """
  
    """
    def __init__(self, 
                 observations: Array,
                 cell_types: Array, # shape (K,2) where K is number of cell types e.g [[0,True],[1,False],...]
                 mask: Array, # shape (N,) where N is number of neurons True for excitatory, False for inhibitory
                 latent_dims: int
                 ):
        self.observations = observations
        backend=DynamaxLGSSMBackend
        dynamics_fn = self._build_A
        emissions_fn = self._build_C
        super().__init__(backend, dynamics_fn, emissions_fn)
        

    def _build_A(self, V_dale, U, cell_type_dimensions) -> ParamsCTDSDynamics:

        A=V_dale.T @ U
        D=V_dale.shape[1]
        Q=1e-2 * jnp.eye(D)

        return ParamsCTDSDynamics(A, Q, cell_type_dimensions)

    def _build_C(self, U_E, U_I, Y ) -> ParamsCTDSEmissions:
        """Return emission matrix C and observation noise covariance R."""
        NE, K1 = U_E.shape
        NI, K2 = U_I.shape

        top    = jnp.concatenate([U_E, jnp.zeros((NE, K2))], axis=1)
        bottom = jnp.concatenate([jnp.zeros((NI, K1)), U_I], axis=1)
        # Mean across time (axis 1 = timepoints)
        mean_Y = jnp.mean(Y, axis=1, keepdims=True)      # shape (N, 1)
        # Per-neuron variance
        var_Y = jnp.mean((Y - mean_Y)**2, axis=1)        # shape (N,)
        # floor variance to prevent instability
        var_Y = jnp.maximum(var_Y, 1e-6)
        diag_R = 1 * var_Y                           # shape (N,)
        
        R=jnp.diag(diag_R)                          # shape (N, N)
        C = jnp.concatenate([top, bottom], axis=0)
        return ParamsCTDSEmissions(C, R)

    def initialize_params(self, Y: Array, mask: Array, D_E:int, D_I:int, **kwargs) -> ParamsCTDS:
        """
        Estimate initial parameters from observed data Y and cell-type mask.
        """
        J = estimate_J(Y, mask)
        U_E, V_E, U_I, V_I=blockwise_nmf(J, mask, D_E, D_I)
        state_dim=D_E+D_I
        V_dale=build_v_dale(V_E, V_I)
        initial=ParamsCTDSInitial(mean = jnp.zeros(state_dim), cov = 1e-2 * jnp.eye(state_dim))
        emissions=self.emissions_fn(U_E, U_I, Y )
        dynamics=self.dynamics_fn(V_dale, emissions.C, jnp.array([D_E, D_I]))
        return ParamsCTDS(initial,dynamics, emissions, 
                          cell_types_mask=mask)
    
    #@jax.jit
    def m_step(self, params: ParamsCTDS, stats: SufficientStats) -> ParamsCTDS:
        """
        Perform the M-step of the EM algorithm: update model parameters from sufficient statistics.

        Args:
            params: ParamsCTDS
                Current model parameters, including dynamics, emissions, and cell type information.
            stats: SufficientStats
                Sufficient statistics computed from the E-step. 
                - stats.latent_mean: Array of shape (T, D), posterior means of latent states.
                - stats.latent_second_moment: Array of shape (T, D, D), posterior second moments.
                - stats.cross_time_moment: Array of shape (T-1, D, D), cross-covariances between consecutive latent states.

        Returns:
            Updated ParamsCTDS object with new parameters:
                - Dynamics matrix A and noise covariance Q are updated using constrained least squares.
                - Emission matrix C is updated using non-negative least squares.
                - Observation noise covariance R is updated as the mean squared residual.
                - Initial state mean and covariance are set to the first posterior mean and covariance.

        The update logic is as follows:
            - A is estimated by solving a constrained quadratic program for each latent dimension.
            - Q is computed from the residuals of the latent dynamics.
            - C is estimated by solving a non-negative least squares problem for each neuron.
            - R is set to the diagonal covariance of the residuals between observed and predicted data.
            - The initial state is set from the first timepoint's posterior statistics.
        """
        #------------Update A---------------------
        # We construct matrices:
        #   X = [x_1, ..., x_{T-1}] ∈ ℝ^{D × (T-1)}  (past)
        X= stats.latent_mean[:-1].T  # shape (D, T-1)
        #   Y = [x_2, ..., x_T]     ∈ ℝ^{D × (T-1)}  (future)
        Y= stats.latent_mean[1:].T  # shape (D, T-1)
        #   Z = X.T ∈ ℝ^{(T-1) × D}  # regression features
        Z = X.T  # shape (T-1, D)
        
        # Objective Function: min_a_j  (1/2) * a_jᵀ H a_j - fᵀ a_j
        # where H = Zᵀ Z ∈ ℝ^{D × D}, f = Zᵀ y^{(j)} ∈ ℝ^D
        H1= Z.T @ Z  # shape (D, D)
        F1= Y @ Z  # shape (D, D)

        #(D,) array of jnp.bool_ indicating cell type (excitatory/inhibitory)
        #cell_type_mask = params.cell_types_mask  # shape (D,)
        #(D,D) array where all entries are true except diagonals
        masks = jnp.logical_not(jnp.eye(H1.shape[0], dtype=jnp.bool_))
        D_E, D_I = params.dynamics.cell_type_dimensions
        dynamics_dale_mask = jnp.concatenate([jnp.full((D_E,), True, dtype=jnp.bool_), jnp.full((D_I,), False, dtype=jnp.bool_)]) #TODO: add as parameter to Dynamics
        #vmap over each column of F and masks and each value in cell_type_mask
        vmap_solver1= jax.vmap(solve_constrained_QP, in_axes=(None, 1,1,0))

        A= vmap_solver1(H1, F1, masks, dynamics_dale_mask)  # shape (D, D)

        #---------Update Q------------------------
        S11 = jnp.sum(stats.latent_second_moment[1:], axis=0)  #sums T-1 matrices returns (D, D) matrix
        S10 = jnp.sum(stats.cross_time_moment, axis=0)  #sums T-1 matrices returns  shape (D, D) 
        S00 = jnp.sum(stats.latent_second_moment[:-1], axis=0)  # shape (D, D)

        # Compute Q using the formula
        # Q = 1/(T-1) * (S11 - A @ S10.T - S10 @ A.T + A @ S00 @ A.T
        Q=1/(stats.latent_second_moment.shape[0]-1)*(S11 - A @ S10.T - S10 @ A.T + A @ S00 @ A.T)

        dynamics = ParamsCTDSDynamics(A=A, Q=Q, cell_type_dimensions=params.dynamics.cell_type_dimensions)
        #---------Update C---------------------------
        # Goal:Estimate C ∈ ℝ^{N × D} such that:
        #     Y ≈ C @ X
        #   subject to:
        #     C ≥ 0  (elementwise non-negativity constraint)

        Y_obs=self.observations # shape (N, T)
        Ex= stats.latent_mean.T #shape (D, T)
        H2=Ex @ Ex.T  # shape (D, D)
        F2=Ex @ Y_obs.T  # shape (D, N)

        #Vmap over N columns of F2. Each column corresponds to f= X @ y_i ∈ ℝ^D
        vmap_solver2 = jax.vmap(NNLS, in_axes=(None, 1)) 
        C= vmap_solver2(H2, F2) #shape (D, N)
        #print(C)
        print("C T shape:", C.T.shape)  # shape (N, D)
        print("C  shape:", C.shape) 
        
        #---------Update R----------------------------
        Y_pred = C @ Ex  # shape (N, T)
        resid = Y_obs - Y_pred  # shape (N, T)
        R_diag = jnp.mean(resid ** 2, axis=1)  # shape (N,)
        R = jnp.diag(R_diag)

        emissions = ParamsCTDSEmissions(C=C, R=R)
        
        #---------Update initial state-----------------
        # Initial state mean is the first latent mean
        initial_mean = stats.latent_mean[0]  # shape (D,)
        # Initial state covariance is the first latent covariance
        initial_cov = stats.latent_second_moment[0]  # shape (D,)

        initial = ParamsCTDSInitial(mean=initial_mean, cov=initial_cov)

        return ParamsCTDS(initial, dynamics, emissions, 
                          cell_types_mask=params.cell_types_mask)

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

