from ast import Tuple
import jax
from typing import Optional, Any, Callable
from jaxtyping import Array
import jax.numpy as jnp
from params import ParamsCTDS, SufficientStats, ParamsCTDSDynamics, ParamsCTDSEmissions, ParamsCTDSInitial, ParamsCTDSConstraints
from abc import ABC, abstractmethod
from inference import InferenceBackend, DynamaxLGSSMBackend
from utlis import estimate_J,blockwise_NMF, build_v_dale, solve_constrained_QP, NNLS

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




class CTDS(BaseCTDS):
    """
    #TODO: add redion idenity
    """
    def __init__(self, 
                 observations: Array,
                 cell_types:  Array, #(k,) where k is number of cell types. contains cell type labels
                 cell_sign: Array, #(k,) where k is number of cell types. values are 1 for excitatory, -1 for inhibitory
                 cell_type: Array, #(k,) where k is number of cell types. contains latent dimensions for each cell type
                 cell_type_mask: Array, #(N,) where N is number of neurons. contains cell type label for each neuron
                 region_identity: Optional[Array] = None,
                 ):
        self.observations = observations
        self.inference_backend = DynamaxLGSSMBackend
        self.dynamics_fn = self._initialize_dynamics
        self.emissions_fn = self._initialize_emissions
        self.constraints = ParamsCTDSConstraints(
            cell_types=cell_types,
            cell_sign=cell_sign,
            cell_type_dimensions=cell_type,
            cell_type_mask=cell_type_mask
        )
        #super().__init__(inference_backend, dynamics_fn, emissions_fn)
        

    def _initialize_dynamics(self, V_list, U) -> ParamsCTDSDynamics:
        #build V_dale 
        #V_list should have have the same length and order as cell_types and cell_sign
        assert len(V_list) == len(self.constraints.cell_types), "Number of V matrices must match number of cell types."
        V_dale_list = []
        for i in range(len(V_list)):
            if self.constraints.cell_sign[i] == -1:
                V_dale_list.append(-V_list[i])  # Invert sign for inhibitory cell types
            else:
                V_dale_list.append(V_list[i])
 
        V_dale = jnp.concatenate(V_dale_list, axis=1)

        A=V_dale.T @ U
        D=V_dale.shape[1]
        Q=1e-2 * jnp.eye(D)

        return ParamsCTDSDynamics(weights=A,cov=Q)

    def _initialize_emissions(self, Y, U_list, state_dim) -> ParamsCTDSEmissions:
        #the number of columns K of each U in U_list should add up to state_dim D
        assert sum(U.shape[1] for U in U_list) == state_dim, "Latent dimensions do not match state dimension."
        
        #------------Compute C---------------------
        # create block-diagonal emission matrix
        C_blocks = []
        col_start = 0
        
        for U in U_list:
            N_type, K_type = U.shape
            
            # create padded block: [zeros_left, U, zeros_right]
            left_pad = jnp.zeros((N_type, col_start))
            right_pad = jnp.zeros((N_type, state_dim - col_start - K_type))
            
            C_block = jnp.concatenate([left_pad, U, right_pad], axis=1)
            C_blocks.append(C_block)
            
            col_start += K_type
        
        # concatenate vertically to form complete emission matrix
        C = jnp.concatenate(C_blocks, axis=0)  # Shape: (total_neurons, state_dim)
        
        #------------Compute R------------------
        mean_Y = jnp.mean(Y, axis=1, keepdims=True)
        var_Y = jnp.mean((Y - mean_Y)**2, axis=1)
        var_Y = jnp.maximum(var_Y, 1e-6)  # Floor for numerical stability
        
        R = jnp.diag(var_Y)  # Shape: (total_neurons, total_neurons)
        
        return ParamsCTDSEmissions(weights=C, cov=R)


    def initialize(self) -> ParamsCTDS:
        """
        Estimate initial parameters from observed data Y and cell-type mask.
        """
        Y=self.observations  # (N, T) where N is number of neurons, T is number of time steps
        mask=jnp.where(self.constraints.dale_mask==1, True, False)  # (N,) where N is number of neurons
        J = estimate_J(Y, mask) #TODO: UPDATE NMF and estimate J to take dale_mask not true or false mask
        block_factors = blockwise_NMF(J, self.constraints)
        state_dim=int(jnp.sum(self.constraints.cell_type_dimensions)) # total state dimension across all cell types
        U_list=[tup[0] for tup in block_factors]  # list of (N, K) matrices for each cell type
        V_list=[tup[1] for tup in block_factors]  # list of (N, K) matrices for each cell type
        #initial param for CTDSParams
        initial=ParamsCTDSInitial(mean = jnp.zeros(state_dim), cov = 1e-2 * jnp.eye(state_dim))

        #Initalize emissions
        emissions = self._initialize_emissions(Y, U_list, state_dim)
        #Initalize dynamics
        dynamics = self._initialize_dynamics(V_list, emissions.weights)

        return ParamsCTDS(initial,dynamics, emissions, self.constraints)

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

