
from typing import Optional, Any, Callable
from jaxtyping import Array
import jax as jnp
from params import ParamsCTDS, SufficientStats, ParamsCTDSDynamics, ParamsCTDSEmissions, ParamsCTDSInitial
from abc import ABC, abstractmethod
from inference import InferenceBackend, DynamaxLGSSMBackend
from utlis import estimate_J,blockwise_nmf, build_v_dale

#maybe later implement as subclass of dynamax SSM. right now dont see a reason to do so
class BaseCTDS(ABC):
    def __init__(self,
                 backend: InferenceBackend,
                 dynamics_fn: Callable,
                 emissions_fn: Callable,
                 m_step: Optional[Callable] = None
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
                 cell_types: Array,
                 latent_dims: int
                 ):
        backend=DynamaxLGSSMBackend
        dynamics_fn = self._build_A
        emissions_fn = self._build_C
        super().__init__(backend, dynamics_fn, emissions_fn)
        

    def _build_A(self, V_dale, U)-> ParamsCTDSDynamics:

        A=V_dale.T @ U
        D=V_dale.shape[1]
        Q=1e-2 * jnp.eye(D)
        
        return ParamsCTDSDynamics(A, Q)

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
        U_E, V_E, U_I, V_I=blockwise_nmf(jnp.abs(J), mask, D_E, D_I)
        state_dim=D_E+D_I
        V_dale=build_v_dale(V_E, V_I)
        initial=ParamsCTDSInitial(mean = jnp.zeros(state_dim), cov = 1e-2 * jnp.eye(state_dim))
        emissions=self.emissions_fn(U_E, U_I, Y )
        dynamics=self.dynamics_fn(V_dale, emissions.C)
        return ParamsCTDS(initial,dynamics, emissions)
    
    def m_step(self, params: ParamsCTDS, stats) -> ParamsCTDS:
        """
        M-step: update model parameters from sufficient statistics.
        """
    
        return ParamsCTDS()
    

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

