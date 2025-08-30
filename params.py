"""
Core parameter structures for Cell-Type Dynamical Systems (CTDS).

This module defines the parameter classes that encapsulate all model components
including initial distributions, dynamics, emissions, and biological constraints.
"""
import jax.numpy as jnp
from typing import NamedTuple, Union, Optional
from jaxtyping import Float, Array
import chex
from dynamax.parameters import ParameterProperties
from dynamax.linear_gaussian_ssm import ParamsLGSSM, ParamsLGSSMInitial, ParamsLGSSMDynamics, ParamsLGSSMEmissions


class ParamsCTDSInitial(NamedTuple):
    """
    Initial state distribution parameters for CTDS: p(x_1) = N(x_1 | mean, cov).
    
    Parameters
    ----------
    mean : Array, shape (D,)
        Initial state mean vector.
    cov : Array, shape (D, D) or (D,)
        Initial state covariance matrix (full) or diagonal variances.
    
    Notes
    -----
    For the CTDS model, the initial state follows a multivariate Gaussian:
    $$x_1 \\sim \\mathcal{N}(\\mu_1, \\Sigma_1)$$
    where D is the total latent state dimension across all cell types.
    """
    mean: Union[Float[Array, "D"], ParameterProperties]
    cov: Union[
        Float[Array, "D D"],
        Float[Array, "D"],  # for diagonal
        ParameterProperties
    ]


class ParamsCTDSDynamics(NamedTuple):
    """
    Dynamics parameters for the CTDS latent state evolution.

    Parameters
    ----------
    weights : Array, shape (D, D)
        State transition matrix A in x_{t+1} = A x_t + B u_t + ε_t.
    cov : Array, shape (D, D) or (D,)
        Process noise covariance Q (full matrix or diagonal variances).
    input_weights : Array, shape (D, U) or (T, D, U), optional
        Input weight matrix B (static or time-varying).
    dynamics_mask : Array, shape (D,), optional
        Cell-type assignment mask for each latent dimension:
        1 for excitatory, -1 for inhibitory, 0 for unassigned.

    Notes
    -----
    The dynamics follow the linear state-space evolution:
    $$x_{t+1} = A x_t + B u_t + \\varepsilon_t, \\quad 
    \\varepsilon_t \\sim \\mathcal{N}(0, Q)$$
    where Dale's law constraints apply to enforce biologically plausible
    sign patterns in the transition matrix A.
    """
    weights: Union[Float[Array, "D D"], ParameterProperties]
    cov: Union[
        Float[Array, "D D"],
        Float[Array, "D"],
        ParameterProperties
    ]
    input_weights: Optional[Union[ParameterProperties,
                    Float[Array, "D U"],
                    Float[Array, "T D U"]]] = None
    dynamics_mask: Optional[chex.Array] = None



class ParamsCTDSEmissions(NamedTuple):
    """
    Emission model parameters mapping latent states to observations.

    Parameters
    ----------
    weights : Array, shape (N, D)
        Emission weight matrix C in y_t = C x_t + D u_t + η_t.
        Each row corresponds to one observed neuron.
    cov : Array, shape (N, N) or (N,)
        Observation noise covariance R (full matrix or diagonal variances).
    emission_dims : Array, shape (K, 2), optional
        Start and end column indices for each cell type's emission block.
    left_padding_dims : Array, shape (K, 2), optional  
        Dimensions of left zero-padding for each cell type block.
    right_padding_dims : Array, shape (K, 2), optional
        Dimensions of right zero-padding for each cell type block.

    Notes
    -----
    The emission model follows:
    $$y_t = C x_t + D u_t + \\eta_t, \\quad \\eta_t \\sim \\mathcal{N}(0, R)$$
    Dale's law sign constraints apply row-wise: excitatory neurons have
    non-negative weights, inhibitory neurons have non-positive weights.
    The block structure allows modeling cell-type-specific connectivity patterns.
    """
    weights: Union[Float[Array, "N D"], ParameterProperties] 
    cov: Union[
        Float[Array, "N N"],
        Float[Array, "N"],
        ParameterProperties
    ]
    emission_dims: Optional[chex.Array] = None
    left_padding_dims: Optional[chex.Array] = None
    right_padding_dims: Optional[chex.Array] = None


class ParamsCTDSConstraints(NamedTuple):
    """
    Biological constraints for enforcing Dale's law and cell-type structure.

    Parameters
    ----------
    cell_types : Array, shape (K,)
        Cell type labels as contiguous integers [0, 1, ..., K-1].
    cell_sign : Array, shape (K,)  
        Sign for each cell type: +1 (excitatory), -1 (inhibitory).
    cell_type_dimensions : Array, shape (K,)
        Number of latent dimensions allocated to each cell type.
    cell_type_mask : Array, shape (N,)
        Cell type assignment for each observed neuron.

    Notes
    -----
    These constraints enforce Dale's law: excitatory neurons have non-negative
    synaptic weights, inhibitory neurons have non-positive weights. The
    cell_type_mask maps each observed neuron to its cell type, enabling
    block-structured computations and biologically informed regularization.
    
    The constraint cell_types[i] corresponds to cell_sign[i] and 
    cell_type_dimensions[i]. For neuron j, cell_type_mask[j] gives its type index.
    """
    cell_types: chex.Array
    cell_sign: chex.Array
    cell_type_dimensions: chex.Array
    cell_type_mask: chex.Array
    

class ParamsCTDS(NamedTuple):
    """
    Complete CTDS model parameters with biological constraints.

    This structure encapsulates all parameters for a Cell-Type Dynamical System,
    including initial conditions, dynamics, emissions, and cell-type constraints.

    Parameters
    ----------
    initial : ParamsCTDSInitial
        Initial state distribution parameters.
    dynamics : ParamsCTDSDynamics  
        State transition and process noise parameters.
    emissions : ParamsCTDSEmissions
        Observation model parameters.
    constraints : ParamsCTDSConstraints
        Dale's law and cell-type structure constraints.
    observations : Array, shape (N, T), optional
        Observed emission data (for reference during fitting).

    Notes
    -----
    The full CTDS model equations are:
    
    **State evolution:**
    $$x_{t+1} = A x_t + B u_t + \\varepsilon_t, \\quad \\varepsilon_t \\sim \\mathcal{N}(0, Q)$$
    
    **Emissions:**  
    $$y_t = C x_t + D u_t + \\eta_t, \\quad \\eta_t \\sim \\mathcal{N}(0, R)$$
    
    where A, C respect Dale's law sign constraints based on cell types.
    
    See Also
    --------
    to_lgssm : Convert to Dynamax-compatible format for inference.
    """
    initial: ParamsCTDSInitial
    dynamics: ParamsCTDSDynamics
    emissions: ParamsCTDSEmissions
    constraints: ParamsCTDSConstraints
    observations: Optional[chex.Array] = None

    def to_lgssm(self) -> ParamsLGSSM:
        """
        Convert CTDS parameters to Dynamax LGSSM format for inference.
        
        Returns
        -------
        ParamsLGSSM
            Dynamax-compatible parameter structure.
            
        Notes
        -----
        This conversion enables use of Dynamax's optimized inference routines
        while preserving CTDS-specific structure. Biases are set to zero since
        CTDS uses centered parameterization.
        """
        # Convert initial distribution
        lgssm_initial = ParamsLGSSMInitial(
            mean=self.initial.mean,
            cov=self.initial.cov
        )
        # Convert dynamics  
        lgssm_dynamics = ParamsLGSSMDynamics(
            weights=self.dynamics.weights,
            bias=jnp.zeros(self.dynamics.weights.shape[0]),
            input_weights=self.dynamics.input_weights,
            cov=self.dynamics.cov
        )
        # Convert emissions
        lgssm_emissions = ParamsLGSSMEmissions(
            weights=self.emissions.weights,
            bias=jnp.zeros(self.emissions.weights.shape[0]),
            input_weights=None,
            cov=self.emissions.cov
        )
        return ParamsLGSSM(
            initial=lgssm_initial,
            dynamics=lgssm_dynamics,
            emissions=lgssm_emissions
        )


class M_Step_State(NamedTuple):
    """
    State tracking for M-step convergence diagnostics.
    
    Parameters
    ----------
    iteration : int
        Current iteration number.
    delta_A : Array, shape (num_iters,)
        Frobenius norm changes in dynamics matrix A across iterations.
    delta_C : Array, shape (num_iters,)  
        Frobenius norm changes in emission matrix C across iterations.
    delta_Q : Array, shape (num_iters,)
        Frobenius norm changes in process noise Q across iterations.
    delta_R : Array, shape (num_iters,)
        Frobenius norm changes in observation noise R across iterations.
    """
    iteration: int
    delta_A: chex.Array
    delta_C: chex.Array
    delta_Q: chex.Array
    delta_R: chex.Array


class PosteriorCTDSSmoothed(NamedTuple):
    """
    RTS-smoothed posterior over latent states.
    
    Parameters
    ----------
    marginal_loglik : float
        Marginal log-likelihood of the observed data.
    smoothed_means : Array, shape (T, D)
        Posterior mean estimates for all time steps.
    smoothed_covariances : Array, shape (T, D, D)  
        Posterior covariance estimates for all time steps.
    smoothed_cross_covariances : Array, shape (T-1, D, D), optional
        Cross-time posterior covariances E[x_{t+1} x_t^T | y_{1:T}].
    """
    marginal_loglik: float
    smoothed_means: Float[Array, "T D"]
    smoothed_covariances: Float[Array, "T D D"]
    smoothed_cross_covariances: Optional[Float[Array, "T-1 D D"]] = None


class PosteriorCTDSFiltered(NamedTuple):
    """
    Kalman filtered posterior over latent states.
    
    Parameters
    ----------
    marginal_loglik : float
        Marginal log-likelihood of the observed data.
    filtered_means : Array, shape (T, D)
        Filtered mean estimates for all time steps.
    filtered_covariances : Array, shape (T, D, D)
        Filtered covariance estimates for all time steps.
    predicted_means : Array, shape (T, D)
        One-step-ahead predicted means.
    predicted_covariances : Array, shape (T, D, D)
        One-step-ahead predicted covariances.
    """
    marginal_loglik: float
    filtered_means: Float[Array, "T D"]
    filtered_covariances: Float[Array, "T D D"]
    predicted_means: Float[Array, "T D"]
    predicted_covariances: Float[Array, "T D D"]


class SufficientStats(NamedTuple):
    """
    Sufficient statistics for CTDS parameter updates in the M-step.
    
    Parameters
    ----------
    latent_mean : Array, shape (T, D)
        Posterior mean of latent states E[x_t].
    latent_second_moment : Array, shape (T, D, D)
        Posterior second moment E[x_t x_t^T].
    cross_time_moment : Array, shape (T-1, D, D)
        Cross-time moments E[x_{t+1} x_t^T] for dynamics updates.
    loglik : float
        Marginal log-likelihood of observed data.
    T : int
        Number of time steps.
        
    Notes
    -----
    These statistics are computed in the E-step and used in the M-step
    to update model parameters via closed-form solutions or constrained
    optimization routines.
    """
    latent_mean: Float[Array, "T D"]
    latent_second_moment: Float[Array, "T D D"]
    cross_time_moment: Float[Array, "T-1 D D"]
    loglik: float              
    T: int                      
