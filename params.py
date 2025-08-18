import jax.numpy as jnp
from typing import NamedTuple, Union, Optional, List
from jaxtyping import Float, Array
from dynamax.parameters import ParameterProperties
from dynamax.linear_gaussian_ssm import ParamsLGSSM, ParamsLGSSMInitial, ParamsLGSSMDynamics, ParamsLGSSMEmissions


class ParamsCTDSInitial(NamedTuple):
    """Initial distribution p(z_1) = N(z_1 | mu, Sigma)"""
    #D is state dim
    mean: Union[Float[Array, "D"], ParameterProperties]
    cov: Union[
        Float[Array, "D D"],
        Float[Array, "D"],  # for diagonal
        ParameterProperties
    ]


class ParamsCTDSDynamics(NamedTuple):
    """
    Parameters for CTDS (Continuous-Time Dynamical System) dynamics.

    Attributes:
        A (Union[Float[Array, "state_dim state_dim"], ParameterProperties]):
            State transition matrix or its parameter properties. Represents the dynamics of the system.
        Q (Union[Float[Array, "state_dim state_dim"], Float[Array, "state_dim"], ParameterProperties]):
            Process noise covariance matrix, vector, or its parameter properties. Models the uncertainty in the system dynamics.
        input_weights (Optional[Union[ParameterProperties, Float[Array, "state_dim input_dim"], Float[Array, "ntime state_dim input_dim"]]]):
            Optional input weights for the system. Can be a parameter property, a matrix for static input weights, or a 3D array for time-varying input weights.
        cell_type_dimensions (Optional[Array]):
            Optional array specifying the dimensions for each cell type. Shape is (K,), where K is the number of cell types. Provides flexibility for modeling heterogeneous populations.
    """
    #N is number of neurons, D is state dim, T is number of time steps
    weights: Union[Float[Array, "D D"], ParameterProperties]
    cov: Union[
        Float[Array, "D D"],
        Float[Array, "D"],
        ParameterProperties
    ]
    input_weights: Optional[Union[ParameterProperties,
                    Float[Array, "D input_dim"],
                    Float[Array, "T D input_dim"]]] = None
    dynamics_mask: Optional[Array] = None  # (D, 1) where D is number of latent dimensions, 1 for excitatory, -1 for inhibitory, 0 for unassigned



class ParamsCTDSEmissions(NamedTuple):
    """
    Emission model: y_t = C z_t + v_t,  v_t ~ N(0, R)
    Each row of C corresponds to an observed neuron.
    Sign constraints (Dale's law) apply row-wise by cell type.
    """
    #N is number of neurons, D is state dim
    weights: Union[Float[Array, "N D"], ParameterProperties] 
    cov: Union[
        Float[Array, "N N"],
        Float[Array, "N"],
        ParameterProperties
    ]




class ParamsCTDSConstraints(NamedTuple):
    """
    Constraints for CTDS model parameters to enforce biological plausibility.
    Attributes:
        cell_types (Array): (K, 1) array where K is number of cell types containing cell type labels.
                            Must be a contiguous range of integers from 0 to K-1 (i.e., [0, 1, ..., K-1]) 
                            so that cell_type_mask can be used as an index into cell_sign.
        cell_sign (Array): (K, 1) array where K is number of cell types; values are 1 for excitatory, -1 for inhibitory.
        cell_type_dimensions (Array): (K, 1) array where K is number of cell types containing cell type dimensions.    
        cell_type_mask (Array): (N, 1) #(N,) where N is number of neurons. contains cell type label for each neuron

    Note: cell_type_mask contains labels for each neuron not sign types. is different from dynamics.dynamics_mask. 
    """
    cell_types: Array #(k,1) array where k is number of cell types containing cell type labes
    cell_sign: Array #(k,1) array where k is number of cell types values are 1 for excitatory, -1 for inhibitory, 0 for
    cell_type_dimensions: Array #(k,1) array where k is number of cell types containing cell type dimensions
    cell_type_mask: Array  #(N, 1) where N is number of neurons. contains cell type label for each neuron
    

class ParamsCTDS(NamedTuple):
    """
    Full CTDS model parameters.
    This structure mirrors Dynamax ParamsLGSSM, 
    but includes explicit sign constraints and cell-type structure.
    """
    initial: ParamsCTDSInitial
    dynamics: ParamsCTDSDynamics
    emissions: ParamsCTDSEmissions
    constraints: ParamsCTDSConstraints
    observations: Optional[Array]  # Optional observed data for the model, e.g., spike counts or firing rates


    #TODO: include region identity for each neuron region_identity: Float[Array, "emission_dim"]  # region index for each neuron
    def to_lgssm(self) -> ParamsLGSSM:
        """
        Convert CTDS parameters to LGSSM format for compatibility with Dynamax.
        This is useful for interfacing with Dynamax inference and sampling methods.
        Returns:
            ParamsLGSSM: Converted parameters.
        """
        #Convert initial distribution
        lggsm_initial = ParamsLGSSMInitial(
            mean=self.initial.mean,
            cov=self.initial.cov
        )
        #Convert dynamics
        lggsm_dynamics = ParamsLGSSMDynamics(
            weights=self.dynamics.weights,
            bias=jnp.zeros(self.dynamics.weights.shape[0]),  # Dynamax expects bias, set to zero
            input_weights=self.dynamics.input_weights,
            cov=self.dynamics.cov
        )
        #Convert emissions
        lggsm_emissions = ParamsLGSSMEmissions(
            weights=self.emissions.weights,
            bias=jnp.zeros(self.emissions.weights.shape[0]),  # Dynamax expects bias, set to zero
            input_weights=None,
            cov=self.emissions.cov
            
        )
        return ParamsLGSSM(
            initial=lggsm_initial,
            dynamics=lggsm_dynamics,
            emissions=lggsm_emissions
        )


class M_Step_State(NamedTuple):
    iteration: int
    delta_A: jnp.ndarray
    delta_C: jnp.ndarray
    delta_Q: jnp.ndarray
    delta_R: jnp.ndarray
class PosteriorCTDSSmoothed(NamedTuple):
    """
    RTS-smoothed posterior over latents.
    """
    marginal_loglik: float
    smoothed_means: Float[Array, "T state_dim"]
    smoothed_covariances: Float[Array, "T state_dim state_dim"]
    smoothed_cross_covariances: Optional[Float[Array, "T_minus_1 state_dim state_dim"]] = None


class PosteriorCTDSFiltered(NamedTuple):
    """
    Kalman filtered posterior over latents.
    """
    marginal_loglik: float
    filtered_means: Float[Array, "T state_dim"]
    filtered_covariances: Float[Array, "T state_dim state_dim"]
    predicted_means: Float[Array, "T state_dim"]
    predicted_covariances: Float[Array, "T state_dim state_dim"]


class SufficientStats(NamedTuple):
    """
    Sufficient statistics for CTDS model parameters.
    Attributes:
        latent_mean: (T, D)  E[x_t]
        latent_second_moment: (T, D, D)  E[x_t x_t^T]
        cross_time_moment: (T-1, D, D)  E[x_t x_{t-1}^T]
        loglik: scalar - marginal log-likelihood
        T: int - number of time steps
    """
    latent_mean: jnp.ndarray             # shape (T, K)        E[x_t]
    latent_second_moment: jnp.ndarray           # shape (T, K, K)     E[x_t x_t^T]
    cross_time_moment: jnp.ndarray      # shape (T-1, K, K)  - E[x_t x_{t-1}^T]
    loglik: float               # scalar             - marginal log-likelihood
    T: int                      # number of time steps
