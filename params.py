import jax.numpy as jnp
from typing import NamedTuple, Union, Optional
from jaxtyping import Float, Array
from dynamax.parameters import ParameterProperties
from dynamax.linear_gaussian_ssm import ParamsLGSSM


class ParamsCTDSInitial(NamedTuple):
    """Initial distribution p(z_1) = N(z_1 | mu, Sigma)"""
    mean: Union[Float[Array, "state_dim"], ParameterProperties]
    cov: Union[
        Float[Array, "state_dim state_dim"],
        Float[Array, "state_dim"],  # for diagonal
        ParameterProperties
    ]


class ParamsCTDSDynamics(NamedTuple):
    """
    Latent dynamics: z_{t+1} = A z_t + w_t,  w_t ~ N(0, Q)
    A has Dale-type sign constraints depending on cell type.
    """
    A: Union[Float[Array, "state_dim state_dim"], ParameterProperties]
    Q: Union[
        Float[Array, "state_dim state_dim"],
        Float[Array, "state_dim"],
        ParameterProperties
    ]
    cell_type_dimensions: Optional[Array]   # shape (K,) where K is number of cell types, optional for flexibility



class ParamsCTDSEmissions(NamedTuple):
    """
    Emission model: y_t = C z_t + v_t,  v_t ~ N(0, R)
    Each row of C corresponds to an observed neuron.
    Sign constraints (Dale's law) apply row-wise by cell type.
    """
    C: Union[Float[Array, "emission_dim state_dim"], ParameterProperties]
    R: Union[
        Float[Array, "emission_dim emission_dim"],
        Float[Array, "emission_dim"],
        ParameterProperties
    ]



class ParamsCTDS(NamedTuple):
    """
    Full CTDS model parameters.
    This structure mirrors Dynamax ParamsLGSSM, 
    but includes explicit sign constraints and cell-type structure.
    """
    initial: ParamsCTDSInitial
    dynamics: ParamsCTDSDynamics
    emissions: ParamsCTDSEmissions
    # true for excitatory, false for inhibitory
    cell_types_mask: jnp.ndarray  # shape (state_dim,) - boolean array indicating cell type
    #TODO: include region identity for each neuron
    #region_identity: Float[Array, "emission_dim"]  # region index for each neuron


    def to_lgssm(self) -> ParamsLGSSM:
        """
        Convert CTDS parameters to LGSSM format for compatibility with Dynamax.
        This is useful for interfacing with Dynamax inference and sampling methods.
        Returns:
            ParamsLGSSM: Converted parameters.
        """
        return ParamsLGSSM(
            initial=self.initial,
            dynamics=self.dynamics,
            emissions=self.emissions
        )



class PosteriorCTDSSmoothed(NamedTuple):
    """
    RTS-smoothed posterior over latents.
    """
    marginal_loglik: float
    smoothed_means: Float[Array, "T state_dim"]
    smoothed_covariances: Float[Array, "T state_dim state_dim"]
    smoothed_cross_covariances: Optional[Float[Array, "T_minus_1 state_dim state_dim"]] = None



class SufficientStats(NamedTuple):
    """
    Sufficient statistics for CTDS model parameters.
    Attributes:
        latent_mean: (T, D) - E[z_t]
        latent_second_moment: (T, D, D) - E[z_t z_t^T]
        cross_time_moment: (T-1, D, D) - E[z_t z_{t-1}^T]
        loglik: scalar - marginal log-likelihood
        T: int - number of time steps
    """
    latent_mean: jnp.ndarray             # shape (T, K)       - E[z_t]
    latent_second_moment: jnp.ndarray           # shape (T, K, K)    - E[z_t z_t^T]
    cross_time_moment: jnp.ndarray      # shape (T-1, K, K)  - E[z_t z_{t-1}^T]
    loglik: float               # scalar             - marginal log-likelihood
    T: int                      # number of time steps
