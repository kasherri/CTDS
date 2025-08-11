from typing import Protocol, Tuple, Union, Optional
from jaxtyping import Array, Float
import jax.numpy as jnp

#TODO: ask if i can make linear_gaussian jittable
from dynamax.linear_gaussian_ssm.inference import lgssm_smoother, lgssm_filter, lgssm_posterior_sample
from params import ParamsCTDS
from utlis import compute_sufficient_statistics
import jax
from functools import partial

class InferenceBackend(Protocol):
    """
    Abstract interface for CTDS inference backends.

    Any inference backend (e.g., Parallel, Variational, Switching)
    must implement the following methods:

    - `e_step`: Compute sufficient statistics using posterior expectations.
    - `filter`: Compute forward filtered means/covariances (optional).
    - `smoother`: Compute smoothed means/covariances (required).
    - `posterior_sample`: Draw a posterior sample of the latent states (optional).

    All implementations must be JAX-compatible and support JIT compilation.

    Attributes:
        - None (stateless interface). Implementations can carry internal config if needed.
    """

    def e_step(
        self,
        params: ParamsCTDS,
        emissions: Float[Array, "T D"],
        inputs: Optional[Float[Array, "T U"]] = None,
    ):
        """
        E-step of EM: compute expected sufficient statistics and marginal log likelihood.

        Args:
            params: Model parameters for CTDS.
            emissions: Observed data of shape (T, D).
            inputs: Optional control inputs of shape (T, U).

        Returns:
            - Dictionary of sufficient statistics (e.g., Ex, Vx, ExxT, etc.).
            - Scalar marginal log-likelihood.
        """
        ...

    def smoother(
        self,
        params: ParamsCTDS,
        emissions: Float[Array, "T D"],
        inputs: Optional[Float[Array, "T U"]] = None,
    ) -> Tuple[Float[Array, "T K"], Float[Array, "T K K"]]:
        """
        Compute posterior means and covariances for all time steps.

        Args:
            params: CTDS model parameters.
            emissions: Emission data (T, D).
            inputs: Optional control inputs (T, U).

        Returns:
            - Smoothed means (T, K)
            - Smoothed covariances (T, K, K)
        """
        ...

    def filter(
        self,
        params: ParamsCTDS,
        emissions: Float[Array, "T D"],
        inputs: Optional[Float[Array, "T U"]] = None,
    ) -> Tuple[Float[Array, "T K"], Float[Array, "T K K"]]:
        """
        (Optional) Compute forward filtered means and covariances.

        Returns:
            - Filtered means (T, K)
            - Filtered covariances (T, K, K)
        """
        ...

    def posterior_sample(
        self,
        key: jnp.ndarray,
        params: ParamsCTDS,
        emissions: Float[Array, "T D"],
        inputs: Optional[Float[Array, "T U"]] = None,
    ) -> Float[Array, "T K"]:
        """
        (Optional) Sample a posterior trajectory of latent states.

        Args:
            key: PRNG key for sampling.
            params: CTDS model parameters.
            emissions: Observed data (T, D).
            inputs: Optional inputs (T, U).

        Returns:
            Sampled latent state trajectory (T, K).
        """
        ...





class DynamaxLGSSMBackend:
    #@partial(jax.jit, static_argnums=0)
    @staticmethod
    def e_step( params: ParamsCTDS, emissions, inputs=None):
        """
        Compute expected sufficient statistics and marginal log likelihood using the smoother.
        """
        print("emissions shape", emissions.shape)
        print("lggsm emission shape", params.to_lgssm().emissions.weights.shape)
        posterior = lgssm_smoother(params.to_lgssm(), emissions, inputs)
        stats = compute_sufficient_statistics(posterior)
        return stats,  stats.loglik

    
    def filter(self,params: ParamsCTDS, emissions, inputs=None):
        """
        Compute forward filtered means and covariances.
        """
        posterior = lgssm_filter(params.to_lgssm(), emissions, inputs)
        return posterior.filtered_means, posterior.filtered_covariances
    @staticmethod
    def smoother( params: ParamsCTDS, emissions, inputs=None):
        """
        Compute posterior means and covariances for all time steps.
        """
        posterior = lgssm_smoother(params.to_lgssm(), emissions, inputs)
        return posterior.smoothed_means, posterior.smoothed_covariances

    def posterior_sample(self, key, params: ParamsCTDS, emissions, inputs=None):
        """
        Sample a posterior trajectory of latent states.
        """
        return lgssm_posterior_sample(key, params.to_lgssm(), emissions, inputs)













