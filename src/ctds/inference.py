"""
Inference backends for Cell-Type Dynamical Systems (CTDS).

This module provides abstract interfaces and concrete implementations
for performing posterior inference in CTDS models. The main components are:

1. **InferenceBackend Protocol**: Abstract interface defining required methods
   for any CTDS inference backend.

2. **DynamaxLGSSMBackend**: Concrete implementation using Dynamax's linear
   Gaussian state-space model routines for exact inference.

Classes
-------
InferenceBackend : Protocol
    Abstract interface for CTDS inference backends.
DynamaxLGSSMBackend : class
    Dynamax-based exact inference implementation.

Notes
-----
All implementations must be JAX-compatible and support JIT compilation
for high-performance inference on large neural datasets.
"""

from typing import Protocol, Tuple, Union, Optional
from jaxtyping import Array, Float
import jax.numpy as jnp

#TODO: ask if i can make linear_gaussian jittable
from dynamax.linear_gaussian_ssm.inference import lgssm_smoother, lgssm_filter, lgssm_posterior_sample
from .params import ParamsCTDS
from .utils import compute_sufficient_statistics
import jax
from functools import partial

class InferenceBackend(Protocol):
    """
    Abstract interface for CTDS inference backends.

    This protocol defines the required methods that any CTDS inference
    backend must implement. Concrete backends can use different algorithms
    (e.g., exact Kalman smoothing, variational inference, particle filtering)
    while maintaining a consistent interface.

    Methods
    -------
    e_step : Compute sufficient statistics for EM algorithm
    smoother : Compute posterior state estimates
    filter : Compute forward filtered estimates (optional)
    posterior_sample : Sample from posterior (optional)

    Notes
    -----
    All implementations must be JAX-compatible and support JIT compilation.
    Backends should handle time-major format (T, D) for emissions and states.

    Examples
    --------
    >>> backend = DynamaxLGSSMBackend()
    >>> stats, loglik = backend.e_step(params, emissions)
    >>> means, covs = backend.smoother(params, emissions)
    """

    def e_step(
        self,
        params: ParamsCTDS,
        emissions: Float[Array, "T D"],
        inputs: Optional[Float[Array, "T U"]] = None,
    ):
        """
        E-step of EM: compute expected sufficient statistics.

        Parameters
        ----------
        params : ParamsCTDS
            Current model parameters.
        emissions : Array, shape (T, D)
            Observed emission sequence.
        inputs : Array, shape (T, U), optional
            Exogenous input sequence.

        Returns
        -------
        stats : SufficientStats
            Expected sufficient statistics for M-step.
        loglik : float
            Marginal log-likelihood of emissions.

        Notes
        -----
        Sufficient statistics typically include:
        - E[x_t]: posterior means
        - E[x_t x_t^T]: posterior second moments  
        - E[x_t x_{t-1}^T]: cross-time moments
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

        Parameters
        ----------
        params : ParamsCTDS
            Model parameters.
        emissions : Array, shape (T, D)
            Observed emission sequence.
        inputs : Array, shape (T, U), optional
            Exogenous input sequence.

        Returns
        -------
        means : Array, shape (T, K)
            Posterior state means.
        covariances : Array, shape (T, K, K)
            Posterior state covariances.

        Notes
        -----
        This is the primary inference method providing full posterior
        estimates conditioned on all observations y_{1:T}.
        """
        ...

    def filter(
        self,
        params: ParamsCTDS,
        emissions: Float[Array, "T D"],
        inputs: Optional[Float[Array, "T U"]] = None,
    ) -> Tuple[Float[Array, "T K"], Float[Array, "T K K"]]:
        """
        Compute forward filtered means and covariances.

        Parameters
        ----------
        params : ParamsCTDS
            Model parameters.
        emissions : Array, shape (T, D)
            Observed emission sequence.
        inputs : Array, shape (T, U), optional
            Exogenous input sequence.

        Returns
        -------
        means : Array, shape (T, K)
            Filtered state means.
        covariances : Array, shape (T, K, K)
            Filtered state covariances.

        Notes
        -----
        Optional method providing causal estimates p(x_t | y_{1:t}).
        Useful for online inference and real-time applications.
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
        Sample a posterior trajectory of latent states.

        Parameters
        ----------
        key : PRNGKey
            Random key for sampling.
        params : ParamsCTDS
            Model parameters.
        emissions : Array, shape (T, D)
            Observed emission sequence.
        inputs : Array, shape (T, U), optional
            Exogenous input sequence.

        Returns
        -------
        states : Array, shape (T, K)
            Sampled latent state trajectory.

        Notes
        -----
        Optional method for sampling from p(x_{1:T} | y_{1:T}).
        Useful for uncertainty quantification and model validation.
        """
        ...





class DynamaxLGSSMBackend:
    """
    Dynamax-based exact inference backend for CTDS.

    This backend implements exact Bayesian inference for CTDS models using
    Dynamax's efficient linear Gaussian state-space model routines. It provides
    optimal posterior estimates when the CTDS model satisfies linearity and
    Gaussian noise assumptions.

    Methods
    -------
    e_step : Compute sufficient statistics using RTS smoother
    filter : Compute forward Kalman filter estimates  
    smoother : Compute RTS smoother estimates
    posterior_sample : Sample posterior trajectories

    Notes
    -----
    All methods are static and JAX-compatible. The backend automatically
    converts CTDS parameters to Dynamax LGSSM format using the `to_lgssm()`
    method from ParamsCTDS.

    Examples
    --------
    >>> backend = DynamaxLGSSMBackend()
    >>> stats, loglik = backend.e_step(params, emissions)
    >>> posterior_means, posterior_covs = backend.smoother(params, emissions)
    """

    @staticmethod
    def e_step(params: ParamsCTDS, emissions, inputs=None):
        """
        Compute expected sufficient statistics using RTS smoother.

        Parameters
        ----------
        params : ParamsCTDS
            CTDS model parameters.
        emissions : Array, shape (T, N)
            Observed emission sequence.
        inputs : Array, shape (T, U), optional
            Exogenous input sequence.

        Returns
        -------
        stats : SufficientStats
            Expected sufficient statistics containing:
            - latent_mean: E[x_t], shape (T, D)
            - latent_second_moment: E[x_t x_t^T], shape (T, D, D)
            - cross_time_moment: E[x_t x_{t-1}^T], shape (T-1, D, D)
            - loglik: marginal log-likelihood
            - T: sequence length
        loglik : float
            Marginal log-likelihood p(y_{1:T}).

        Notes
        -----
        Uses Dynamax's RTS smoother for optimal posterior estimates
        under linear Gaussian assumptions. Sufficient statistics are
        computed from smoothed posterior moments.
        """
        posterior = lgssm_smoother(params.to_lgssm(), emissions, inputs)
        stats = compute_sufficient_statistics(posterior)
        return stats, stats.loglik

    @staticmethod
    def filter(params: ParamsCTDS, emissions, inputs=None):
        """
        Compute forward Kalman filter estimates.

        Parameters
        ----------
        params : ParamsCTDS
            CTDS model parameters.
        emissions : Array, shape (T, N)
            Observed emission sequence.
        inputs : Array, shape (T, U), optional
            Exogenous input sequence.

        Returns
        -------
        filtered_means : Array, shape (T, D)
            Forward filtered state means E[x_t | y_{1:t}].
        filtered_covariances : Array, shape (T, D, D)
            Forward filtered state covariances Cov[x_t | y_{1:t}].

        Notes
        -----
        Provides causal estimates suitable for online applications.
        Uses Dynamax's forward Kalman filter implementation.
        """
        posterior = lgssm_filter(params.to_lgssm(), emissions, inputs)
        return posterior.filtered_means, posterior.filtered_covariances

    @staticmethod
    def smoother(params: ParamsCTDS, emissions, inputs=None):
        """
        Compute RTS smoother estimates.

        Parameters
        ----------
        params : ParamsCTDS
            CTDS model parameters.
        emissions : Array, shape (T, N)
            Observed emission sequence.
        inputs : Array, shape (T, U), optional
            Exogenous input sequence.

        Returns
        -------
        smoothed_means : Array, shape (T, D)
            Smoothed state means E[x_t | y_{1:T}].
        smoothed_covariances : Array, shape (T, D, D)
            Smoothed state covariances Cov[x_t | y_{1:T}].

        Notes
        -----
        Provides optimal posterior estimates using full observation sequence.
        Uses Dynamax's Rauch-Tung-Striebel (RTS) smoother implementation.
        """
        posterior = lgssm_smoother(params.to_lgssm(), emissions, inputs)
        return posterior.smoothed_means, posterior.smoothed_covariances

    @staticmethod
    def posterior_sample(key, params: ParamsCTDS, emissions, inputs=None):
        """
        Sample posterior trajectory of latent states.

        Parameters
        ----------
        key : PRNGKey
            Random key for sampling.
        params : ParamsCTDS
            CTDS model parameters.
        emissions : Array, shape (T, N)
            Observed emission sequence.
        inputs : Array, shape (T, U), optional
            Exogenous input sequence.

        Returns
        -------
        states : Array, shape (T, D)
            Sampled latent state trajectory from p(x_{1:T} | y_{1:T}).

        Notes
        -----
        Uses Dynamax's posterior sampling algorithm which first runs
        the smoother then samples backwards from the posterior.
        """
        return lgssm_posterior_sample(key, params.to_lgssm(), emissions, inputs)













