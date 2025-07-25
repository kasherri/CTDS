import jax
import jax.numpy as jnp
from typing import Optional, Dict, Any
from dynamax.linear_gaussian_ssm.models import LinearGaussianSSM
from dynamax.linear_gaussian_ssm.inference import (
    ParamsLGSSM,
    ParamsLGSSMInitial,
    ParamsLGSSMDynamics,
    ParamsLGSSMEmissions
)
from components.ctds_dynamics import CTDSDynamics
from .components.ctds_emissions import CTDSEmissions



#write switching method also later
class CTDSModel:
    """
    Cell-Type Dynamical System (CTDS) wrapper around Dynamax's LinearGaussianSSM.

      1. Instantiate with dimensional and metadata information.
      2. Call `initialize()` to construct constrained A, Q, C, R matrices and initialize ParamsLGSSM.
      3. Use a trainer to run EM inference with these initialized parameters.

    ------------------------
    Args:
        num_timesteps(T):   Number of time steps in the time series data.
        num_latents(D):     Dimensionality of the latent space.
        num_observations(N): Number of recorded neurons.
        cell_identity:      Dictionary with cell label: [cell_type(E or I), #of dimensions. #of neurons] key value pair
        oberservations(Y): (N, T) array of observed neuron activity.


    Attributes:
        model:              Dynamax LinearGaussianSSM instance.
        params:             Parameters of the model (set via initialize_params or manually).
    
    """
    def __init__(
        self,
        num_timesteps: int,
        num_latents: int,
        num_observations: int,
        cell_identity: jnp.ndarray,
        list_of_dimensions: jnp.ndarray,
        observations: jnp.ndarray,
        region_identity: Optional[jnp.ndarray] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.num_timesteps = num_timesteps
        self.num_latents = num_latents
        self.num_observations = num_observations
        self.cell_identity = cell_identity.astype(int) #just incase not ints
        self.list_of_dimensions = list_of_dimensions.astype(int)
        self.region_identity = (
            region_identity.astype(int)
            if region_identity is not None
            else jnp.zeros_like(cell_identity, dtype=int)
        )
        self.config = config or {}

        #  Dynamax model
        self.model = LinearGaussianSSM(
            state_dim=self.num_latents,
            emission_dim=self.num_observations
        )
        self.params: Optional[ParamsLGSSM] = None

    def initialize(self) -> ParamsLGSSM:
        """
        Build and set CTDS-specific parameters:
          - A, Q via CTDSDynamics
          - C, R via CTDSEmissions
          - mu0, Sigma0 as zeros and identity

        returns:
            The constructed ParamsLGSSM object
        """
        #Learn J

        # Dynamics
        dyn_builder = CTDSDynamics(
            list_of_dimensions=self.list_of_dimensions,
            within_region=self.config.get("within_region", True),
            across_region=self.config.get("across_region", True),
            base_strength=self.config.get("dynamics_base_strength", 0.99),
            noise_scale=self.config.get("dynamics_noise_scale", 0.1)
        )
        A, Q = dyn_builder.build()

        # Emissions
        emis_builder = CTDSEmissions(40, len(self.list_of_dimensions),
            self.cell_identity,
            self.region_identity,
            self.list_of_dimensions, 

        )
        C, R = emis_builder.build()

        # Initial state
        mu0 = jnp.zeros(self.num_latents)
        Sigma0 = jnp.eye(self.num_latents)

        # Assemble ParamsLGSSM
        params = ParamsLGSSM(
            initial=ParamsLGSSMInitial(mean=mu0, cov=Sigma0),
            dynamics=ParamsLGSSMDynamics(
                weights=A,
                cov=Q,
                bias=None,
                input_weights=jnp.zeros((self.num_latents, 0))
            ),
            emissions=ParamsLGSSMEmissions(
                weights=C,
                cov=R,
                bias=None,
                input_weights=jnp.zeros((self.num_observations, 0))
            )
        )

        self.params = params
        return params

    def set_params(self, params: ParamsLGSSM):
        """
        Override model parameters with a user-supplied ParamsLGSSM object.
        """
        self.params = params

    def get_params(self) -> ParamsLGSSM:
        """
        Retrieve the current parameter set.
        """
        if self.params is None:
            raise ValueError("Parameters have not been initialized. Call `initialize_params()` first.")
        return self.params

    def sample(
        self,
        T: Optional[int] = None,
        prefix: Optional[Any] = None,
        rng: Optional[jax.random.key_data] = None,
        **kwargs
    ):
        """Sample latent states and observations from the model.

        Args:
            T: Number of timesteps. Defaults to ``self.num_timesteps``.
            prefix: Optional initial state for the underlying model.
            rng: JAX PRNG key.

        Returns:
            A tuple ``(states, observations)`` from the generative model.
        """
        if self.params is None:
            raise ValueError(
                "Parameters have not been initialized. Call `initialize_params()` first."
            )

        T = self.num_timesteps if T is None else T
        rng = jax.random.PRNGKey(0) if rng is None else rng

        return self.model.sample(self.params, T=T, prefix=prefix, rng=rng, **kwargs)
