import jax.numpy as jnp
from typing import Optional, Dict, Any
from dynamax.linear_gaussian_ssm.models import LinearGaussianSSM
from dynamax.linear_gaussian_ssm.inference import (
    ParamsLGSSM,
    ParamsLGSSMInitial,
    ParamsLGSSMDynamics,
    ParamsLGSSMEmissions
)

from models.components.ctds_dynamics import CTDSDynamics
from models.components.ctds_emissions import CTDSEmissions


class CTDSModel:
    """
    Cell-Type Dynamical System (CTDS) using Dynamax's LinearGaussianSSM.

    how to use:
      1. Instantiate with dimensions and metadata
      2. Call `initialize_params()` to build CTDS-specific A, Q, C, R and set `self.params`
      3. Use external trainer/eval for EM, prediction, likelihood, etc.
    """
    def __init__(
        self,
        num_timesteps: int,
        num_latents: int,
        num_observations: int,
        cell_identity: jnp.ndarray,
        list_of_dimensions: jnp.ndarray,
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

    def initialize_params(self) -> ParamsLGSSM:
        """
        Build and set CTDS-specific parameters:
          - A, Q via CTDSDynamics
          - C, R via CTDSEmissions
          - mu0, Sigma0 as zeros and identity

        returns:
            The constructed ParamsLGSSM object
        """
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
        emis_builder = CTDSEmissions(
            cell_identity=self.cell_identity,
            list_of_dimensions=self.list_of_dimensions,
            region_identity=self.region_identity,
            base_strength=self.config.get("emissions_base_strength", 1.0),
            noise_scale=self.config.get("emissions_noise_scale", 0.5),
            normalize=self.config.get("emissions_normalize", False)
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
