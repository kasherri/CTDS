# Quick Start

This guide will help you get started with CTDS for modeling neural population dynamics.

## Basic Usage

```python
import jax.numpy as jnp
import jax.random as jr
from ctds import CTDS

# Set up model parameters
emission_dim = 50  # Number of neurons
cell_types = 2     # Number of cell types (e.g., excitatory/inhibitory)
state_dim = 10     # Latent state dimension

# Create model
key = jr.PRNGKey(0)
model = CTDS(
    emission_dim=emission_dim,
    cell_types=cell_types,
    state_dim=state_dim
)

# Generate synthetic data
num_timesteps = 100
synthetic_data = model.sample(
    model.initialize(key, num_timesteps),
    key=key,
    num_timesteps=num_timesteps
)

# Fit the model using EM
params = model.initialize(key, num_timesteps)
fitted_params, _ = model.fit_em(params, synthetic_data, num_iters=50)

# Perform inference
posterior = model.smoother(fitted_params, synthetic_data)
```

## Key Concepts

### Cell Types
CTDS models different cell types (e.g., excitatory and inhibitory neurons) with distinct dynamics.

### Dale's Principle
The model enforces Dale's principle, ensuring neurons maintain consistent excitatory or inhibitory roles.

### State Space Model
CTDS uses a linear Gaussian state space model with cell-type-specific constraints.

## Next Steps

- Explore the [examples](examples/) for detailed use cases
- Check the [API reference](reference/) for complete documentation
- See the notebooks for visualization and analysis examples
