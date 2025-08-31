# CTDS: Cell-Type Dynamical Systems

Welcome to the documentation for **Cell-Type Dynamical Systems (CTDS)**, a JAX-native implementation for modeling neural population dynamics with biologically-constrained connectivity.

## Overview

CTDS models neural population activity using linear dynamical systems that respect cell-type constraints and Dale's law. The model decomposes neural connectivity into cell-type-specific factors while maintaining biological plausibility.

### Key Features

- **Cell-type structure**: Models connectivity through cell-type-specific factorization
- **Dale's law constraints**: Enforces biologically plausible sign constraints  
- **JAX-native**: Full JAX compatibility for GPU acceleration and JIT compilation
- **EM training**: Efficient parameter estimation via expectation-maximization
- **Dynamax integration**: Uses Dynamax for exact inference in linear Gaussian models

## Mathematical Framework

The CTDS model describes neural dynamics as:

**Latent dynamics:**
$$x_{t+1} = A x_t + B u_t + w_t, \quad w_t \sim \mathcal{N}(0, Q)$$

**Observations:**  
$$y_t = C x_t + v_t, \quad v_t \sim \mathcal{N}(0, R)$$

**Connectivity factorization:**
$$A = U V^T$$

where:

- $U \in \mathbb{R}^{D \times K}$: cell-type factor matrix (non-negative for excitatory factors)
- $V \in \mathbb{R}^{D \times K}$: Dale-constrained connectivity weights 
- $C \in \mathbb{R}^{N \times D}$: emission matrix (non-negative, cell-type structured)

## Quick Start

```python
import jax.numpy as jnp
import jax.random as jr
from ctds import CTDS

# Generate synthetic data
key = jr.PRNGKey(0)
T, N, D = 100, 50, 10  # time steps, neurons, latent dim

# Create CTDS model
model = CTDS(
    state_dim=D,
    emission_dim=N, 
    num_cell_types=2,  # E and I populations
    cell_type_percentages=[0.8, 0.2]  # 80% E, 20% I
)

# Fit to neural data Y (shape: N × T)
params = model.initialize(Y)
params, losses = model.fit_em(params, Y, num_iters=100)

# Generate predictions
states = model.sample(params, key, T)
forecasted = model.forecast(params, Y[:, :T//2], T//2)
```

## Navigation

- **[Getting Started](getting-started/installation.md)**: Installation and basic setup
- **[User Guide](user-guide/mathematical-framework.md)**: Comprehensive usage documentation  
- **[API Reference](reference/)**: Detailed API documentation
- **[Examples](examples/)**: Jupyter notebook examples


