# User Guide

This guide provides conceptual explanations and detailed tutorials for using CTDS effectively.

## Table of Contents

1. [What is CTDS?](#what-is-ctds)
2. [Key Concepts](#key-concepts)
3. [Model Architecture](#model-architecture)
4. [Parameter Structure](#parameter-structure)
5. [Training Workflow](#training-workflow)
6. [Inference and Prediction](#inference-and-prediction)
7. [Constraints and Dale's Law](#constraints-and-dales-law)
8. [Best Practices](#best-practices)

## What is CTDS?

Cell-Type Dynamical Systems (CTDS) is a probabilistic model for analyzing neural population dynamics while respecting the biological structure of neural circuits. Unlike traditional latent variable models, CTDS incorporates:

- **Cell-type structure**: Different neuron types (excitatory vs inhibitory)
- **Dale's law**: Sign constraints reflecting biological connectivity patterns
- **Population dynamics**: Shared latent dynamics across cell types

### When to Use CTDS

CTDS is particularly useful when:
- You have neural population recordings with known cell types
- You want to enforce biological constraints on connectivity
- You need interpretable latent dynamics
- You're studying circuit-level neural computation

## Key Concepts

### State-Space Model

CTDS is built on a linear state-space model:

```
States:       x_{t+1} = A x_t + w_t,  w_t ~ N(0, Q)
Observations: y_t = C x_t + v_t,      v_t ~ N(0, R)
```

Where:
- `x_t` ∈ ℝ^D: Latent neural state (shared dynamics)
- `y_t` ∈ ℝ^N: Observed neural activity (spikes, calcium, etc.)
- `A` ∈ ℝ^(D×D): Dynamics matrix (how states evolve)
- `C` ∈ ℝ^(N×D): Emission matrix (how states map to observations)
- `Q`, `R`: Process and observation noise covariances

### Cell-Type Structure

The key innovation is incorporating cell-type information:

1. **Cell Types**: Each neuron belongs to a type (e.g., excitatory, inhibitory)
2. **Type Masks**: Binary matrices indicating which neurons belong to each type
3. **Sign Constraints**: Cell types have characteristic signs (+ for excitatory, - for inhibitory)

### Dale's Law

Dale's law states that neurons are either excitatory or inhibitory. In CTDS:
- Excitatory neurons have positive outgoing weights
- Inhibitory neurons have negative outgoing weights
- This is enforced on both dynamics (A) and emissions (C) matrices

## Model Architecture

### Core Components

```python
import ctds

# Basic model creation
model = ctds.CTDS(
    emission_dim=50,    # Number of neurons (N)
    state_dim=10,       # Latent dimension (D)
    cell_types=jnp.array([0, 1]),        # Cell type labels
    cell_sign=jnp.array([1, -1]),        # Type signs (E+, I-)
    cell_type_mask=mask,                 # N×K binary mask
    dale_law=True                        # Enforce Dale's law
)
```

### Cell Type Specification

#### Method 1: Simple Binary Types
```python
# For binary E/I classification
cell_types = jnp.array([0, 1])  # 0=excitatory, 1=inhibitory
cell_sign = jnp.array([1, -1])  # Signs for each type
```

#### Method 2: Custom Mask Matrix
```python
# For fine-grained control
# cell_type_mask[neuron, type] = 1 if neuron belongs to type
cell_type_mask = jnp.array([
    [1, 0],  # Neuron 0 → Type 0 (excitatory)
    [1, 0],  # Neuron 1 → Type 0 (excitatory)
    [0, 1],  # Neuron 2 → Type 1 (inhibitory)
    [0, 1],  # Neuron 3 → Type 1 (inhibitory)
])
```

## Parameter Structure

CTDS parameters are organized hierarchically:

### `ParamsCTDS` (Top Level)
```python
params = ParamsCTDS(
    dynamics=dynamics_params,      # A, Q matrices
    emissions=emissions_params,    # C, R matrices  
    initial=initial_params,        # μ₀, Σ₀
    constraints=constraint_params  # Cell type info
)
```

### Dynamics Parameters (`ParamsCTDSDynamics`)
```python
dynamics = ParamsCTDSDynamics(
    A=A_matrix,              # State transition matrix (D×D)
    Q=Q_matrix,              # Process noise covariance (D×D)
    dynamics_mask=mask       # Dale's law mask for A
)
```

### Emission Parameters (`ParamsCTDSEmissions`)
```python
emissions = ParamsCTDSEmissions(
    C=C_matrix,              # Emission matrix (N×D)
    R=R_matrix,              # Observation noise covariance (N×N)
    emissions_mask=mask      # Dale's law mask for C
)
```

### Constraint Parameters (`ParamsCTDSConstraints`)
```python
constraints = ParamsCTDSConstraints(
    cell_types=types,        # Cell type assignments
    cell_sign=signs,         # Type signs
    cell_type_mask=mask      # Type membership matrix
)
```

## Training Workflow

### 1. Data Preparation

```python
# Load your neural data
observations = load_neural_data()  # Shape: (T, N)

# Standardize (recommended)
observations = (observations - jnp.mean(observations, axis=0)) / jnp.std(observations, axis=0)

# Define cell types (from experimental annotation)
cell_types = get_cell_type_labels()  # Shape: (N,)
```

### 2. Model Initialization

```python
# Create model
model = ctds.CTDS(
    emission_dim=observations.shape[1],
    state_dim=latent_dim,
    cell_types=cell_types,
    cell_sign=jnp.array([1, -1]),  # E+, I-
    dale_law=True
)

# Initialize parameters from data
params = model.initialize(observations)
```

### 3. EM Training

```python
# Fit model using EM algorithm
fitted_params, losses = model.fit_em(
    params, 
    observations,
    num_iters=100,
    verbose=True
)

# Monitor convergence
import matplotlib.pyplot as plt
plt.plot(losses)
plt.xlabel('EM Iteration')
plt.ylabel('Log-likelihood')
plt.title('Training Progress')
```

### 4. Model Validation

```python
# Check constraint satisfaction
constraints = fitted_params.constraints
A = fitted_params.dynamics.A
dale_satisfied = ctds.utils.check_dale_law(A, constraints.cell_sign)
print(f"Dale's law satisfied: {dale_satisfied}")

# Evaluate fit quality
final_loglik = losses[-1]
print(f"Final log-likelihood: {final_loglik}")
```

## Inference and Prediction

### State Estimation (Smoothing)

```python
# Kalman smoothing for state inference
smoothed_means, smoothed_covs = model.smoother(fitted_params, observations)

# Extract latent trajectories
latent_dynamics = smoothed_means  # Shape: (T, D)
```

### Forecasting

```python
# Predict future states
history = observations[-20:]  # Use last 20 steps as context
forecast_steps = 10

forecast_means, forecast_covs = model.forecast(
    fitted_params,
    history,
    num_steps=forecast_steps
)

# Generate predicted observations
predicted_obs = forecast_means @ fitted_params.emissions.C.T
```

### Log-Likelihood Computation

```python
# Evaluate model likelihood
log_prob = model.log_prob(fitted_params, observations)
print(f"Data log-likelihood: {log_prob}")

# Per-timestep likelihoods
timestep_logliks = model.log_prob_timesteps(fitted_params, observations)
```

## Constraints and Dale's Law

### Understanding Dale's Law

Dale's law constrains neural connectivity based on cell types:

1. **Excitatory neurons** (E): All outgoing connections are positive
2. **Inhibitory neurons** (I): All outgoing connections are negative

### Implementation in CTDS

```python
# Dale's law affects two matrices:

# 1. Dynamics matrix A: How latent states interact
# A[i,j] sign determined by the cell type of state j

# 2. Emissions matrix C: How states map to observations  
# C[i,j] sign determined by the cell type of neuron i
```

### Constraint Enforcement

CTDS enforces constraints during the M-step:

```python
# Automatic enforcement during EM
model = ctds.CTDS(dale_law=True)  # Enables constraint enforcement

# Manual constraint checking
A_constrained = ctds.utils.enforce_dale_law(A, cell_types, cell_sign)
```

### Flexible Constraint Specification

```python
# Mixed cell types
cell_types = jnp.array([0, 0, 1, 1, 2])  # 3 cell types
cell_sign = jnp.array([1, -1, 0])        # E+, I-, unconstrained

# Partial constraints
cell_type_mask = jnp.array([
    [1, 0, 0],  # Neuron 0 → Type 0 only
    [0.7, 0.3, 0],  # Neuron 1 → Mixed type (soft assignment)
    [0, 1, 0],  # Neuron 2 → Type 1 only
])
```

## Best Practices

### Model Selection

#### State Dimension
```python
# Try multiple state dimensions
state_dims = [5, 10, 15, 20]
results = {}

for D in state_dims:
    model = ctds.CTDS(emission_dim=N, state_dim=D)
    params = model.initialize(observations)
    _, losses = model.fit_em(params, observations)
    results[D] = losses[-1]  # Final log-likelihood

# Select best dimension
best_D = max(results, key=results.get)
```

#### Cross-Validation
```python
# Time-series split
train_frac = 0.8
split_idx = int(train_frac * len(observations))

train_obs = observations[:split_idx]
test_obs = observations[split_idx:]

# Fit on training data
model = ctds.CTDS(emission_dim=N, state_dim=D)
params = model.initialize(train_obs)
fitted_params, _ = model.fit_em(params, train_obs)

# Evaluate on test data
test_loglik = model.log_prob(fitted_params, test_obs)
```

### Data Preprocessing

#### Standardization
```python
# Z-score normalization (recommended)
observations = (observations - observations.mean(0)) / observations.std(0)

# Min-max scaling (alternative)
observations = (observations - observations.min(0)) / (observations.max(0) - observations.min(0))
```

#### Handling Missing Data
```python
# Simple masking approach
mask = jnp.isfinite(observations)
observations = jnp.where(mask, observations, 0.0)

# More sophisticated: interpolation
from scipy.interpolate import interp1d
# ... interpolation code ...
```

### Performance Optimization

#### GPU Usage
```python
# Move data to GPU
observations = jax.device_put(observations)
params = jax.tree_map(jax.device_put, params)

# Check GPU availability
import jax
print(f"Devices: {jax.devices()}")
```

#### Memory Management
```python
# For large datasets, consider chunking
def fit_chunked(model, params, observations, chunk_size=1000):
    chunks = [observations[i:i+chunk_size] 
              for i in range(0, len(observations), chunk_size)]
    
    for chunk in chunks:
        params, _ = model.fit_em(params, chunk, num_iters=10)
    
    return params
```

#### JIT Compilation
```python
# Pre-compile inference functions
@jax.jit
def fast_smoother(params, obs):
    return model.smoother(params, obs)

@jax.jit  
def fast_forecast(params, history, steps):
    return model.forecast(params, history, steps)
```

### Initialization Strategies

#### From Factor Analysis
```python
# Initialize with factor analysis
from sklearn.decomposition import FactorAnalysis

fa = FactorAnalysis(n_components=state_dim)
fa.fit(observations)

# Use FA loadings as initial C matrix
initial_C = fa.components_.T
```

#### Random Initialization
```python
# Custom random initialization
key = jax.random.PRNGKey(42)
params = model.initialize(observations, key=key)
```

### Diagnostic Tools

#### Parameter Inspection
```python
# Check eigenvalues for stability
A = fitted_params.dynamics.A
eigenvals = jnp.linalg.eigvals(A)
stable = jnp.all(jnp.abs(eigenvals) < 1.0)
print(f"Dynamics stable: {stable}")

# Analyze emission loadings
C = fitted_params.emissions.C
loading_norms = jnp.linalg.norm(C, axis=1)
print(f"Average loading magnitude: {loading_norms.mean()}")
```

#### Convergence Monitoring
```python
# Track multiple metrics during training
metrics = {
    'log_likelihood': [],
    'param_change': [],
    'constraint_violation': []
}

# Custom EM loop with monitoring
for iter in range(num_iters):
    new_params, loglik = model.em_step(params, observations)
    
    metrics['log_likelihood'].append(loglik)
    metrics['param_change'].append(param_distance(params, new_params))
    metrics['constraint_violation'].append(check_constraints(new_params))
    
    params = new_params
```

## Troubleshooting

### Common Issues and Solutions

1. **Non-convergence**
   - Check data scaling and initialization
   - Try smaller learning rates
   - Increase regularization

2. **Constraint violations**
   - Verify cell type specification
   - Check sign constraints are correct
   - Use stronger regularization

3. **Numerical instability**
   - Add diagonal regularization to covariance matrices
   - Use double precision (`jax.config.update("jax_enable_x64", True)`)
   - Scale down learning rates

4. **Memory errors**
   - Reduce state dimension
   - Process data in chunks
   - Move to CPU if GPU memory limited

### Debugging Workflow

```python
# 1. Validate inputs
assert observations.shape[0] > observations.shape[1], "Need T > N"
assert jnp.isfinite(observations).all(), "Remove NaN/Inf values"

# 2. Check model configuration
model.validate_config()

# 3. Monitor parameter evolution
param_norms = jax.tree_map(lambda x: jnp.linalg.norm(x), params)
print(f"Parameter norms: {param_norms}")

# 4. Validate final result
final_params = fitted_params
constraints_satisfied = ctds.utils.validate_constraints(final_params)
print(f"All constraints satisfied: {constraints_satisfied}")
```

## Next Steps

- Try the [examples](../examples/) to see CTDS in action
- Explore the [API reference](../reference/) for detailed documentation
- Run the quickstart script for hands-on experience
- Check out the Jupyter notebooks for in-depth tutorials
