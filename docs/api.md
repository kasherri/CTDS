# CTDS API Documentation

## Core Components

### Parameters

The parameter system in CTDS uses a hierarchical structure to organize model components:

```python
ParamsCTDS
├── initial: ParamsCTDSInitial      # Initial state distribution
├── dynamics: ParamsCTDSDynamics    # Transition dynamics A, Q
├── emissions: ParamsCTDSEmissions  # Observation model C, R
├── constraints: ParamsCTDSConstraints  # Cell-type constraints
└── observations: Array             # Observed data
```

#### ParamsCTDSInitial
- `mean`: Initial state mean μ₀ ∈ ℝᴰ
- `cov`: Initial state covariance Σ₀ ∈ ℝᴰˣᴰ

#### ParamsCTDSDynamics  
- `weights`: Dynamics matrix A ∈ ℝᴰˣᴰ
- `cov`: Process noise covariance Q ∈ ℝᴰˣᴰ
- `dynamics_mask`: Dale's law sign constraints

#### ParamsCTDSEmissions
- `weights`: Emission matrix C ∈ ℝᴺˣᴰ  
- `cov`: Observation noise covariance R ∈ ℝᴺˣᴺ
- `emission_dims`: Cell-type block sizes
- `left_padding_dims`, `right_padding_dims`: Block structure

#### ParamsCTDSConstraints
- `cell_type_dimensions`: Number of cells per type
- `cell_sign`: Dale's law signs (+1 excitatory, -1 inhibitory)
- `num_cell_types`: Total number of cell types

### Models

#### CTDS Class

The main model class inheriting from Dynamax SSM:

```python
class CTDS(SSM):
    def __init__(self, state_dim, emission_dim, num_cell_types=2, 
                 cell_type_percentages=None, **kwargs)
```

**Key Methods:**

- `initialize(observations)`: Initialize parameters from data
- `fit_em(params, observations, num_iters=100, **kwargs)`: EM training
- `sample(params, key, T)`: Generate synthetic data
- `forecast(params, emissions, steps, **kwargs)`: Missing data prediction
- `smoother(params, emissions)`: Posterior inference
- `filter(params, emissions)`: Causal inference

### Inference

#### InferenceBackend Protocol

Abstract interface for inference methods:

```python
class InferenceBackend(Protocol):
    def e_step(params, emissions, inputs=None) -> Tuple[SufficientStats, float]
    def smoother(params, emissions, inputs=None) -> Tuple[Array, Array]  
    def filter(params, emissions, inputs=None) -> Tuple[Array, Array]
    def posterior_sample(key, params, emissions, inputs=None) -> Array
```

#### DynamaxLGSSMBackend

Exact inference using Kalman filtering/smoothing:

- Uses Dynamax's linear Gaussian SSM routines
- Converts CTDS parameters via `params.to_lgssm()`
- Provides optimal posterior estimates under linearity

### Utilities

Key utility functions for constrained optimization:

- `solve_dale_QP(Q, c, mask, key)`: Dale's law constrained QP
- `solve_constrained_QP(...)`: General constrained optimization
- `blockwise_NNLS(...)`: Non-negative least squares with block structure
- `compute_sufficient_statistics(posterior)`: EM sufficient statistics
- `estimate_J(Y, mask)`: Connectivity estimation from data

## Usage Patterns

### Basic Workflow

1. **Data preparation**: Format neural data as `(N, T)` array
2. **Model creation**: Instantiate CTDS with appropriate dimensions
3. **Initialization**: Use `model.initialize(Y)` for data-driven initialization  
4. **Training**: Run `model.fit_em(params, Y)` with convergence monitoring
5. **Inference**: Use `model.smoother()` or `model.filter()` for posterior estimates
6. **Forecasting**: Use `model.forecast()` for missing data prediction

### Advanced Configuration

For custom cell-type structures:

```python
# Define custom constraints
constraints = ParamsCTDSConstraints(
    cell_type_dimensions=[n_E, n_I],
    cell_sign=signs,  # +1 for E, -1 for I  
    num_cell_types=2
)

# Create model with constraints
model = CTDS(state_dim=D, emission_dim=N, constraints=constraints)
```

### Performance Optimization

- Enable JIT compilation: Most functions are `@jax.jit` compatible
- Use GPU acceleration: Full JAX GPU support
- Batch processing: Add batch dimension for multiple sequences
- Memory management: Monitor convergence to avoid over-fitting

## Mathematical Details

### Model Equations

**State dynamics:**
$$x_{t+1} = A x_t + w_t,  w_t ~ N(0, Q)$$



**Observations:**
```  
y_t = C x_t + v_t,     v_t ~ N(0, R)
```

**Factorization:**
```
A = U V^T
```

### Constraints

**Dale's Law:**
- A[i,j] ≥ 0 if neuron j is excitatory
- A[i,j] ≤ 0 if neuron j is inhibitory

**Cell-Type Structure:**
- C has block structure respecting cell types
- Non-negative emission weights preserve Dale signs

### EM Algorithm

**E-step:** Compute sufficient statistics via Kalman smoother
- E[x_t | y_{1:T}]: posterior means
- E[x_t x_t^T | y_{1:T}]: posterior second moments
- E[x_t x_{t-1}^T | y_{1:T}]: cross-time moments

**M-step:** Update parameters via constrained optimization
- A: Constrained QP with Dale's law
- C: Block-wise NNLS with cell-type structure  
- Q, R: Residual covariance updates
- Initial state: First time point posterior
