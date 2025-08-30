# CTDS Examples

## Example 1: Basic Model Fitting

```python
import jax.numpy as jnp
import jax.random as jr
from models import CTDS

# Synthetic data: 50 neurons, 200 time steps
key = jr.PRNGKey(42)
T, N, D = 200, 50, 10

# Create synthetic neural activity (replace with real data)
Y = jr.normal(key, (N, T))

# Create CTDS model
model = CTDS(
    state_dim=D,
    emission_dim=N,
    num_cell_types=2,
    cell_type_percentages=[0.8, 0.2]  # 80% excitatory, 20% inhibitory
)

# Initialize parameters
params = model.initialize(Y)
print(f"Initialized with {params.dynamics.weights.shape} dynamics matrix")

# Fit model with EM
params_fitted, losses = model.fit_em(
    params, Y, 
    num_iters=100,
    tolerance=1e-6,
    verbose=True
)

print(f"Final log-likelihood: {losses[-1]:.2f}")
```

## Example 2: Custom Cell Types

```python
from params import ParamsCTDSConstraints

# Define custom cell-type structure
n_E, n_I = 8, 2  # 8 excitatory, 2 inhibitory
cell_signs = jnp.array([1]*n_E + [-1]*n_I)  # Dale's law signs

constraints = ParamsCTDSConstraints(
    cell_type_dimensions=[n_E, n_I],
    cell_sign=cell_signs,
    num_cell_types=2
)

# Create model with custom constraints
model = CTDS(
    state_dim=n_E + n_I,
    emission_dim=25,
    constraints=constraints
)

# Rest of fitting process same as Example 1
params = model.initialize(Y)
params_fitted, losses = model.fit_em(params, Y, num_iters=100)
```

## Example 3: Posterior Inference

```python
# After fitting model (from Examples 1 or 2)

# Compute posterior state estimates
posterior_means, posterior_covs = model.smoother(params_fitted, Y)
print(f"Posterior means shape: {posterior_means.shape}")  # (T, D)

# Compute causal (filtered) estimates  
filtered_means, filtered_covs = model.filter(params_fitted, Y)
print(f"Filtered means shape: {filtered_means.shape}")   # (T, D)

# Sample from posterior
key = jr.PRNGKey(123)
posterior_sample = model.posterior_sample(key, params_fitted, Y)
print(f"Posterior sample shape: {posterior_sample.shape}")  # (T, D)
```

## Example 4: Forecasting Missing Data

```python
# Split data into observed and missing portions
T_obs = T // 2  # Use first half for fitting
Y_obs = Y[:, :T_obs]
Y_true = Y[:, T_obs:]  # Ground truth for validation

# Fit model on observed data
params = model.initialize(Y_obs)
params_fitted, _ = model.fit_em(params, Y_obs, num_iters=50)

# Forecast missing data
steps_ahead = T - T_obs
forecasted = model.forecast(
    params_fitted, 
    Y_obs, 
    steps=steps_ahead,
    method='mean'  # Use posterior mean
)

print(f"Forecasted shape: {forecasted.shape}")  # (N, steps_ahead)

# Compute forecast error
mse = jnp.mean((forecasted - Y_true)**2)
print(f"Forecast MSE: {mse:.4f}")
```

## Example 5: Generating Synthetic Data

```python
# Generate synthetic data from fitted model
key = jr.PRNGKey(456) 
T_synth = 300

# Sample latent states
synthetic_states = model.sample(params_fitted, key, T_synth)
print(f"Synthetic states shape: {synthetic_states.shape}")  # (T_synth, D)

# Generate corresponding observations
key, subkey = jr.split(key)
synthetic_obs = model.sample(params_fitted, subkey, T_synth, states=synthetic_states)
print(f"Synthetic observations shape: {synthetic_obs.shape}")  # (N, T_synth)
```

## Example 6: Batch Processing Multiple Sequences

```python
# Multiple sequences (e.g., different trials)
n_trials = 5
Y_batch = jr.normal(jr.PRNGKey(789), (n_trials, N, T))

# Fit to batch data
batch_params = []
for i in range(n_trials):
    params = model.initialize(Y_batch[i])
    params_fitted, _ = model.fit_em(params, Y_batch[i], num_iters=50)
    batch_params.append(params_fitted)

print(f"Fitted {len(batch_params)} models")

# Analyze dynamics matrices across trials
A_matrices = [p.dynamics.weights for p in batch_params]
A_mean = jnp.mean(jnp.stack(A_matrices), axis=0)
A_std = jnp.std(jnp.stack(A_matrices), axis=0)

print(f"Mean dynamics norm: {jnp.linalg.norm(A_mean):.3f}")
print(f"Dynamics variability: {jnp.mean(A_std):.3f}")
```

## Example 7: Model Validation

```python
# Cross-validation for model selection
from sklearn.model_selection import KFold

def cross_validate_ctds(Y, state_dims, n_folds=5):
    """Cross-validate CTDS models with different state dimensions."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = {}
    
    for D in state_dims:
        fold_scores = []
        
        for train_idx, test_idx in kf.split(Y.T):  # Split time points
            Y_train = Y[:, train_idx]
            Y_test = Y[:, test_idx]
            
            # Fit model
            model = CTDS(state_dim=D, emission_dim=N)
            params = model.initialize(Y_train)
            params_fitted, _ = model.fit_em(params, Y_train, num_iters=50)
            
            # Evaluate on test set
            test_loglik = model.marginal_log_prob(params_fitted, Y_test.T)
            fold_scores.append(test_loglik)
        
        results[D] = jnp.mean(jnp.array(fold_scores))
    
    return results

# Run cross-validation
state_dims = [5, 10, 15, 20]
cv_results = cross_validate_ctds(Y, state_dims)

best_dim = max(cv_results, key=cv_results.get)
print(f"Best state dimension: {best_dim}")
print("CV results:", cv_results)
```

## Tips and Best Practices

### Data Preprocessing
- Center neural activity: `Y = Y - jnp.mean(Y, axis=1, keepdims=True)`
- Scale variance: `Y = Y / jnp.std(Y, axis=1, keepdims=True)`
- Handle missing data via masking

### Model Selection  
- Use cross-validation for state dimension selection
- Monitor EM convergence via log-likelihood
- Check Dale's law constraint satisfaction

### Performance
- Enable JIT compilation for production use
- Use GPU for large datasets (N > 100, T > 1000)
- Consider batch processing for multiple sequences

### Diagnostics
- Plot EM convergence curves
- Visualize learned connectivity matrices
- Check residual autocorrelations
- Validate forecasting performance
