# CTDS: Cell-Type Dynamical Systems

CTDS is a Python package for fitting **Cell-Type Dynamical Systems** — a class of constrained linear dynamical system (LDS) models designed for neural population data. It enforces **Dale's Law** (sign constraints on connectivity) and **cell-type block structure** on the emission matrix, using an expectation-maximization (EM) algorithm built on [JAX](https://github.com/google/jax) and [Dynamax](https://dynamax.readthedocs.io).

## Installation

Requires Python 3.10 or 3.11.

```bash
git clone https://github.com/kasherri/ctds.git
cd ctds
pip install -e .
```

All dependencies (including JAX) are installed automatically. For development extras or notebook support:

```bash
pip install -e ".[dev]"        # linting, type checking, testing
pip install -e ".[notebooks]"  # Jupyter + plotting libraries
```

### Verify your installation

```python
import jax
jax.config.update("jax_enable_x64", True)  # required — see note below

import ctds
print(ctds.__version__)  # 0.1.0
```

> **Important — 64-bit precision:** JAX defaults to 32-bit floats; CTDS requires 64-bit.
> Always set `jax.config.update("jax_enable_x64", True)` **before** importing `ctds`.

## Quickstart

```python
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from ctds import CTDS, ParamsCTDS, ParamsCTDSInitial, ParamsCTDSDynamics, ParamsCTDSEmissions
from ctds.initialization import fa_initialize_ctds

# --- 1. Define cell-type structure ---
# K=2 cell types: excitatory (label 0) and inhibitory (label 1)
cell_types      = jnp.array([0, 1])          # (K,)  contiguous labels 0..K-1
cell_sign       = jnp.array([1, -1])         # (K,)  +1 excitatory, -1 inhibitory
cell_type_dims  = jnp.array([D_e, D_i])      # (K,)  latent dims per type; D = D_e + D_i
cell_type_mask  = ...                         # (N,)  int array, cell-type label per neuron
observations=....                             # (B,T,N) array

# Boolean masks selecting neurons of each type (required by fa_initialize_ctds)
e_mask = (cell_type_mask == 0)               # (N,) bool
i_mask = (cell_type_mask == 1)               # (N,) bool

# --- 2. Build model and parameter objects ---
model = CTDS(
    emission_dim=N,
    cell_types=cell_types,
    cell_sign=cell_sign,
    cell_type_dimensions=cell_type_dims,
    cell_type_mask=cell_type_mask,
)

#--- 2. Initialize parameters with NMF ---
init_params=model.initiliaze(observations)

# --- Alternatively, tnitialize parameters with FA ---
# Y : (B, T, N) — B trials, T time steps, N neurons
init = fa_initialize_ctds(
    Y, e_mask, i_mask, D_e + D_i,
    cell_types, cell_sign, cell_type_dims, cell_type_mask,)
D = D_e + D_i
dynamics_mask = jnp.repeat(cell_sign, cell_type_dims)  # (D,) sign per latent dim
params = ParamsCTDS(
    initial=ParamsCTDSInitial(mean=jnp.zeros(D), cov=jnp.eye(D)),
    dynamics=ParamsCTDSDynamics(
        weights=init["A0"], cov=init["Q0"], dynamics_mask=dynamics_mask
    ),
    emissions=ParamsCTDSEmissions(
        weights=init["C0"], cov=jnp.diag(init["R0"]), bias=init["d0"]
    ),
    constraints=model.constraints,
    observations=Y,
)

# --- 3. Fit with EM ---
fitted_params, log_probs = model.fit_em(params, Y, num_iters=100)
```

## Model Overview

| Symbol | Description |
|--------|-------------|
| `A` | Dynamics matrix (Dale's Law constraints per cell type) |
| `C` | Emission matrix (block-diagonal, non-negative per cell type) |
| `Q` | Latent noise covariance |
| `R` | Observation noise covariance (diagonal) |

The latent state evolves as:
```
z_t = A z_{t-1} + q_t,   q_t ~ N(0, Q)
y_t = C z_t + r_t,       r_t ~ N(0, R)
```

Constraints are enforced at each M-step via bounded quadratic programming (via [jaxopt](https://jaxopt.github.io)).

## API Reference

### `CTDS`

The main model class. Inherits from Dynamax `SSM`.

```python
model = CTDS(
    emission_dim: int,               # number of observed neurons N
    cell_types: Array,               # (K,) int, contiguous labels 0..K-1
    cell_sign: Array,                # (K,) int, +1 excitatory / -1 inhibitory
    cell_type_dimensions: Array,     # (K,) int, latent dims per cell type
    cell_type_mask: Array,           # (N,) int, cell-type label per neuron
    inputs_dim: int | None = None,   # optional external input dimension
)
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `fit_em` | `(params, Y, inputs=None, num_iters=100)` → `(ParamsCTDS, log_probs)` | Run EM. `Y` is `(B, T, N)`. Returns fitted params and per-iteration log-likelihoods. |
| `e_step` | `(params, Y, inputs=None)` → `SufficientStats` | Kalman smoother E-step for a single trial `Y (T, N)`. |
| `m_step` | `(params, props, batch_stats, m_step_state)` → `(ParamsCTDS, M_Step_State)` | Constrained M-step (QP updates for A and C). |
| `marginal_log_prob` | `(params, Y, inputs=None)` → `float` | Log marginal likelihood for one trial. |
| `filter` | `(params, Y, inputs=None)` → filtered posterior | Kalman filter (forward pass only). |
| `smoother` | `(params, Y, inputs=None)` → smoothed posterior | Kalman smoother (forward + backward). |
| `sample` | `(params, key, num_timesteps, inputs=None)` → `(z, y)` | Sample latent trajectory `z (T, D)` and observations `y (T, N)`. |
| `initialize` | `(batch_observations)` → `ParamsCTDS` | NMF-based in-class initializer (alternative to `fa_initialize_ctds`). |

---

### Parameter NamedTuples

All parameter objects are immutable `NamedTuple`s, compatible with JAX transformations.

#### `ParamsCTDS`
Top-level container passed to every model method.

| Field | Shape | Description |
|-------|-------|-------------|
| `initial` | `ParamsCTDSInitial` | Initial distribution $p(z_1)$ |
| `dynamics` | `ParamsCTDSDynamics` | Transition model |
| `emissions` | `ParamsCTDSEmissions` | Emission model |
| `constraints` | `ParamsCTDSConstraints` | Cell-type structure (read-only during EM) |
| `observations` | `(B, T, N)` or `None` | Data attached for M-step access |

#### `ParamsCTDSInitial`

| Field | Shape | Description |
|-------|-------|-------------|
| `mean` | `(D,)` | Prior mean $\mu_0$ |
| `cov` | `(D, D)` | Prior covariance $\Sigma_0$ |

#### `ParamsCTDSDynamics`

| Field | Shape | Description |
|-------|-------|-------------|
| `weights` | `(D, D)` | Dynamics matrix $A$ |
| `cov` | `(D, D)` | Process noise covariance $Q$ |
| `dynamics_mask` | `(D,)` | Sign per latent dim (`+1`/`-1`), derived from `cell_sign` |

#### `ParamsCTDSEmissions`

| Field | Shape | Description |
|-------|-------|-------------|
| `weights` | `(N, D)` | Emission matrix $C$ (block-diagonal, non-negative within blocks) |
| `cov` | `(N, N)` | Observation noise covariance $R$ (diagonal in practice) |
| `bias` | `(N,)` | Per-neuron bias $d$ |

#### `ParamsCTDSConstraints`

| Field | Shape | Description |
|-------|-------|-------------|
| `cell_types` | `(K,)` | Cell-type labels (must be `0..K-1`) |
| `cell_sign` | `(K,)` | `+1` excitatory / `-1` inhibitory per type |
| `cell_type_dimensions` | `(K,)` | Number of latent dims per type |
| `cell_type_mask` | `(N,)` | Cell-type label for each neuron |

---

### Initialization

```python
from ctds.initialization import fa_initialize_ctds, pca_initialize_ctds
```

#### `fa_initialize_ctds`

Factor-analysis-based initialization. Runs per-cell-type FA, then fits constrained $C$ and $A$ via projected gradient descent.

```python
init = fa_initialize_ctds(
    Y,                      # (B, T, N)  observed data
    e_mask,                 # (N,) bool  excitatory neurons
    i_mask,                 # (N,) bool  inhibitory neurons
    D,                      # int        total latent dimension
    cell_types,             # (K,)
    cell_sign,              # (K,)
    cell_type_dimensions,   # (K,)
    cell_type_mask,         # (N,)
    key=jr.PRNGKey(0),      # PRNG key
    fa_iters=100,           # EM iterations for FA
    pgd_iters=400,          # PGD iterations for C regression
)
# Returns dict with keys:
#   X0  (B, T, D)  — latent trajectories from FA posterior
#   C0  (N, D)     — constrained emission matrix
#   d0  (N,)       — observation bias
#   R0  (N,)       — diagonal observation noise
#   A0  (D, D)     — Dale-constrained dynamics matrix
#   Q0  (D, D)     — process noise covariance
```

#### `pca_initialize_ctds`

PCA-based alternative to `fa_initialize_ctds`. Same return format.

---

### Simulation utilities

```python
from ctds.simulation_utils import stationary_latent_cov, observation_snr, build_R_targetSNR
```

| Function | Description |
|----------|-------------|
| `stationary_latent_cov(A, Q)` | Solve discrete Lyapunov equation for steady-state latent covariance |
| `observation_snr(A, C, Q, R)` | Compute observation signal-to-noise ratio |
| `build_R_targetSNR(A, C, Q, snr)` | Construct $R = rI$ achieving a target SNR |

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| [notebooks/exp0_benchmark.ipynb](notebooks/exp0_benchmark.ipynb) | Benchmark against standard SSM |
| [notebooks/exp1a_code_correctness.ipynb](notebooks/exp1a_code_correctness.ipynb) | Unit-level correctness checks |
| [notebooks/exp1b_jax_benchmarks.ipynb](notebooks/exp1b_jax_benchmarks.ipynb) | JAX JIT / vmap performance |
| [notebooks/exp2_parameter_recovery.ipynb](notebooks/exp2_parameter_recovery.ipynb) | Parameter recovery experiments |
| [notebooks/exp3_eirnn_equivalence.ipynb](notebooks/exp3_eirnn_equivalence.ipynb) | EI-RNN equivalence |
| [notebooks/exp4_initialization_quality.ipynb](notebooks/exp4_initialization_quality.ipynb) | Initialization quality (NMF vs FA vs random) |
| [notebooks/exp5_weak_identifiability_ctds.ipynb](notebooks/exp5_weak_identifiability_ctds.ipynb) | Weak identifiability analysis |

## Package Structure

```
ctds/
  __init__.py          # Public API
  models.py            # CTDS model class and M-step helpers
  params.py            # Parameter dataclasses (ParamsCTDS and friends)
  inference.py         # DynamaxLGSSMBackend (E-step)
  initialization.py    # fa_initialize_ctds, pca_initialize_ctds
  simulation_utils.py  # Synthetic data generation and evaluation
  utils.py             # Sufficient statistics and shared utilities
  evaluation/
    metrics.py         # Evaluation metrics
tests/                 # pytest test suite
notebooks/             # Experiment notebooks
```

## Citation

This package was developed as part of a senior thesis. 

<!-- TODO: will attach thesis PDF or future paper-->

Contact: [njerinjoroge9@gmail.com](mailto:njerinjoroge9@gmail.com)

## License

MIT — see [LICENSE](LICENSE).

Copyright (c) 2026 Njeri Njoroge
