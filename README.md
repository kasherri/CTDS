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
jax.config.update("jax_enable_x64", True)

from ctds import CTDS, ParamsCTDS
from ctds.initialization import fa_initialize_ctds

# Initialize parameters from data using factor analysis
params = fa_initialize_ctds(Y, cell_types, cell_sign, state_dim=D)

# Fit model with EM
model = CTDS(params)
fitted_params, lls = model.fit_em(Y, num_iters=100)
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

This package was developed as part of a senior thesis. If you use CTDS in your work, please cite:

<!-- TODO: attach thesis PDF and fill in citation details -->
```
Njoroge, N. (2026). [Thesis title]. Senior Thesis. [Institution].
```

Contact: [njerinjoroge9@gmail.com](mailto:njerinjoroge9@gmail.com)

## License

MIT — see [LICENSE](LICENSE).

Copyright (c) 2026 Njeri Njoroge
