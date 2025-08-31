
# Cell-Type Dynamical Systems (CTDS)

A JAX implementation of Cell-Type Dynamical Systems from Jha et al. (2024)
for modeling neural population dynamics with biologically-constrained connectivity.


CTDS models neural population activity using linear dynamical systems that respect cell-type constraints and Dale's law. The model decomposes neural connectivity into cell-type-specific factors while maintaining Dale's law (excitatory neurons have non-negative outgoing weights, inhibitory neurons have non-positive outgoing weights).

### Key Features

- **Cell-type structure**: Models connectivity through cell-type-specific factorization
- **Dale's law constraints**: Enforces biologically plausible sign constraints  
- **JAX-native**: Full JAX compatibility for GPU acceleration and JIT compilation
- **EM training**: Efficient parameter estimation via expectation-maximization
- **Dynamax integration**: Uses Dynamax for exact inference in linear Gaussian models

## Mathematical Framework

The CTDS model describes neural dynamics as:

**Latent dynamics:**

$$x_{t+1} = A x_t +B u_t+ w_t, \quad  w_t ~ N(0, Q)$$


**Observations:**  

$$
y_t = C x_t + v_t,  \quad   v_t ~ N(0, R)
$$

**EI RNN Initialization**

TODO



### Installation

#### Quick Setup (Recommended)

Clone the repository and run the setup script:

```bash
git clone https://github.com/kasherri/CTDS.git
cd CTDS
./setup.sh
```

The setup script will:
- Check Python version (3.9+ required)
- Create a virtual environment
- Install all dependencies
- Install CTDS in development mode
- Run basic tests to verify installation

#### Manual Installation

If you prefer manual setup:

```bash
# Clone repository
git clone https://github.com/kasherri/CTDS.git
cd CTDS

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package
pip install -e .

# Install development dependencies (optional)
pip install -e ".[dev,docs,examples]"
```

#### From PyPI (Future Release)

```bash
pip install ctds
```

#### Development Commands

After installation, use the provided Makefile for common tasks:

```bash
make help      # Show all available commands
make demo      # Run quickstart demo
make test      # Run test suite
make docs      # Build and serve documentation
make format    # Format code
make lint      # Lint code
```

#### Requirements

- Python 3.9+
- JAX 0.4.0+
- See `pyproject.toml` for complete dependency list

#### Verify Installation

After installation, verify everything works:

```bash
# Activate environment (if not already active)
source .venv/bin/activate

# Test basic import
python -c "import ctds; print('CTDS imported successfully!')"

# Run test suite
python -m pytest tests/

# Test JAX backend
python -c "import jax; print(f'JAX backend: {jax.default_backend()}')"

# Run quickstart demo
python quickstart.py
```

### Basic Usage

```python
import jax.numpy as jnp
import jax.random as jr
from models import CTDS

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

### Advanced Example

```python
from params import ParamsCTDSConstraints

# Custom cell-type configuration
constraints = ParamsCTDSConstraints(
    cell_type_dimensions=[8, 2],     # 8 E-cells, 2 I-cells  
    cell_sign=jnp.array([1, 1, 1, 1, 1, 1, 1, 1, -1, -1]),  # E/I labels
    num_cell_types=2
)

model = CTDS(
    state_dim=10,
    emission_dim=25,
    constraints=constraints
)

# Fit with convergence monitoring
params = model.initialize(Y)
params, losses = model.fit_em(
    params, Y, 
    num_iters=200,
    tolerance=1e-6,
    verbose=True
)

# Posterior inference
posterior_means, posterior_covs = model.smoother(params, Y)
filtered_means, filtered_covs = model.filter(params, Y)
```

## API Reference

### Core Classes

- **`CTDS`**: Main model class inheriting from Dynamax SSM
- **`ParamsCTDS`**: Parameter container for dynamics, emissions, initial state
- **`ParamsCTDSConstraints`**: Cell-type and Dale's law constraint specification

### Key Methods

- **`initialize(Y)`**: Initialize parameters from data using PCA and NMF
- **`fit_em(params, Y, ...)`**: Fit model parameters via EM algorithm  
- **`sample(params, key, T)`**: Sample synthetic trajectories
- **`forecast(params, Y, steps)`**: Forecast future observations
- **`smoother(params, Y)`**: Compute posterior state estimates
- **`filter(params, Y)`**: Compute causal state estimates

## Model Components

### Parameters (`params.py`)
- `ParamsCTDS`: Complete parameter specification
- `ParamsCTDSDynamics`: Dynamics matrix A, process noise Q
- `ParamsCTDSEmissions`: Emission matrix C, observation noise R  
- `ParamsCTDSInitial`: Initial state distribution
- `ParamsCTDSConstraints`: Cell-type structure and constraints

### Inference (`inference.py`)
- `InferenceBackend`: Abstract interface for inference methods
- `DynamaxLGSSMBackend`: Exact inference via Kalman smoothing

### Utilities (`utlis.py`)
- Constrained optimization solvers (QP, NNLS)
- Dale's law constraint handlers
- Block-wise non-negative matrix factorization
- Sufficient statistics computation

### Models (`models.py`)
- `CTDS`: Main implementation with EM training
- Parameter initialization routines
- Missing data forecasting
- Gauge-fixing for numerical stability

## Data Format

- **Observations**: `Y` with shape `(N, T)` where `N` = neurons, `T` = time steps
- **Time-major**: Internal format uses `(T, D)` for states, `(T, N)` for emissions
- **Missing data**: Supported via masking in emission model

## Constraints

### Dale's Law
- Excitatory cells: non-negative outgoing weights
- Inhibitory cells: non-positive outgoing weights  
- Enforced via constrained optimization in M-step

### Cell-Type Structure
- Factorized connectivity: `A = U V^T`
- Block-wise emission matrix respecting cell types
- Non-negative emission weights preserving Dale signs

## Performance

- **JAX JIT compilation**: Most core functions are JIT-compatible
- **GPU acceleration**: Full JAX GPU support
- **Batched processing**: Supports multiple sequences via batch dimensions
- **Memory efficient**: Iterative EM with convergence monitoring

## Limitations

### Model Assumptions
- **Single region**: Current implementation supports only single-region models (no multi-region connectivity)
- **Linear-Gaussian**: Assumes linear dynamics and Gaussian noise (no nonlinearities)
- **No input dynamics**: While the mathematical framework includes `B u_t` terms, the current implementation does not support exogenous inputs to the dynamics


### Implementation Constraints
- **M-step not JIT-compiled**: The M-step contains constrained optimization routines that are not currently JIT-compatible, making `fit_em` only partially JIT-compiled
- **Solver dependencies**: Relies on specific optimization solvers (BoxOSQP, BoxCDQP) which may have convergence limitations
- **Memory scaling**: Large state dimensions (D > 50) or long sequences (T > 10,000) may encounter memory constraints
- **Initialization sensitivity**: EM convergence can depend on initialization quality, particularly for high-dimensional models


### Computational Limitations
- **Constrained optimization**: QP solvers may fail for ill-conditioned problems
- **Numerical stability**: Large condition numbers in covariance matrices can cause instability


### Future Extensions
Planned improvements to address current limitations:
- Multi-region support with inter-region connectivity
- Nonlinear dynamics via neural network components  
- Input-driven dynamics with `B u_t` implementation
- Fully JIT-compiled M-step with custom solvers
- Time-varying parameters and adaptive models

## Examples

See Jupyter notebooks:
- `ctds_initialization_diagnostics.ipynb`: Parameter initialization
- `ctds_model_validation.ipynb`: Model validation and testing


## License

TODO
## Contributing

TODO
