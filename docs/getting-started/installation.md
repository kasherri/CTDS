# Installation

## Prerequisites

CTDS requires Python 3.9 or later and is built on JAX for high-performance computing.

## Install from PyPI (when released)

```bash
pip install ctds
```

## Install from Source

### Clone the Repository

```bash
git clone https://github.com/kasherri/CTDS.git
cd CTDS
```

### Create Environment

We recommend using conda or virtualenv:

```bash
# Using conda
conda create -n ctds python=3.10
conda activate ctds

# Or using virtualenv
python -m venv ctds-env
source ctds-env/bin/activate  # On Windows: ctds-env\Scripts\activate
```

### Install Dependencies

For basic usage:

```bash
pip install -e .
```

For development (includes testing and documentation tools):

```bash
pip install -e ".[dev,docs]"
```

For examples (includes matplotlib, jupyter):

```bash
pip install -e ".[examples]"
```

For everything:

```bash
pip install -e ".[all]"
```

## Verify Installation

Test that the installation works:

```python
import ctds
print(f"CTDS version: {ctds.__version__}")

# Create a simple model
model = ctds.CTDS(state_dim=5, emission_dim=10)
print("Installation successful!")
```

## GPU Support

For GPU acceleration, install JAX with CUDA support:

```bash
# For CUDA 12
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# For CUDA 11
pip install -U "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Verify GPU is available:

```python
import jax
print(f"JAX devices: {jax.devices()}")
print(f"GPU available: {len(jax.devices('gpu')) > 0}")
```

## Troubleshooting

### Common Issues

**ImportError: No module named 'dynamax'**
```bash
pip install dynamax
```

**JAX installation issues**
- See the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html)
- For M1/M2 Macs, ensure you're using the correct JAX build

**Optimization solver issues**
```bash
pip install --upgrade jaxopt
```

### Performance Tips

- Use JAX's JIT compilation for production code
- Enable 64-bit precision if needed: `jax.config.update("jax_enable_x64", True)`
- For large models, consider using GPU acceleration
