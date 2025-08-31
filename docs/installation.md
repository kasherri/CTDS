# Installation

## Requirements

- Python 3.8 or higher
- JAX
- Dynamax
- NumPy
- SciPy

## Install from Source

1. Clone the repository:
   ```bash
   git clone https://github.com/kasherri/CTDS.git
   cd CTDS
   ```

2. Install in development mode:
   ```bash
   pip install -e ".[dev,docs,examples]"
   ```

## Install Core Dependencies Only

For basic functionality:
```bash
pip install -e .
```

## Optional Dependencies

- `examples`: Dependencies for running examples and notebooks
- `docs`: Dependencies for building documentation  
- `dev`: Development dependencies including testing and linting tools

## Verification

To verify the installation:
```python
import ctds
print(ctds.__version__)
```
