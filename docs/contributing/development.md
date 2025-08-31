# Development Setup

## Setting up a Development Environment

### Clone and Setup

```bash
git clone https://github.com/kasherri/CTDS.git
cd CTDS

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev,docs,examples]"
```

### Pre-commit Hooks

Install pre-commit hooks to ensure code quality:

```bash
pre-commit install
```

This will run linting and formatting checks before each commit.

## Code Style

We use several tools to maintain code quality:

### Black (Code Formatting)
```bash
# Format all code
black src/ tests/

# Check formatting without changes
black --check src/ tests/
```

### Ruff (Linting)
```bash
# Lint all code
ruff check src/ tests/

# Auto-fix issues where possible
ruff check --fix src/ tests/
```

### MyPy (Type Checking)
```bash
# Type check the source code
mypy src/
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/ctds

# Run specific test file
pytest tests/test_models.py

# Run with verbose output
pytest -v
```

### Test Structure

Tests are organized by module:
- `tests/test_models.py`: Model class tests
- `tests/test_params.py`: Parameter structure tests  
- `tests/test_inference.py`: Inference backend tests
- `tests/test_utils.py`: Utility function tests

## Documentation

### Building Docs Locally

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build and serve locally
mkdocs serve

# Build static site
mkdocs build
```

The documentation will be available at http://localhost:8000

### Writing Documentation

- Use NumPy-style docstrings for all functions and classes
- Include mathematical equations using LaTeX syntax
- Add examples to docstrings where helpful
- Update the user guide for major features

## Contributing Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Write code following the style guidelines
- Add tests for new functionality  
- Update documentation as needed
- Ensure all tests pass

### 3. Commit Changes

```bash
# Pre-commit hooks will run automatically
git add .
git commit -m "Add feature: descriptive commit message"
```

### 4. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Release Process

### Version Bumping

1. Update version in `pyproject.toml`
2. Update `src/ctds/__init__.py`
3. Update `CHANGELOG.md`
4. Create git tag: `git tag v0.x.x`
5. Push: `git push origin main --tags`

### PyPI Release

```bash
# Build distributions
python -m build

# Upload to PyPI (requires credentials)
twine upload dist/*
```

## Project Structure

```
CTDS/
├── src/ctds/           # Main package
│   ├── __init__.py     # Package initialization
│   ├── models.py       # CTDS model implementation
│   ├── params.py       # Parameter structures
│   ├── inference.py    # Inference backends
│   └── utils.py        # Utility functions
├── tests/              # Test suite
├── docs/               # Documentation source
├── examples/           # Example notebooks
├── .github/workflows/  # CI/CD configuration
├── pyproject.toml      # Package configuration
└── mkdocs.yml         # Documentation configuration
```
