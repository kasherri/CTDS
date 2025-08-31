# Contributing

We welcome contributions to the CTDS project!

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/kasherri/CTDS.git
   cd CTDS
   ```

2. Install in development mode:
   ```bash
   pip install -e ".[dev,docs,examples]"
   ```

3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Code Style

We use:
- **Black** for code formatting
- **Ruff** for linting
- **mypy** for type checking

Run all checks:
```bash
pre-commit run --all-files
```

## Testing

Run the test suite:
```bash
pytest
```

## Documentation

Build documentation locally:
```bash
mkdocs serve
```

## Pull Requests

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Ensure all checks pass
6. Submit a pull request

## Issues

Please use GitHub Issues to report bugs or request features.
