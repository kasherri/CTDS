"""
Pytest configuration and fixtures.
"""
import pytest
import jax

# Configure JAX for float64 precision
jax.config.update("jax_enable_x64", True)


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
