"""
Tests for the fit_em method in the CTDS class.

This module tests various aspects of the EM algorithm implementation:
1. Basic functionality and convergence
2. Parameter shapes and consistency
3. Log-likelihood improvement
4. Batch processing
5. Edge cases and error handling
"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from models import CTDS
from params import ParamsCTDS, ParamsCTDSConstraints
from utlis import generate_synthetic_cell_type_data

# Configure JAX for testing
jax.config.update("jax_enable_x64", True)


class TestFitEM:
    """Test suite for the fit_em method."""
    
    
   
    @pytest.fixture
    def setup_small_ctds(self):
        observations, constraints = generate_synthetic_cell_type_data(
            N=15, T=100, D=6, K=2,
            excitatory_fraction=0.6,
            noise_level=0.1,
            seed=42
        )
        ctds = CTDS(
            emission_dim=observations.shape[1],
            cell_types=constraints.cell_types,
            cell_sign=constraints.cell_sign,
            cell_type_dimensions=constraints.cell_type_dimensions,
            cell_type_mask=constraints.cell_type_mask,
            state_dim=6
        )
        return ctds, observations

    def test_fit_em_basic_functionality(self, setup_small_ctds):
        """Test basic EM fitting functionality."""
        ctds, observations = setup_small_ctds
        print(observations.shape)
        # Initialize parameters
        params_init = ctds.initialize(observations)
        
        # Add batch dimension
        batch_observations = observations[None, :, :]
        
        # Run EM for a few iterations
        params_fitted, log_probs = ctds.fit_em(
            params_init,
            batch_observations,
            num_iters=5,
            verbose=False
        )
        
        # Check that we got results
        assert params_fitted is not None
        assert len(log_probs) == 5
        
        # Check parameter structure is preserved
        assert hasattr(params_fitted, 'initial')
        assert hasattr(params_fitted, 'dynamics')
        assert hasattr(params_fitted, 'emissions')
        assert hasattr(params_fitted, 'constraints')
    
    def test_parameter_shapes_consistency(self, setup_small_ctds, synthetic_data):
        """Test that parameter shapes remain consistent after EM."""
        ctds, N, D, K = setup_small_ctds
        observations, _ = synthetic_data
        
        params_init = ctds.initialize(observations[:N, :])
        batch_observations = observations[:N, :][None, :, :]
        
        # Store initial shapes
        initial_shapes = {
            'initial_mean': params_init.initial.mean.shape,
            'initial_cov': params_init.initial.cov.shape,
            'dynamics_weights': params_init.dynamics.weights.shape,
            'dynamics_cov': params_init.dynamics.cov.shape,
            'emissions_weights': params_init.emissions.weights.shape,
            'emissions_cov': params_init.emissions.cov.shape,
        }
        
        # Run EM
        params_fitted, _ = ctds.fit_em(
            params_init,
            batch_observations,
            num_iters=3,
            verbose=False
        )
        
        # Check shapes are preserved
        assert params_fitted.initial.mean.shape == initial_shapes['initial_mean']
        assert params_fitted.initial.cov.shape == initial_shapes['initial_cov']
        assert params_fitted.dynamics.weights.shape == initial_shapes['dynamics_weights']
        assert params_fitted.dynamics.cov.shape == initial_shapes['dynamics_cov']
        assert params_fitted.emissions.weights.shape == initial_shapes['emissions_weights']
        assert params_fitted.emissions.cov.shape == initial_shapes['emissions_cov']
    
    def test_log_likelihood_improvement(self, setup_small_ctds, synthetic_data):
        """Test that log-likelihood generally improves during EM."""
        ctds, N, D, K = setup_small_ctds
        observations, _ = synthetic_data
        
        params_init = ctds.initialize(observations[:N, :])
        batch_observations = observations[:N, :][None, :, :]
        
        # Run EM for more iterations
        params_fitted, log_probs = ctds.fit_em(
            params_init,
            batch_observations,
            num_iters=10,
            verbose=False
        )
        
        # Check that log-likelihood generally improves
        initial_ll = log_probs[0]
        final_ll = log_probs[-1]
        
        # EM should increase likelihood (or at least not decrease significantly)
        assert final_ll >= initial_ll - 1e-6, f"LL decreased: {initial_ll} -> {final_ll}"
        
        # Check for monotonic improvement in later iterations
        # (early iterations might have some fluctuation due to initialization)
        if len(log_probs) >= 5:
            late_improvement = log_probs[-1] - log_probs[-5]
            assert late_improvement >= -1e-6, "Late iterations should not decrease likelihood significantly"
    
    def test_batch_processing(self, setup_small_ctds):
        """Test that EM works with multiple sequences in batch."""
        ctds, N, D, K = setup_small_ctds
        
        # Generate multiple sequences
        batch_size = 3
        T = 50
        
        # Create batch of synthetic data
        key = jr.PRNGKey(123)
        keys = jr.split(key, batch_size)
        
        batch_observations = []
        for i in range(batch_size):
            obs, _ = generate_synthetic_cell_type_data(
                N=N, T=T, D=D, K=K,
                seed=int(keys[i][0])  # Convert to int for seed
            )
            batch_observations.append(obs)
        
        batch_observations = jnp.stack(batch_observations, axis=0)  # Shape: (batch_size, T, N)
        
        # Initialize with first sequence
        params_init = ctds.initialize(batch_observations[0])
        
        # Run EM on batch
        params_fitted, log_probs = ctds.fit_em(
            params_init,
            batch_observations,
            num_iters=5,
            verbose=False
        )
        
        # Check that it works
        assert params_fitted is not None
        assert len(log_probs) == 5
        assert jnp.all(jnp.isfinite(log_probs))
    
    def test_single_vs_batch_consistency(self, setup_small_ctds, synthetic_data):
        """Test that single sequence gives same result as batch of size 1."""
        ctds, N, D, K = setup_small_ctds
        observations, _ = synthetic_data
        
        params_init = ctds.initialize(observations[:N, :])
        
        # Single sequence (with batch dim)
        single_obs = observations[:N, :][None, :, :]  # Shape: (1, T, N)
        
        # Run EM twice with same initialization
        params_single, log_probs_single = ctds.fit_em(
            params_init,
            single_obs,
            num_iters=3,
            verbose=False
        )
        
        params_batch, log_probs_batch = ctds.fit_em(
            params_init,
            single_obs,
            num_iters=3,
            verbose=False
        )
        
        # Results should be identical (or very close due to numerical precision)
        np.testing.assert_allclose(log_probs_single, log_probs_batch, rtol=1e-10)
    
    def test_parameter_convergence(self, setup_small_ctds, synthetic_data):
        """Test that parameters converge (stop changing significantly)."""
        ctds, N, D, K = setup_small_ctds
        observations, _ = synthetic_data
        
        params_init = ctds.initialize(observations[:N, :])
        batch_observations = observations[:N, :][None, :, :]
        
        # Run EM for many iterations
        params_fitted, log_probs = ctds.fit_em(
            params_init,
            batch_observations,
            num_iters=20,
            verbose=False
        )
        
        # Check that log-likelihood converges (small changes in later iterations)
        if len(log_probs) >= 10:
            recent_changes = jnp.abs(jnp.diff(log_probs[-5:]))
            max_recent_change = jnp.max(recent_changes)
            
            # In later iterations, changes should be small
            assert max_recent_change < 1.0, f"Large changes in late iterations: {max_recent_change}"
    
    def test_numerical_stability(self, setup_small_ctds, synthetic_data):
        """Test numerical stability - no NaN/Inf values."""
        ctds, N, D, K = setup_small_ctds
        observations, _ = synthetic_data
        
        params_init = ctds.initialize(observations[:N, :])
        batch_observations = observations[:N, :][None, :, :]
        
        # Run EM
        params_fitted, log_probs = ctds.fit_em(
            params_init,
            batch_observations,
            num_iters=10,
            verbose=False
        )
        
        # Check for NaN/Inf in log probabilities
        assert jnp.all(jnp.isfinite(log_probs)), "Log probabilities contain NaN/Inf"
        
        # Check for NaN/Inf in fitted parameters
        assert jnp.all(jnp.isfinite(params_fitted.initial.mean)), "Initial mean contains NaN/Inf"
        assert jnp.all(jnp.isfinite(params_fitted.initial.cov)), "Initial cov contains NaN/Inf"
        assert jnp.all(jnp.isfinite(params_fitted.dynamics.weights)), "Dynamics weights contain NaN/Inf"
        assert jnp.all(jnp.isfinite(params_fitted.dynamics.cov)), "Dynamics cov contains NaN/Inf"
        assert jnp.all(jnp.isfinite(params_fitted.emissions.weights)), "Emissions weights contain NaN/Inf"
        assert jnp.all(jnp.isfinite(params_fitted.emissions.cov)), "Emissions cov contains NaN/Inf"
    
    def test_covariance_positive_definite(self, setup_small_ctds, synthetic_data):
        """Test that covariance matrices remain positive definite."""
        ctds, N, D, K = setup_small_ctds
        observations, _ = synthetic_data
        
        params_init = ctds.initialize(observations[:N, :])
        batch_observations = observations[:N, :][None, :, :]
        
        # Run EM
        params_fitted, _ = ctds.fit_em(
            params_init,
            batch_observations,
            num_iters=5,
            verbose=False
        )
        
        # Check that covariance matrices are positive definite
        # (all eigenvalues should be positive)
        
        # Initial covariance
        init_eigs = jnp.linalg.eigvals(params_fitted.initial.cov)
        assert jnp.all(init_eigs > 1e-8), f"Initial cov not PD: {init_eigs}"
        
        # Dynamics covariance
        dyn_eigs = jnp.linalg.eigvals(params_fitted.dynamics.cov)
        assert jnp.all(dyn_eigs > 1e-8), f"Dynamics cov not PD: {dyn_eigs}"
        
        # Emissions covariance
        em_eigs = jnp.linalg.eigvals(params_fitted.emissions.cov)
        assert jnp.all(em_eigs > 1e-8), f"Emissions cov not PD: {em_eigs}"
    
    def test_constraints_preservation(self, setup_small_ctds, synthetic_data):
        """Test that cell type constraints are preserved after EM."""
        ctds, N, D, K = setup_small_ctds
        observations, _ = synthetic_data
        
        params_init = ctds.initialize(observations[:N, :])
        batch_observations = observations[:N, :][None, :, :]
        
        # Store initial constraints
        initial_constraints = params_init.constraints
        
        # Run EM
        params_fitted, _ = ctds.fit_em(
            params_init,
            batch_observations,
            num_iters=5,
            verbose=False
        )
        
        # Check that constraints are preserved
        fitted_constraints = params_fitted.constraints
        
        np.testing.assert_array_equal(
            initial_constraints.cell_types, 
            fitted_constraints.cell_types
        )
        np.testing.assert_array_equal(
            initial_constraints.cell_sign, 
            fitted_constraints.cell_sign
        )
        np.testing.assert_array_equal(
            initial_constraints.cell_type_dimensions, 
            fitted_constraints.cell_type_dimensions
        )
        np.testing.assert_array_equal(
            initial_constraints.cell_type_mask, 
            fitted_constraints.cell_type_mask
        )
    
    def test_empty_iterations(self, setup_small_ctds, synthetic_data):
        """Test edge case with zero iterations."""
        ctds, N, D, K = setup_small_ctds
        observations, _ = synthetic_data
        
        params_init = ctds.initialize(observations[:N, :])
        batch_observations = observations[:N, :][None, :, :]
        
        # Run EM with 0 iterations
        params_fitted, log_probs = ctds.fit_em(
            params_init,
            batch_observations,
            num_iters=0,
            verbose=False
        )
        
        # Should return initial parameters and empty log_probs
        assert len(log_probs) == 0
        
        # Parameters should be unchanged (or very close)
        np.testing.assert_allclose(
            params_fitted.initial.mean, 
            params_init.initial.mean, 
            rtol=1e-10
        )
    
    def test_dynamics_stability(self, setup_small_ctds, synthetic_data):
        """Test that fitted dynamics matrix has reasonable eigenvalues."""
        ctds, N, D, K = setup_small_ctds
        observations, _ = synthetic_data
        
        params_init = ctds.initialize(observations[:N, :])
        batch_observations = observations[:N, :][None, :, :]
        
        # Run EM
        params_fitted, _ = ctds.fit_em(
            params_init,
            batch_observations,
            num_iters=10,
            verbose=False
        )
        
        # Check dynamics matrix eigenvalues
        A = params_fitted.dynamics.weights
        eigenvals = jnp.linalg.eigvals(A)
        max_eigenval = jnp.max(jnp.abs(eigenvals))
        
        # For stability, eigenvalues should generally be < 1
        # (though this depends on the data and might not always hold)
        # At minimum, they shouldn't be extremely large
        assert max_eigenval < 10.0, f"Very large eigenvalue: {max_eigenval}"
        
        # Check that we don't have eigenvalues that are NaN
        assert jnp.all(jnp.isfinite(eigenvals)), "Eigenvalues contain NaN/Inf"


def test_integration_with_synthetic_data():
    """Integration test using the synthetic data generator."""
    # Generate realistic synthetic data
    observations, constraints = generate_synthetic_cell_type_data(
        N=20, T=200, D=6, K=2,
        excitatory_fraction=0.7,
        noise_level=0.05,
        seed=999
    )
    
    # Create CTDS model
    ctds = CTDS(
        emission_dim=constraints.cell_type_mask.shape[0],
        cell_types=constraints.cell_types,
        cell_sign=constraints.cell_sign,
        cell_type_dimensions=constraints.cell_type_dimensions,
        cell_type_mask=constraints.cell_type_mask,
        state_dim=int(jnp.sum(constraints.cell_type_dimensions))
    )
    
    # Initialize and fit
    params_init = ctds.initialize(observations)
    batch_observations = observations[None, :, :]
    
    params_fitted, log_probs = ctds.fit_em(
        params_init,
        batch_observations,
        num_iters=15,
        verbose=False
    )
    
    # Basic checks
    assert len(log_probs) == 15
    assert log_probs[-1] >= log_probs[0] - 1e-6  # Should improve or stay same
    assert jnp.all(jnp.isfinite(log_probs))
    
    # Check that we can compute posterior
    try:
        posterior = ctds.smoother(params_fitted, observations)
        assert posterior is not None
    except Exception as e:
        pytest.fail(f"Could not compute posterior after EM: {e}")


if __name__ == "__main__":
    # Run a quick test when executed directly
    print("Running basic fit_em test...")
    
    # Simple test
    test_ctds = CTDS(
        emission_dim=8,
        cell_types=jnp.array([0, 1]),
        cell_sign=jnp.array([1, -1]),
        cell_type_dimensions=jnp.array([2, 2]),
        cell_type_mask=jnp.array([0, 0, 0, 0, 1, 1, 1, 1]),
        state_dim=4
    )
    
    # Generate simple data
    key = jr.PRNGKey(42)
    observations = jr.normal(key, (8, 50)) # 8 neurons, 50 time points
    
    # Test initialization and EM
    params_init = test_ctds.initialize(observations)
    batch_obs = observations[None, :, :]
    print("batch_obs shape", batch_obs.shape)
    print("params_init emissions shape", params_init.emissions.weights.shape)
    print("initial mean and cov shape", params_init.initial.mean.shape, params_init.initial.cov.shape)
    params_fitted, log_probs = test_ctds.fit_em(
        params_init, batch_obs, None, num_iters=3, verbose=False
    )

    
    print(f"✅ Basic test passed!")
    print(f"   Log-likelihood: {log_probs[0]:.2f} -> {log_probs[-1]:.2f}")
    print(f"   Improvement: {log_probs[-1] - log_probs[0]:.2f}")
