import jax
import jax.numpy as jnp
import pytest
from params import (
    ParamsCTDS, ParamsCTDSInitial, ParamsCTDSDynamics, ParamsCTDSEmissions, 
    ParamsCTDSConstraints, SufficientStats
)
from models import CTDS
from inference import DynamaxLGSSMBackend
from dynamax.linear_gaussian_ssm import ParamsLGSSM

class TestParamsCTDS:
    """Test the ParamsCTDS class and its methods."""
    
    def create_test_ctds_params(self):
        """Create test CTDS parameters."""
        state_dim = 6  # 3 excitatory + 3 inhibitory
        emission_dim = 15
        
        # Initial distribution
        initial = ParamsCTDSInitial(
            mean=jnp.zeros(state_dim),
            cov=jnp.eye(state_dim) * 0.1
        )
        
        # Dynamics
        A = jnp.array([
            [0.8, 0.1, 0.1, -0.2, -0.2, -0.2],
            [0.1, 0.8, 0.1, -0.2, -0.2, -0.2],
            [0.1, 0.1, 0.8, -0.2, -0.2, -0.2],
            [0.3, 0.3, 0.3, -0.6, -0.1, -0.1],
            [0.3, 0.3, 0.3, -0.1, -0.6, -0.1],
            [0.3, 0.3, 0.3, -0.1, -0.1, -0.6],
        ])
        Q = jnp.eye(state_dim) * 0.01
        dynamics = ParamsCTDSDynamics(
            weights=A, 
            cov=Q
        )
        
        # Emissions
        C = jax.random.normal(jax.random.PRNGKey(0), (emission_dim, state_dim)) * 0.5
        C = C.at[:, :3].set(jnp.abs(C[:, :3]))  # excitatory weights positive
        C = C.at[:, 3:].set(-jnp.abs(C[:, 3:]))  # inhibitory weights negative
        R = jnp.eye(emission_dim) * 0.1
        emissions = ParamsCTDSEmissions(weights=C, cov=R)
        
        # Cell constraints
        cell_types = jnp.array([0, 1])  # excitatory and inhibitory
        cell_sign = jnp.array([1, -1])
        cell_type_dimensions = jnp.array([3, 3])
        cell_type_mask = jnp.array([0, 0, 0, 1, 1, 1])  # first 3 excitatory, last 3 inhibitory

        constraints = ParamsCTDSConstraints(
            cell_types=cell_types,
            cell_sign=cell_sign,
            cell_type_dimensions=cell_type_dimensions,
            cell_type_mask=cell_type_mask,
        )
        
        return ParamsCTDS(
            initial=initial,
            dynamics=dynamics,
            emissions=emissions,
            constraints=constraints,
            observations=None  # No observed data for this test
        )
    
    def test_params_ctds_creation(self):
        """Test ParamsCTDS creation and basic properties."""
        params = self.create_test_ctds_params()
        
        # Check that all components are present
        assert hasattr(params, 'initial')
        assert hasattr(params, 'dynamics')
        assert hasattr(params, 'emissions')
        assert hasattr(params, 'constraints')
        
        # Check shapes
        assert params.initial.mean.shape == (6,)
        assert params.initial.cov.shape == (6, 6)
        assert params.dynamics.weights.shape == (6, 6)
        assert params.dynamics.cov.shape == (6, 6)
        assert params.emissions.weights.shape == (15, 6)
        assert params.emissions.cov.shape == (15, 15)

        
        
        print("✓ ParamsCTDS creation works correctly")
    
    def test_to_lgssm_conversion(self):
        """Test conversion from ParamsCTDS to ParamsLGSSM."""
        params = self.create_test_ctds_params()
        
        # Convert to LGSSM format
        lgssm_params = params.to_lgssm()
        
        # Check that it's a ParamsLGSSM object
        assert isinstance(lgssm_params, ParamsLGSSM)
        
        # Check initial distribution conversion
        assert jnp.allclose(params.initial.mean, lgssm_params.initial.mean)
        assert jnp.allclose(params.initial.cov, lgssm_params.initial.cov)
        
        # Check dynamics conversion
        assert jnp.allclose(params.dynamics.weights, lgssm_params.dynamics.weights)
        assert jnp.allclose(params.dynamics.cov, lgssm_params.dynamics.cov)
        assert jnp.allclose(lgssm_params.dynamics.bias, jnp.zeros(params.dynamics.weights.shape[0]))
        
        # Check emissions conversion
        assert jnp.allclose(params.emissions.weights, lgssm_params.emissions.weights)
        assert jnp.allclose(params.emissions.cov, lgssm_params.emissions.cov)
        assert jnp.allclose(lgssm_params.emissions.bias, jnp.zeros(params.emissions.weights.shape[0]))
        
        # Check that input_weights is None for emissions (as expected)
        assert lgssm_params.emissions.input_weights is None
        
        print("✓ ParamsCTDS.to_lgssm() conversion works correctly")
    
    def test_to_lgssm_preserves_structure(self):
        """Test that to_lgssm preserves the mathematical structure."""
        params = self.create_test_ctds_params()
        lgssm_params = params.to_lgssm()
        
        # Test that the dynamics matrix is preserved
        assert jnp.allclose(params.dynamics.weights, lgssm_params.dynamics.weights)
        
        # Test that the emission matrix is preserved
        assert jnp.allclose(params.emissions.weights, lgssm_params.emissions.weights)
        
        # Test that covariances are preserved
        assert jnp.allclose(params.dynamics.cov, lgssm_params.dynamics.cov)
        assert jnp.allclose(params.emissions.cov, lgssm_params.emissions.cov)
        
        # Test that initial distribution is preserved
        assert jnp.allclose(params.initial.mean, lgssm_params.initial.mean)
        assert jnp.allclose(params.initial.cov, lgssm_params.initial.cov)
        
        print("✓ to_lgssm preserves mathematical structure")
    
    def test_to_lgssm_different_sizes(self):
        """Test to_lgssm with different parameter sizes."""
        # Test with different state and emission dimensions
        state_dim = 4
        emission_dim = 8
        
        initial = ParamsCTDSInitial(
            mean=jnp.zeros(state_dim),
            cov=jnp.eye(state_dim) * 0.1
        )
        
        A = jnp.eye(state_dim) * 0.9
        Q = jnp.eye(state_dim) * 0.01
        dynamics = ParamsCTDSDynamics(
            weights=A, 
            cov=Q, 
        )
        
        C = jax.random.normal(jax.random.PRNGKey(1), (emission_dim, state_dim))
        R = jnp.eye(emission_dim) * 0.1
        emissions = ParamsCTDSEmissions(weights=C, cov=R)
        
        constraints = ParamsCTDSConstraints(
            cell_types=jnp.array([0, 1]),
            cell_sign=jnp.array([1, -1]),
            cell_type_dimensions=jnp.array([2, 2]),
            cell_type_mask=jnp.array([0, 0, 1, 1]),
        )

        params = ParamsCTDS(
            initial=initial,
            dynamics=dynamics,
            emissions=emissions,
            constraints=constraints,
            observations=None  # No observed data for this test
        )
        
        lgssm_params = params.to_lgssm()
        
        # Check shapes are correct
        assert lgssm_params.initial.mean.shape == (state_dim,)
        assert lgssm_params.initial.cov.shape == (state_dim, state_dim)
        assert lgssm_params.dynamics.weights.shape == (state_dim, state_dim)
        assert lgssm_params.dynamics.cov.shape == (state_dim, state_dim)
        assert lgssm_params.emissions.weights.shape == (emission_dim, state_dim)
        assert lgssm_params.emissions.cov.shape == (emission_dim, emission_dim)
        
        print("✓ to_lgssm works with different parameter sizes")


class TestCTDSEStep:
    """Test the E-step of CTDS."""
    
    def create_test_ctds_model(self):
        """Create a test CTDS model."""
        state_dim = 4
        emission_dim = 10
        
        # Create parameters
        initial = ParamsCTDSInitial(
            mean=jnp.zeros(state_dim),
            cov=jnp.eye(state_dim) * 0.1
        )
        
        A = jnp.array([
            [0.9, 0.1, -0.2, -0.2],
            [0.1, 0.9, -0.2, -0.2],
            [0.3, 0.3, -0.7, -0.1],
            [0.3, 0.3, -0.1, -0.7],
        ])
        Q = jnp.eye(state_dim) * 0.01
        dynamics = ParamsCTDSDynamics(
            weights=A, 
            cov=Q
        )
        
        C = jax.random.normal(jax.random.PRNGKey(0), (emission_dim, state_dim)) * 0.5
        C = C.at[:, :2].set(jnp.abs(C[:, :2]))
        C = C.at[:, 2:].set(-jnp.abs(C[:, 2:]))
        R = jnp.eye(emission_dim) * 0.1
        emissions = ParamsCTDSEmissions(weights=C, cov=R)

        constraints = ParamsCTDSConstraints(
            cell_types=jnp.array([0, 1]),
            cell_sign=jnp.array([1, -1]),
            cell_type_dimensions=jnp.array([2, 2]),
            cell_type_mask=jnp.array([0, 0, 1, 1]),
        )
        
        params = ParamsCTDS(
            initial=initial,
            dynamics=dynamics,
            emissions=emissions,
            constraints=constraints,
            observations=None  # No observed data for this test
        )
        
        # Create backend
        backend = DynamaxLGSSMBackend()
        
        return params, backend
    
    def test_e_step_basic(self):
        """Test basic E-step functionality."""
        params, backend = self.create_test_ctds_model()
        
        # Generate test data
        T = 20
        observations = jax.random.normal(jax.random.PRNGKey(1), (T, params.emissions.weights.shape[0]))
        
        # Run E-step
        stats, ll = backend.e_step(params, observations)
        
        # Check that stats is a SufficientStats object
        assert isinstance(stats, SufficientStats)
        
        # Check shapes
        assert stats.latent_mean.shape == (T, params.dynamics.weights.shape[0])
        assert stats.latent_second_moment.shape == (T, params.dynamics.weights.shape[0], params.dynamics.weights.shape[0])
        assert stats.cross_time_moment.shape == (T-1, params.dynamics.weights.shape[0], params.dynamics.weights.shape[0])

        assert isinstance(stats.loglik, jnp.ndarray)
        assert stats.T == T
        
        print("✓ E-step produces correct SufficientStats structure")
    
    def test_e_step_with_inputs(self):
        """Test E-step with control inputs."""
        params, backend = self.create_test_ctds_model()
        
        # Generate test data with inputs
        T = 15
        observations = jax.random.normal(jax.random.PRNGKey(2), (T, params.emissions.weights.shape[0]))
        inputs = jax.random.normal(jax.random.PRNGKey(3), (T, 2))  # 2-dimensional inputs
        
        # Run E-step with inputs
        stats,ll = backend.e_step(params, observations, inputs)
        
        # Check that stats structure is correct
        assert isinstance(stats, SufficientStats)
        assert stats.latent_mean.shape == (T, params.dynamics.weights.shape[0])
        assert stats.latent_second_moment.shape == (T, params.dynamics.weights.shape[0], params.dynamics.weights.shape[0])
        assert stats.cross_time_moment.shape == (T-1, params.dynamics.weights.shape[0], params.dynamics.weights.shape[0])
        
        print("✓ E-step works with control inputs")
    
    def test_e_step_consistency(self):
        """Test that E-step produces consistent results."""
        params, backend = self.create_test_ctds_model()
        
        # Generate test data
        T = 25
        observations = jax.random.normal(jax.random.PRNGKey(4), (T, params.emissions.weights.shape[0]))
        
        # Run E-step multiple times
        stats1,ll1 = backend.e_step(params, observations)
        stats2, ll2 = backend.e_step(params, observations)
        
        # Check that results are consistent
        assert jnp.allclose(stats1.latent_mean, stats2.latent_mean)
        assert jnp.allclose(stats1.latent_second_moment, stats2.latent_second_moment)
        assert jnp.allclose(stats1.cross_time_moment, stats2.cross_time_moment)
        assert jnp.isclose(stats1.loglik, stats2.loglik)
        
        print("✓ E-step produces consistent results")
    
    def test_e_step_mathematical_properties(self):
        """Test mathematical properties of E-step results."""
        params, backend = self.create_test_ctds_model()
        
        # Generate test data
        T = 30
        observations = jax.random.normal(jax.random.PRNGKey(5), (T, params.emissions.weights.shape[0]))
        
        # Run E-step
        stats , ll= backend.e_step(params, observations)
        
        # Check that second moments are positive semi-definite
        for t in range(T):
            eigenvals = jnp.linalg.eigvals(stats.latent_second_moment[t])
            assert jnp.all(eigenvals >= -1e-10), f"Second moment at time {t} is not PSD"
        
        # Check that cross-time moments have reasonable magnitude
        cross_moment_norm = jnp.linalg.norm(stats.cross_time_moment)
        assert cross_moment_norm < 1e6, "Cross-time moments have unreasonable magnitude"
        
        # Check that log-likelihood is finite
        assert jnp.isfinite(stats.loglik), "Log-likelihood is not finite"
        
        print("✓ E-step results have correct mathematical properties")


class TestDynamaxLGSSMBackend:
    """Test the DynamaxLGSSMBackend class."""
    
    def create_test_backend(self):
        """Create a test backend."""
        return DynamaxLGSSMBackend()
    
    def create_test_params(self):
        """Create test CTDS parameters."""
        state_dim = 4
        emission_dim = 8
        
        initial = ParamsCTDSInitial(
            mean=jnp.zeros(state_dim),
            cov=jnp.eye(state_dim) * 0.1
        )
        
        A = jnp.eye(state_dim) * 0.9
        Q = jnp.eye(state_dim) * 0.01
        dynamics = ParamsCTDSDynamics(
            weights=A, 
            cov=Q
        )
        
        C = jax.random.normal(jax.random.PRNGKey(0), (emission_dim, state_dim))
        R = jnp.eye(emission_dim) * 0.1
        emissions = ParamsCTDSEmissions(weights=C, cov=R)

        constraints = ParamsCTDSConstraints(
            cell_types=jnp.array([0, 1]),
            cell_sign=jnp.array([1, -1]),
            cell_type_dimensions=jnp.array([2, 2]),
            cell_type_mask=jnp.array([0, 0, 1, 1]),
        )

        return ParamsCTDS(
            initial=initial,
            dynamics=dynamics,
            emissions=emissions,
            constraints=constraints,
            observations=None  # No observed data for this test
        )
    
    def test_backend_creation(self):
        """Test backend creation."""
        backend = self.create_test_backend()
        assert isinstance(backend, DynamaxLGSSMBackend)
        print("✓ DynamaxLGSSMBackend creation works")
    
    def test_backend_filter(self):
        """Test backend filter method."""
        backend = self.create_test_backend()
        params = self.create_test_params()
        
        # Generate test data
        T = 20
        observations = jax.random.normal(jax.random.PRNGKey(1), (T, params.emissions.weights.shape[0]))
        
        # Run filter
        filtered_means, filtered_covs = backend.filter(params, observations)
        
        # Check shapes
        assert filtered_means.shape == (T, params.dynamics.weights.shape[0])
        assert filtered_covs.shape == (T, params.dynamics.weights.shape[0], params.dynamics.weights.shape[0])
        
        print("✓ Backend filter method works")
    
    def test_backend_smoother(self):
        """Test backend smoother method."""
        backend = self.create_test_backend()
        params = self.create_test_params()
        
        # Generate test data
        T = 15
        observations = jax.random.normal(jax.random.PRNGKey(2), (T, params.emissions.weights.shape[0]))
        
        # Run smoother
        smoothed_means, smoothed_covs = backend.smoother(params, observations)
        
        # Check shapes
        assert smoothed_means.shape == (T, params.dynamics.weights.shape[0])
        assert smoothed_covs.shape == (T, params.dynamics.weights.shape[0], params.dynamics.weights.shape[0])
        
        print("✓ Backend smoother method works")
    
    #posterior_sample doesnt work because dynamax requires an input_weights for dynamics 
    #TODO: update posterior_sample to work with dynamics without inputs
    def test_backend_posterior_sample(self):
        """Test backend posterior sampling."""
        backend = self.create_test_backend()
        params = self.create_test_params()
        
        # Generate test data
        T = 25
        observations = jax.random.normal(jax.random.PRNGKey(3), (T, params.emissions.weights.shape[0]))
        
        # Run posterior sampling
        key = jax.random.PRNGKey(4)
        samples = backend.posterior_sample(key, params, observations)
        
        # Check shape
        assert samples.shape == (T, params.dynamics.weights.shape[0])
        
        print("✓ Backend posterior sampling works")
    
    def test_backend_e_step(self):
        """Test backend E-step method."""
        backend = self.create_test_backend()
        params = self.create_test_params()
        
        # Generate test data
        T = 30
        observations = jax.random.normal(jax.random.PRNGKey(5), (T, params.emissions.weights.shape[0]))
        
        # Run E-step
        stats, ll = backend.e_step(params, observations)
        
        # Check that it returns SufficientStats
        assert isinstance(stats, SufficientStats)
        assert stats.latent_mean.shape == (T, params.dynamics.weights.shape[0])
        assert stats.latent_second_moment.shape == (T, params.dynamics.weights.shape[0], params.dynamics.weights.shape[0])
        assert stats.cross_time_moment.shape == (T-1, params.dynamics.weights.shape[0], params.dynamics.weights.shape[0])
        assert isinstance(stats.loglik, float)
        assert stats.T == T
        
        print("✓ Backend E-step method works")
    
    def test_backend_consistency(self):
        """Test that backend methods are consistent."""
        backend = self.create_test_backend()
        params = self.create_test_params()
        
        # Generate test data
        T = 20
        observations = jax.random.normal(jax.random.PRNGKey(6), (T, params.emissions.weights.shape[0]))
        
        # Run filter and smoother
        filtered_means, filtered_covs = backend.filter(params, observations)
        smoothed_means, smoothed_covs = backend.smoother(params, observations)
        
        # Run E-step
        stats, ll = backend.e_step(params, observations)
        
        # Check that smoothed means from smoother and E-step are consistent
        assert jnp.allclose(smoothed_means, stats.latent_mean)
        
        # Check that smoothed covariances from smoother and E-step are consistent
        assert jnp.allclose(smoothed_covs, stats.latent_second_moment - 
                           jnp.einsum('ti,tj->tij', stats.latent_mean, stats.latent_mean))
        
        print("✓ Backend methods are consistent")


class TestIntegration:
    """Integration tests for CTDS components."""
    
    def test_full_ctds_pipeline(self):
        """Test the full CTDS pipeline with E-step."""
        # Create CTDS parameters
        state_dim = 4
        emission_dim = 10
        
        initial = ParamsCTDSInitial(
            mean=jnp.zeros(state_dim),
            cov=jnp.eye(state_dim) * 0.1
        )
        
        A = jnp.array([
            [0.9, 0.1, -0.2, -0.2],
            [0.1, 0.9, -0.2, -0.2],
            [0.3, 0.3, -0.7, -0.1],
            [0.3, 0.3, -0.1, -0.7],
        ])
        Q = jnp.eye(state_dim) * 0.01
        dynamics = ParamsCTDSDynamics(
            weights=A, 
            cov=Q
        )
        
        C = jax.random.normal(jax.random.PRNGKey(0), (emission_dim, state_dim)) * 0.5
        C = C.at[:, :2].set(jnp.abs(C[:, :2]))
        C = C.at[:, 2:].set(-jnp.abs(C[:, 2:]))
        R = jnp.eye(emission_dim) * 0.1
        emissions = ParamsCTDSEmissions(weights=C, cov=R)
        
        constraints = ParamsCTDSConstraints(
            cell_types=jnp.array([0, 1]),
            cell_sign=jnp.array([1, -1]),
            cell_type_dimensions=jnp.array([2, 2]),
            cell_type_mask=jnp.array([0, 0, 1, 1]),
        )

        params = ParamsCTDS(
            initial=initial,
            dynamics=dynamics,
            emissions=emissions,
            constraints=constraints,
            observations=None

        )
        
        # Create backend
        backend = DynamaxLGSSMBackend()
        
        # Generate synthetic data
        T = 50
        key = jax.random.PRNGKey(1)
        
        # Sample latent states
        z = jnp.zeros((T, state_dim))
        z = z.at[0].set(jax.random.multivariate_normal(key, params.initial.mean, params.initial.cov))
        
        def step(carry, t):
            z_prev, key = carry
            key, subkey = jax.random.split(key)
            z_next = jax.random.multivariate_normal(
                subkey, 
                params.dynamics.weights @ z_prev, 
                params.dynamics.cov
            )
            return (z_next, key), z_next
        
        (z_final, _), z_sequence = jax.lax.scan(step, (z[0], key), jnp.arange(1, T))
        latent_states = z.at[1:].set(z_sequence)
        
        # Generate observations
        observations = jax.random.multivariate_normal(
            jax.random.PRNGKey(2),
            jnp.zeros(emission_dim),
            params.emissions.cov,
            shape=(T,)
        )
        observations = observations + latent_states @ params.emissions.weights.T
        
        # Test to_lgssm conversion
        lgssm_params = params.to_lgssm()
        assert isinstance(lgssm_params, ParamsLGSSM)
        
        # Test E-step
        stats, ll= backend.e_step(params, observations)
        assert isinstance(stats, SufficientStats)
        
        # Test that inference worked
        assert stats.loglik < 0, "Log-likelihood should be negative"
        
        # Test that smoothed estimates are reasonable
        mse = jnp.mean((stats.latent_mean - latent_states) ** 2)
        assert mse < 1.0, f"Mean squared error too high: {mse}"
        
        print("✓ Full CTDS pipeline works correctly")


def main():
    """Run all tests."""
    print("Running CTDS component tests...\n")
    
    try:
        # Test ParamsCTDS
        test_params = TestParamsCTDS()
        test_params.test_params_ctds_creation()
        test_params.test_to_lgssm_conversion()
        test_params.test_to_lgssm_preserves_structure()
        test_params.test_to_lgssm_different_sizes()
        
        # Test E-step
        test_e_step = TestCTDSEStep()
        test_e_step.test_e_step_basic()
        test_e_step.test_e_step_with_inputs()
        test_e_step.test_e_step_consistency()
        test_e_step.test_e_step_mathematical_properties()
        
        # Test Backend
        test_backend = TestDynamaxLGSSMBackend()
        test_backend.test_backend_creation()
        test_backend.test_backend_filter()
        test_backend.test_backend_smoother()
        #test_backend.test_backend_posterior_sample()
        test_backend.test_backend_e_step()
        test_backend.test_backend_consistency()
        
        # Test Integration
        test_integration = TestIntegration()
        test_integration.test_full_ctds_pipeline()
        
        print("\n🎉 All CTDS component tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()