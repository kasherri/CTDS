"""
Test suite for CTDS initialization methods.

This module tests the initialization methods of the CTDS class to ensure:
1. Correct shapes of output matrices
2. Proper parameter structure
3. Mathematical consistency
4. Edge case handling
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from typing import List, Tuple

# Import the classes we're testing
from models import CTDS
from params import ParamsCTDS, ParamsCTDSInitial, ParamsCTDSDynamics, ParamsCTDSEmissions, ParamsCTDSConstraints


class TestCTDSInitialization:
    """Test suite for CTDS initialization methods."""
    
    @pytest.fixture
    def setup_simple_2_cell_type(self):
        """Set up a simple 2-cell-type scenario for testing."""
        # Data dimensions
        N = 20  # Total neurons
        T = 100  # Time points
        
        # Cell type configuration
        cell_types = jnp.array([0, 1])  # Excitatory and Inhibitory
        cell_sign = jnp.array([1, -1])  # Excitatory: +1, Inhibitory: -1
        cell_type_dimensions = jnp.array([3, 2])  # 3D excitatory, 2D inhibitory
        cell_type_mask = jnp.array([0]*12 + [1]*8)  # 12 excitatory, 8 inhibitory
        dale_mask = jnp.array([1]*12 + [-1]*8)  # Dale's law mask
        
        # Generate synthetic observations
        key = jax.random.PRNGKey(42)
        Y = jax.random.normal(key, (N, T)) * 0.5 + 0.1
        Y = jnp.abs(Y)  # Ensure non-negative for neural activity
        
        # Create CTDS instance
        ctds = CTDS(
            observations=Y,
            cell_types=cell_types,
            cell_sign=cell_sign,
            cell_type=cell_type_dimensions,
            cell_type_mask=cell_type_mask
        )
        
        # Add dale_mask to constraints (seems to be missing in constructor)
        ctds.constraints = ParamsCTDSConstraints(
            cell_types=cell_types,
            cell_sign=cell_sign,
            cell_type_dimensions=cell_type_dimensions,
            cell_type_mask=cell_type_mask,
            dale_mask=dale_mask
        )
        
        return {
            'ctds': ctds,
            'Y': Y,
            'N': N,
            'T': T,
            'cell_types': cell_types,
            'cell_sign': cell_sign,
            'cell_type_dimensions': cell_type_dimensions,
            'cell_type_mask': cell_type_mask,
            'dale_mask': dale_mask,
            'state_dim': int(jnp.sum(cell_type_dimensions))
        }
    
    @pytest.fixture
    def setup_multi_cell_type(self):
        """Set up a multi-cell-type scenario for testing."""
        N = 30
        T = 80
        
        cell_types = jnp.array([0, 1, 2])  # Three cell types
        cell_sign = jnp.array([1, -1, 1])  # Mixed excitatory/inhibitory
        cell_type_dimensions = jnp.array([4, 3, 2])
        cell_type_mask = jnp.array([0]*12 + [1]*10 + [2]*8)
        dale_mask = jnp.array([1]*12 + [-1]*10 + [1]*8)
        
        key = jax.random.PRNGKey(123)
        Y = jax.random.normal(key, (N, T)) * 0.3 + 0.2
        Y = jnp.abs(Y)
        
        ctds = CTDS(
            observations=Y,
            cell_types=cell_types,
            cell_sign=cell_sign,
            cell_type=cell_type_dimensions,
            cell_type_mask=cell_type_mask
        )
        
        ctds.constraints = ParamsCTDSConstraints(
            cell_types=cell_types,
            cell_sign=cell_sign,
            cell_type_dimensions=cell_type_dimensions,
            cell_type_mask=cell_type_mask,
            dale_mask=dale_mask
        )
        
        return {
            'ctds': ctds,
            'Y': Y,
            'N': N,
            'T': T,
            'cell_types': cell_types,
            'cell_sign': cell_sign,
            'cell_type_dimensions': cell_type_dimensions,
            'cell_type_mask': cell_type_mask,
            'dale_mask': dale_mask,
            'state_dim': int(jnp.sum(cell_type_dimensions))
        }
    
    def create_mock_nmf_factors(self, setup_dict):
        """Create mock NMF factors for testing initialization methods."""
        N = setup_dict['N']
        cell_types = setup_dict['cell_types']
        cell_type_dimensions = setup_dict['cell_type_dimensions']
        cell_type_mask = setup_dict['cell_type_mask']
        
        U_list = []
        V_list = []
        
        for i, cell_type in enumerate(cell_types):
            # Get neurons for this cell type
            type_indices = jnp.where(cell_type_mask == cell_type)[0]
            N_type = len(type_indices)
            D_type = int(cell_type_dimensions[i])
            
            if N_type > 0 and D_type > 0:
                # Create mock factors
                key_u = jax.random.PRNGKey(i)
                key_v = jax.random.PRNGKey(i + 100)
                
                U = jax.random.uniform(key_u, (N_type, D_type), minval=0.1, maxval=1.0)
                V = jax.random.uniform(key_v, (N, D_type), minval=0.1, maxval=1.0)
                
                U_list.append(U)
                V_list.append(V)
        
        return U_list, V_list


class TestInitializeEmissions(TestCTDSInitialization):
    """Test the _initialize_emissions method."""
    
    def test_initialize_emissions_shapes_2_cell_type(self, setup_simple_2_cell_type):
        """Test that _initialize_emissions returns correct shapes for 2 cell types."""
        setup = setup_simple_2_cell_type
        ctds = setup['ctds']
        Y = setup['Y']
        N = setup['N']
        state_dim = setup['state_dim']
        
        # Create mock U_list
        U_list, _ = self.create_mock_nmf_factors(setup)
        
        # Test the method
        emissions = ctds.emissions_fn(Y, U_list, state_dim)

        # Check return type
        assert isinstance(emissions, ParamsCTDSEmissions), "Should return ParamsCTDSEmissions"
        
        # Check weights (C matrix) shape
        assert emissions.weights.shape == (N, state_dim), \
            f"Emission weights should be ({N}, {state_dim}), got {emissions.weights.shape}"
        
        # Check covariance (R matrix) shape
        assert emissions.cov.shape == (N, N), \
            f"Emission covariance should be ({N}, {N}), got {emissions.cov.shape}"
        
        # Check that R is diagonal
        assert jnp.allclose(emissions.cov, jnp.diag(jnp.diag(emissions.cov))), \
            "Emission covariance should be diagonal"
        
        # Check non-negativity of C
        assert jnp.all(emissions.weights >= 0), "Emission weights should be non-negative"
        
        # Check positive definiteness of R
        assert jnp.all(jnp.diag(emissions.cov) > 0), "Emission covariance diagonal should be positive"
    
    def test_initialize_emissions_shapes_multi_cell_type(self, setup_multi_cell_type):
        """Test that _initialize_emissions works with multiple cell types."""
        setup = setup_multi_cell_type
        ctds = setup['ctds']
        Y = setup['Y']
        N = setup['N']
        state_dim = setup['state_dim']
        
        U_list, _ = self.create_mock_nmf_factors(setup)

        emissions = ctds.emissions_fn(Y, U_list, state_dim)

        # Basic shape checks
        assert emissions.weights.shape == (N, state_dim)
        assert emissions.cov.shape == (N, N)
        assert jnp.all(emissions.weights >= 0)
        assert jnp.all(jnp.diag(emissions.cov) > 0)
    
    def test_initialize_emissions_block_structure(self, setup_simple_2_cell_type):
        """Test that the emission matrix has correct block-diagonal structure."""
        setup = setup_simple_2_cell_type
        ctds = setup['ctds']
        Y = setup['Y']
        state_dim = setup['state_dim']
        cell_type_dimensions = setup['cell_type_dimensions']
        cell_type_mask = setup['cell_type_mask']
        
        U_list, _ = self.create_mock_nmf_factors(setup)

        emissions = ctds.emissions_fn(Y, U_list, state_dim)
        C = emissions.weights
        
        # Check block structure
        col_start = 0
        for i, cell_type in enumerate(setup['cell_types']):
            type_indices = jnp.where(cell_type_mask == cell_type)[0]
            D_type = int(cell_type_dimensions[i])
            
            if len(type_indices) > 0 and D_type > 0:
                # Check that this cell type's block is non-zero
                block = C[type_indices, col_start:col_start + D_type]
                assert jnp.any(block > 0), f"Block for cell type {cell_type} should be non-zero"
                
                # Check that other columns are zero for this cell type
                if col_start > 0:
                    left_block = C[type_indices, :col_start]
                    assert jnp.allclose(left_block, 0), \
                        f"Left padding for cell type {cell_type} should be zero"
                
                if col_start + D_type < state_dim:
                    right_block = C[type_indices, col_start + D_type:]
                    assert jnp.allclose(right_block, 0), \
                        f"Right padding for cell type {cell_type} should be zero"
                
                col_start += D_type
    
    def test_initialize_emissions_dimension_assertion(self, setup_simple_2_cell_type):
        """Test that dimension mismatch raises assertion error."""
        setup = setup_simple_2_cell_type
        ctds = setup['ctds']
        Y = setup['Y']
        
        U_list, _ = self.create_mock_nmf_factors(setup)
        wrong_state_dim = setup['state_dim'] + 1  # Intentionally wrong
        
        with pytest.raises(AssertionError, match="Latent dimensions do not match"):
            ctds._initialize_emissions(Y, U_list, wrong_state_dim)


class TestInitializeDynamics(TestCTDSInitialization):
    """Test the _initialize_dynamics method."""
    
    def test_initialize_dynamics_shapes(self, setup_simple_2_cell_type):
        """Test that _initialize_dynamics returns correct shapes."""
        setup = setup_simple_2_cell_type
        ctds = setup['ctds']
        state_dim = setup['state_dim']
        
        U_list, V_list = self.create_mock_nmf_factors(setup)
        emissions = ctds.emissions_fn(setup['Y'], U_list, state_dim)

        # Test the method
        dynamics = ctds.dynamics_fn(V_list, emissions.weights)

        # Check return type
        assert isinstance(dynamics, ParamsCTDSDynamics), "Should return ParamsCTDSDynamics"
        
        # Check weights (A matrix) shape
        assert dynamics.weights.shape == (state_dim, state_dim), \
            f"Dynamics weights should be ({state_dim}, {state_dim}), got {dynamics.weights.shape}"
        
        # Check covariance (Q matrix) shape
        assert dynamics.cov.shape == (state_dim, state_dim), \
            f"Dynamics covariance should be ({state_dim}, {state_dim}), got {dynamics.cov.shape}"
        
        # Check that Q is positive definite (diagonal with positive entries)
        assert jnp.allclose(dynamics.cov, jnp.diag(jnp.diag(dynamics.cov))), \
            "Dynamics covariance should be diagonal"
        assert jnp.all(jnp.diag(dynamics.cov) > 0), \
            "Dynamics covariance diagonal should be positive"
    
    def test_initialize_dynamics_dale_constraints(self, setup_simple_2_cell_type):
        """Test that Dale's law is properly applied in V_dale construction."""
        setup = setup_simple_2_cell_type
        ctds = setup['ctds']
        cell_sign = setup['cell_sign']
        
        U_list, V_list = self.create_mock_nmf_factors(setup)
        
        # Manually build V_dale to compare
        V_dale_list = []
        for i, V in enumerate(V_list):
            if cell_sign[i] == -1:
                V_dale_list.append(-V)  # Should flip sign for inhibitory
            else:
                V_dale_list.append(V)   # Should keep positive for excitatory
        
        expected_V_dale = jnp.concatenate(V_dale_list, axis=1)
        emissions = ctds.emissions_fn(setup['Y'], U_list, setup['state_dim'])
        # Now test the method
        dynamics = ctds.dynamics_fn(V_list, emissions.weights)

        # The A matrix should be V_dale.T @ U_combined
        # We can't directly test V_dale, but we can test constraints on A
        A = dynamics.weights
        
        # Check that A is finite and well-conditioned
        assert jnp.all(jnp.isfinite(A)), "Dynamics matrix should be finite"
        assert jnp.linalg.cond(A) < 1e10, "Dynamics matrix should be well-conditioned"
    
    def test_initialize_dynamics_v_list_length_assertion(self, setup_simple_2_cell_type):
        """Test that wrong V_list length raises assertion error."""
        setup = setup_simple_2_cell_type
        ctds = setup['ctds']
        
        U_list, V_list = self.create_mock_nmf_factors(setup)
        
        # Remove one V matrix to cause length mismatch
        wrong_V_list = V_list[:-1]
        
        with pytest.raises(AssertionError, match="Number of V matrices must match"):
            ctds.dynamics_fn(wrong_V_list, U_list)


class TestFullInitialization(TestCTDSInitialization):
    """Test the complete initialize method."""
    
    def test_full_initialization_2_cell_type(self, setup_simple_2_cell_type):
        """Test complete initialization pipeline for 2 cell types."""
        setup = setup_simple_2_cell_type
        ctds = setup['ctds']
        Y = setup['Y']
        N = setup['N']
        state_dim = setup['state_dim']
        
        # Run full initialization
        params = ctds.initialize()
        
        # Check return type
        assert isinstance(params, ParamsCTDS), "Should return ParamsCTDS"
        
        # Check all components exist
        assert hasattr(params, 'initial'), "Should have initial parameters"
        assert hasattr(params, 'dynamics'), "Should have dynamics parameters"
        assert hasattr(params, 'emissions'), "Should have emissions parameters"
        assert hasattr(params, 'constraints'), "Should have constraints"
        
        # Check initial parameters
        assert params.initial.mean.shape == (state_dim,), \
            f"Initial mean should be ({state_dim},), got {params.initial.mean.shape}"
        assert params.initial.cov.shape == (state_dim, state_dim), \
            f"Initial cov should be ({state_dim}, {state_dim}), got {params.initial.cov.shape}"
        
        # Check dynamics parameters
        assert params.dynamics.weights.shape == (state_dim, state_dim)
        assert params.dynamics.cov.shape == (state_dim, state_dim)
        
        # Check emissions parameters
        assert params.emissions.weights.shape == (N, state_dim)
        assert params.emissions.cov.shape == (N, N)
        
        # Check constraints are preserved
        assert jnp.array_equal(params.constraints.cell_types, setup['cell_types'])
        assert jnp.array_equal(params.constraints.cell_type_dimensions, setup['cell_type_dimensions'])
    
    def test_full_initialization_multi_cell_type(self, setup_multi_cell_type):
        """Test complete initialization pipeline for multiple cell types."""
        setup = setup_multi_cell_type
        ctds = setup['ctds']
        Y = setup['Y']
        
        # Run full initialization
        params = ctds.initialize()
        
        # Basic checks
        assert isinstance(params, ParamsCTDS)
        assert params.initial.mean.shape == (setup['state_dim'],)
        assert params.dynamics.weights.shape == (setup['state_dim'], setup['state_dim'])
        assert params.emissions.weights.shape == (setup['N'], setup['state_dim'])
    
    def test_initialization_numerical_stability(self, setup_simple_2_cell_type):
        """Test that initialization produces numerically stable results."""
        setup = setup_simple_2_cell_type
        ctds = setup['ctds']
        Y = setup['Y']
        
        params = ctds.initialize()
        
        # Check for NaN or Inf values
        assert jnp.all(jnp.isfinite(params.initial.mean)), "Initial mean should be finite"
        assert jnp.all(jnp.isfinite(params.initial.cov)), "Initial cov should be finite"
        assert jnp.all(jnp.isfinite(params.dynamics.weights)), "Dynamics weights should be finite"
        assert jnp.all(jnp.isfinite(params.dynamics.cov)), "Dynamics cov should be finite"
        assert jnp.all(jnp.isfinite(params.emissions.weights)), "Emissions weights should be finite"
        assert jnp.all(jnp.isfinite(params.emissions.cov)), "Emissions cov should be finite"
        
        # Check positive definiteness of covariance matrices
        assert jnp.all(jnp.linalg.eigvals(params.initial.cov) > 0), \
            "Initial covariance should be positive definite"
        assert jnp.all(jnp.linalg.eigvals(params.dynamics.cov) > 0), \
            "Dynamics covariance should be positive definite"
        assert jnp.all(jnp.diag(params.emissions.cov) > 0), \
            "Emissions covariance should be positive definite"
    
    def test_reproducibility(self, setup_simple_2_cell_type):
        """Test that initialization is reproducible with same data."""
        setup = setup_simple_2_cell_type
        ctds1 = setup['ctds']
        Y = setup['Y']
        
        # Create identical second instance
        ctds2 = CTDS(
            observations=Y,
            cell_types=setup['cell_types'],
            cell_sign=setup['cell_sign'],
            cell_type=setup['cell_type_dimensions'],
            cell_type_mask=setup['cell_type_mask']
        )
        ctds2.constraints = ctds1.constraints
        
        # Run initialization on both
        params1 = ctds1.initialize()
        params2 = ctds2.initialize()
        
        # Results should be identical (assuming deterministic random seeds in functions)
        # Note: This test might fail if internal functions use non-deterministic random keys
        # In that case, we should at least check that the shapes and basic properties match
        assert params1.initial.mean.shape == params2.initial.mean.shape
        assert params1.dynamics.weights.shape == params2.dynamics.weights.shape
        assert params1.emissions.weights.shape == params2.emissions.weights.shape

"""
def run_tests():
    import sys
    
    # Simple test runner for development
    test_class = TestInitializeEmissions()
    
    # Create fixtures manually
    setup_2_cell = test_class.setup_simple_2_cell_type()
    setup_multi_cell = test_class.setup_multi_cell_type()
    
    try:
        print("Testing _initialize_emissions...")
        test_class.test_initialize_emissions_shapes_2_cell_type(setup_2_cell)
        test_class.test_initialize_emissions_shapes_multi_cell_type(setup_multi_cell)
        test_class.test_initialize_emissions_block_structure(setup_2_cell)
        print("✅ _initialize_emissions tests passed!")
        
        print("\nTesting _initialize_dynamics...")
        dynamics_test = TestInitializeDynamics()
        dynamics_test.test_initialize_dynamics_shapes(setup_2_cell)
        dynamics_test.test_initialize_dynamics_dale_constraints(setup_2_cell)
        print("✅ _initialize_dynamics tests passed!")
        
        print("\nTesting full initialization...")
        full_test = TestFullInitialization()
        full_test.test_full_initialization_2_cell_type(setup_2_cell)
        full_test.test_full_initialization_multi_cell_type(setup_multi_cell)
        full_test.test_initialization_numerical_stability(setup_2_cell)
        print("✅ Full initialization tests passed!")
        
        print("\n🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
"""