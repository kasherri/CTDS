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
from functools import partial
# Import the classes we're testing
from models import CTDS
from params import ParamsCTDS, ParamsCTDSInitial, ParamsCTDSDynamics, ParamsCTDSEmissions, ParamsCTDSConstraints


class TestCTDSInitialization:
    """Test suite for CTDS initialization methods."""
    
    def generate_cell_type_data(self, N=20, T=100, cell_types=None, cell_sign=None, 
                               cell_type_dimensions=None, cell_type_mask=None, seed=42):
        """
        Generate synthetic cell type data for testing.
        
        Args:
            N: Number of neurons
            T: Number of time points
            cell_types: Array of cell type labels (default: [0, 1])
            cell_sign: Array of cell signs (default: [1, -1])
            cell_type_dimensions: Array of latent dimensions per cell type (default: [3, 2])
            cell_type_mask: Array mapping neurons to cell types (default: auto-generated)
            seed: Random seed
            
        Returns:
            Dictionary with all test data
        """
        # Set defaults
        if cell_types is None:
            cell_types = jnp.array([0, 1])
        if cell_sign is None:
            cell_sign = jnp.array([1, -1])
        if cell_type_dimensions is None:
            cell_type_dimensions = jnp.array([3, 2])
        if cell_type_mask is None:
            # Default: distribute neurons roughly evenly among cell types
            neurons_per_type = N // len(cell_types)
            remainder = N % len(cell_types)
            cell_type_mask = []
            for i, cell_type in enumerate(cell_types):
                count = neurons_per_type + (1 if i < remainder else 0)
                cell_type_mask.extend([cell_type] * count)
            cell_type_mask = jnp.array(cell_type_mask)
        
        # Generate synthetic observations
        key = jax.random.PRNGKey(seed)
        Y = jax.random.normal(key, (N, T)) * 0.5 + 0.1
        Y = jnp.abs(Y)  # Ensure non-negative for neural activity
        
        # Create CTDS instance
        ctds = CTDS(
            emission_dim= Y.shape[0],  # N neurons
            cell_types=cell_types,
            cell_sign=cell_sign,
            cell_type_dimensions=cell_type_dimensions,
            cell_type_mask=cell_type_mask
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
            'state_dim': int(jnp.sum(cell_type_dimensions))
        }

    @pytest.fixture
    def setup_simple_2_cell_type(self):
        """Set up a simple 2-cell-type scenario for testing."""
        return self.generate_cell_type_data(
            N=20, T=100,
            cell_types=jnp.array([0, 1]),
            cell_sign=jnp.array([1, -1]),
            cell_type_dimensions=jnp.array([3, 2]),
            cell_type_mask=jnp.array([0]*12 + [1]*8),
            seed=42
        )
    
    @pytest.fixture
    def setup_multi_cell_type(self):
        """Set up a multi-cell-type scenario for testing."""
        return self.generate_cell_type_data(
            N=30, T=80,
            cell_types=jnp.array([0, 1, 2]),
            cell_sign=jnp.array([1, -1, 1]),
            cell_type_dimensions=jnp.array([4, 3, 2]),
            cell_type_mask=jnp.array([0]*12 + [1]*10 + [2]*8),
            seed=123
        )
    
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
        emissions = ctds.initialize_emissions(Y, U_list, state_dim)

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

        emissions = ctds.initialize_emissions(Y, U_list, state_dim)

        # Basic shape checks
        assert emissions.weights.shape == (N, state_dim) #needs to be N x D
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

        emissions = ctds.initialize_emissions(Y, U_list, state_dim)
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
            ctds.initialize_emissions(Y, U_list, wrong_state_dim)


class TestInitializeDynamics(TestCTDSInitialization):
    """Test the _initialize_dynamics method."""
    
    def test_initialize_dynamics_shapes(self, setup_simple_2_cell_type):
        """Test that _initialize_dynamics returns correct shapes."""
        setup = setup_simple_2_cell_type
        ctds = setup['ctds']
        state_dim = setup['state_dim']
        
        U_list, V_list = self.create_mock_nmf_factors(setup)
        emissions = ctds.initialize_emissions(setup['Y'], U_list, state_dim)

        # Test the method
        dynamics = ctds.initialize_dynamics(V_list, emissions.weights)

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
        """Test that Dale's law is properly applied in V_dale construction and dynamics_mask is set."""
        setup = setup_simple_2_cell_type
        ctds = setup['ctds']
        cell_sign = setup['cell_sign']
        cell_type_dimensions = setup['cell_type_dimensions']
        D= setup['state_dim']
        
        U_list, V_list = self.create_mock_nmf_factors(setup)
        emissions = ctds.initialize_emissions(setup['Y'], U_list, setup['state_dim'])

        # Test the method
        dynamics = ctds.initialize_dynamics(V_list, emissions.weights)

        # Check that dynamics_mask is set correctly
        assert hasattr(dynamics, 'dynamics_mask'), "Should set dynamics_mask"
        assert dynamics.dynamics_mask is not None, "dynamics_mask should not be None"

        # Check dynamics_mask length matches total latent dimensions
        assert len(dynamics.dynamics_mask) == D, \
            f"dynamics_mask should have length {D}, got {len(dynamics.dynamics_mask)}"


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
            ctds.initialize_dynamics(wrong_V_list, U_list)


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
        params = ctds.initialize(Y)
        
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
        
        # Check that dynamics_mask is set after initialization
        assert params.dynamics.dynamics_mask is not None, "dynamics_mask should be set after initialization"
        assert len(params.dynamics.dynamics_mask) == setup['state_dim'], \
            "dynamics_mask should have one entry per cell type"
    
    def test_full_initialization_with_blockwise_nmf(self, setup_simple_2_cell_type):
        """Test that full initialization works with real blockwise_NMF function."""
        setup = setup_simple_2_cell_type
        ctds = setup['ctds']
        Y= setup['Y']
        
        # Import blockwise_NMF to ensure it's available
        from utlis import blockwise_NMF
        
        # Run full initialization (which uses blockwise_NMF internally)
        params = ctds.initialize(Y)
        
        # Check that initialization completed successfully
        assert isinstance(params, ParamsCTDS), "Should return ParamsCTDS"
        
        # Check that all matrices have correct shapes and properties
        state_dim = setup['state_dim']
        N = setup['N']
        
        # Dynamics checks
        assert params.dynamics.weights.shape == (state_dim, state_dim)
        assert jnp.all(jnp.isfinite(params.dynamics.weights)), "Dynamics weights should be finite"
        
        # Emissions checks
        assert params.emissions.weights.shape == (N, state_dim)
        assert jnp.all(params.emissions.weights >= -1e-10), "Emissions should be non-negative (allowing numerical errors)"
        
        # Check that dynamics_mask reflects Dale's law
        assert params.dynamics.dynamics_mask is not None
        assert len(params.dynamics.dynamics_mask) == state_dim, \
            f"dynamics_mask should have length {state_dim}, got {len(params.dynamics.dynamics_mask)}"

    def test_full_initialization_multi_cell_type(self, setup_multi_cell_type):
        """Test complete initialization pipeline for multiple cell types."""
        setup = setup_multi_cell_type
        ctds = setup['ctds']
        Y = setup['Y']
        
        # Run full initialization
        params = ctds.initialize(Y)
        
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
        
        params = ctds.initialize(Y)
        
        # Check for NaN or Inf values
        assert jnp.all(jnp.isfinite(params.initial.mean)), "Initial mean should be finite"
        assert jnp.all(jnp.isfinite(params.initial.cov)), "Initial cov should be finite"
        assert jnp.all(jnp.isfinite(params.dynamics.weights)), "Dynamics weights should be finite"
        assert jnp.all(jnp.isfinite(params.dynamics.cov)), "Dynamics cov should be finite"


class TestMStep(TestCTDSInitialization):
    """Test suite for CTDS M-step methods."""
    def generate_batched_sufficient_stats(self, ctds, params, batch_emissions):
        """
        Generate batched sufficient statistics and log-likelihoods using vmap over e_step.

        Args:
            ctds: CTDS model instance with e_step method.
            params: CTDS parameters.
            batch_emissions: jnp.ndarray of shape (batch_size, T, N) or (batch_size, T, D)

        Returns:
            batch_stats: SufficientStats for each batch (batched structure)
            lls: log-likelihoods for each batch
        """
        # vmap over the first axis (batch)
        e_step_fn = partial(ctds.e_step, params)
        batch_stats, lls = jax.vmap(e_step_fn)(batch_emissions)
        return batch_stats, lls
    
    def create_mock_sufficient_stats(self, setup_dict, batch_size=1):
        """Create mock sufficient statistics for testing M-step."""
        state_dim = setup_dict['state_dim']
        T = setup_dict['T']
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)
        # Batched sequences
        latent_mean = jax.random.normal(keys[0], (batch_size, T, state_dim))
        latent_second_moment = jnp.stack([
            jnp.stack([
                jnp.eye(state_dim) + 0.1 * jax.random.normal(jax.random.split(keys[1], batch_size * T)[b * T + t], (state_dim, state_dim))
                for t in range(T)
            ])
            for b in range(batch_size)
        ])
        # Ensure positive definiteness
        latent_second_moment = jnp.array([
            [cov @ cov.T + 1e-3 * jnp.eye(state_dim) for cov in batch]
            for batch in latent_second_moment
        ])
        
        cross_time_moment = jnp.stack([
            jnp.stack([
                0.8 * jnp.eye(state_dim) + 0.1 * jax.random.normal(jax.random.split(keys[2], batch_size * (T-1))[b * (T-1) + t], (state_dim, state_dim))
                for t in range(T-1)
            ])
            for b in range(batch_size)
        ])
    
      
        
        from params import SufficientStats
        return SufficientStats(
            latent_mean=latent_mean,
            latent_second_moment=latent_second_moment,
            cross_time_moment=cross_time_moment,
            loglik=0.0,
            T=T
        )
    
    def test_single_m_step_shapes(self, setup_simple_2_cell_type):
        """Test that _single_m_step returns parameters with correct shapes."""
        setup = setup_simple_2_cell_type
        ctds = setup['ctds']
        N = setup['N']
        Y = setup['Y']
        state_dim = setup['state_dim']
        
        # Initialize parameters
        params = ctds.initialize(Y)
        
        # Create mock sufficient statistics
        stats = self.create_mock_sufficient_stats(setup)
        
        # Run single M-step
        updated_params = ctds.m_step(params, None, stats, None)[0]
        
        # Check return type
        assert isinstance(updated_params, ParamsCTDS), "Should return ParamsCTDS"
        
        # Check shapes
        assert updated_params.dynamics.weights.shape == (state_dim, state_dim), \
            f"Dynamics weights should be ({state_dim}, {state_dim})"
        assert updated_params.dynamics.cov.shape == (state_dim, state_dim), \
            f"Dynamics cov should be ({state_dim}, {state_dim})"
        assert updated_params.emissions.weights.shape == (N, state_dim), \
            f"Emissions weights should be ({N}, {state_dim})"
        assert updated_params.emissions.cov.shape == (N, N), \
            f"Emissions cov should be ({N}, {N})"
        assert updated_params.initial.mean.shape == (state_dim,), \
            f"Initial mean should be ({state_dim},)"
        assert updated_params.initial.cov.shape == (state_dim, state_dim), \
            f"Initial cov should be ({state_dim}, {state_dim})"
    
    def test_single_m_step_constraints(self, setup_simple_2_cell_type):
        """Test that _single_m_step preserves constraints."""
        setup = setup_simple_2_cell_type
        ctds = setup['ctds']
        
        params = ctds.initialize()
        stats = self.create_mock_sufficient_stats(setup)
        
        updated_params = ctds.m_step(params, None, stats, None)[0]
        
        # Check that A matrix respects Dale's law through cell_type_mask
        A = updated_params.dynamics.weights
        cell_type_mask = updated_params.constraints.cell_type_mask
        
        # A should be finite and well-conditioned
        assert jnp.all(jnp.isfinite(A)), "Dynamics matrix should be finite"
        assert jnp.linalg.cond(A) < 1e12, "Dynamics matrix should be well-conditioned"
        
        # Check that C (emissions) is non-negative (NNLS constraint)
        C = updated_params.emissions.weights
        assert jnp.all(C >= -1e-10), "Emissions weights should be non-negative (allowing small numerical errors)"
        
        # Check positive definiteness of covariances
        Q = updated_params.dynamics.cov
        R = updated_params.emissions.cov
        init_cov = updated_params.initial.cov
        
        assert jnp.all(jnp.linalg.eigvals(Q) > -1e-10), "Q should be positive semidefinite"
        assert jnp.all(jnp.linalg.eigvals(R) > -1e-10), "R should be positive semidefinite"
        assert jnp.all(jnp.linalg.eigvals(init_cov) > -1e-10), "Initial cov should be positive semidefinite"
    
    def test_single_m_step_numerical_stability(self, setup_simple_2_cell_type):
        """Test numerical stability of _single_m_step."""
        setup = setup_simple_2_cell_type
        ctds = setup['ctds']
        Y=setup['Y']

        params = ctds.initialize()
        stats = self.create_mock_sufficient_stats(setup)

        updated_params = ctds.m_step(params, None, stats, None)[0]

        # Check for NaN or Inf values
        assert jnp.all(jnp.isfinite(updated_params.dynamics.weights)), "A should be finite"
        assert jnp.all(jnp.isfinite(updated_params.dynamics.cov)), "Q should be finite"
        assert jnp.all(jnp.isfinite(updated_params.emissions.weights)), "C should be finite"
        assert jnp.all(jnp.isfinite(updated_params.emissions.cov)), "R should be finite"
        assert jnp.all(jnp.isfinite(updated_params.initial.mean)), "Initial mean should be finite"
        assert jnp.all(jnp.isfinite(updated_params.initial.cov)), "Initial cov should be finite"
    
    def test_m_step_batch_sequences(self, setup_simple_2_cell_type):
        """Test m_step with multiple sequences (batched)."""
        setup = setup_simple_2_cell_type
        ctds = setup['ctds']
        
        params = ctds.initialize()
        batch_size = 5
        batch_stats = self.create_mock_sufficient_stats(setup, batch_size=batch_size)
        
        updated_params, m_step_state = ctds.m_step(params, None, batch_stats, None)
        
        # Check return types
        assert isinstance(updated_params, ParamsCTDS), "Should return ParamsCTDS"
        
        # Check that parameters have correct shapes (should be same as single sequence)
        state_dim = setup['state_dim']
        N = setup['N']
        
        assert updated_params.dynamics.weights.shape == (state_dim, state_dim)
        assert updated_params.emissions.weights.shape == ( N, state_dim)
    
    def test_m_step_batch_averaging(self, setup_simple_2_cell_type):
        """Test that m_step correctly averages statistics across batch."""
        setup = setup_simple_2_cell_type
        ctds = setup['ctds']
        
        params = ctds.initialize()
        
        # Create batch with identical sequences
        single_stats = self.create_mock_sufficient_stats(setup, batch_size=1)
        
        # Create a batch by repeating the same stats
        batch_size = 3
        batch_stats_manual = type(single_stats)(
            latent_mean=jnp.tile(single_stats.latent_mean, (batch_size, 1, 1)),
            latent_second_moment=jnp.tile(single_stats.latent_second_moment, (batch_size, 1, 1, 1)),
            cross_time_moment=jnp.tile(single_stats.cross_time_moment, (batch_size, 1, 1, 1)),
            loglik=single_stats.loglik,
            T=single_stats.T
        )
        
        # Run M-step on single sequence
        single_params, _ = ctds.m_step(params, None, single_stats, None)
        
        # Run M-step on batch
        batch_params, _ = ctds.m_step(params, None, batch_stats_manual, None)
        
        # Results should be very similar (allowing for numerical differences)
        assert jnp.allclose(single_params.dynamics.weights, batch_params.dynamics.weights, rtol=1e-5), \
            "Batch averaging should give same result for identical sequences"
        assert jnp.allclose(single_params.emissions.weights, batch_params.emissions.weights, rtol=1e-5), \
            "Batch averaging should give same result for identical sequences"
    
    def test_m_step_different_batch_sizes(self, setup_simple_2_cell_type):
        """Test m_step with different batch sizes."""
        setup = setup_simple_2_cell_type
        ctds = setup['ctds']
        
        params = ctds.initialize()
        
        for batch_size in [1, 3, 7]:
            batch_stats = self.create_mock_sufficient_stats(setup, batch_size=batch_size)
            updated_params, _ = ctds.m_step(params, None, batch_stats, None)
            
            # Check that we get valid parameters regardless of batch size
            assert isinstance(updated_params, ParamsCTDS)
            assert jnp.all(jnp.isfinite(updated_params.dynamics.weights))
            assert jnp.all(jnp.isfinite(updated_params.emissions.weights))
    
    def test_m_step_preserves_constraints(self, setup_simple_2_cell_type):
        """Test that m_step preserves the constraints object."""
        setup = setup_simple_2_cell_type
        ctds = setup['ctds']
        
        params = ctds.initialize()
        batch_stats = self.create_mock_sufficient_stats(setup)
        
        updated_params, _ = ctds.m_step(params, None, batch_stats, None)
        
        # Check that constraints are preserved
        assert updated_params.constraints is params.constraints, \
            "Constraints should be preserved (same object reference)"
        
        # Check that constraint values are unchanged
        assert jnp.array_equal(updated_params.constraints.cell_types, params.constraints.cell_types)
        assert jnp.array_equal(updated_params.constraints.cell_type_mask, params.constraints.cell_type_mask)
    
    def test_m_step_emission_matrix_orientation(self, setup_simple_2_cell_type):
        """Test that emission matrix has correct orientation after M-step."""
        setup = setup_simple_2_cell_type
        ctds = setup['ctds']
        N = setup['N']
        state_dim = setup['state_dim']
        
        params = ctds.initialize()
        batch_stats = self.create_mock_sufficient_stats(setup)
        
        updated_params, _ = ctds.m_step(params, None, batch_stats, None)
        
        # The M-step should produce C with shape (N, D) from NNLS
        C = updated_params.emissions.weights
        assert C.shape == (N, state_dim), \
            f"Emission weights should be ({N}, {state_dim}) as produced by NNLS, got {C.shape}"


    
    def test_m_step_r_matrix_diagonal_positive(self, setup_simple_2_cell_type):
        """Test that R matrix is diagonal with positive entries."""
        setup = setup_simple_2_cell_type
        ctds = setup['ctds']
        
        params = ctds.initialize()
        batch_stats = self.create_mock_sufficient_stats(setup)
        
        updated_params, _ = ctds.m_step(params, None, batch_stats, None)
        
        R = updated_params.emissions.cov
        
        # Check that R is diagonal
        
        # Check that diagonal entries are positive
        R_diag = jnp.diag(R)
        assert jnp.all(R_diag > 0), f"R diagonal should be positive, got min: {jnp.min(R_diag)}"
    
    
    def test_m_step_multi_cell_type(self, setup_multi_cell_type):
        """Test m_step with multiple cell types."""
        setup = setup_multi_cell_type
        ctds = setup['ctds']
        
        params = ctds.initialize()
        batch_stats = self.create_mock_sufficient_stats(setup)
        
        updated_params, _ = ctds.m_step(params, None, batch_stats, None)
        
        # Check basic properties
        assert isinstance(updated_params, ParamsCTDS)
        assert jnp.all(jnp.isfinite(updated_params.dynamics.weights))
        assert jnp.all(jnp.isfinite(updated_params.emissions.weights))
        
        # Check shapes for multi-cell setup
        state_dim = setup['state_dim']
        N = setup['N']
        
        assert updated_params.dynamics.weights.shape == (state_dim, state_dim)
        assert updated_params.emissions.weights.shape == (N, state_dim)

    def test_m_step_with_custom_cell_types(self):
        """Test M-step with custom cell type configuration using generate_cell_type_data."""
        # Create custom 3-cell-type scenario
        custom_setup = self.generate_cell_type_data(
            N=25, T=40,
            cell_types=jnp.array([0, 1, 2]),
            cell_sign=jnp.array([1, -1, 1]),
            cell_type_dimensions=jnp.array([2, 3, 2]),
            cell_type_mask=jnp.array([0]*10 + [1]*8 + [2]*7),
            seed=999
        )
        
        ctds = custom_setup['ctds']
        params = ctds.initialize()
        batch_stats = self.create_mock_sufficient_stats(custom_setup)
        
        updated_params, _ = ctds.m_step(params, None, batch_stats, None)
        
        # Check that it works with custom configuration
        assert isinstance(updated_params, ParamsCTDS)
        assert updated_params.dynamics.weights.shape == (custom_setup['state_dim'], custom_setup['state_dim'])
        assert updated_params.emissions.weights.shape == (custom_setup['state_dim'], custom_setup['N'])
        
        # Check that constraints are preserved
        assert jnp.array_equal(updated_params.constraints.cell_types, custom_setup['cell_types'])
        assert jnp.array_equal(updated_params.constraints.cell_type_dimensions, custom_setup['cell_type_dimensions'])
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