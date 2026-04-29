"""
Test structural invariants and constraints after M-step.
"""
import pytest
import jax
import jax.numpy as jnp
from ctds.models import CTDS
from tests.test_helpers import (
    assert_psd, assert_dale_columns, assert_nonnegative,
    generate_synthetic_ssm, perturb_params
)

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def small_synthetic_problem():
    """Generate small synthetic problem for testing."""
    D = 6
    N = 20
    T = 50
    
    # 2 cell types: 3 excitatory, 3 inhibitory dims
    cell_types = jnp.array([0, 1])
    cell_sign = jnp.array([1, -1])  # excitatory, inhibitory
    cell_type_dimensions = jnp.array([3, 3])
    
    # Neurons: 10 excitatory, 10 inhibitory
    cell_type_mask = jnp.concatenate([
        jnp.zeros(10, dtype=jnp.int32),  # First 10 neurons are type 0
        jnp.ones(10, dtype=jnp.int32)    # Last 10 neurons are type 1
    ])
    
    key = jax.random.PRNGKey(42)
    
    params_true, latent, obs = generate_synthetic_ssm(
        D, N, T,
        cell_types, cell_sign, cell_type_dimensions, cell_type_mask,
        key
    )
    
    # Create model instance
    ctds = CTDS(
        emission_dim=N,
        cell_types=cell_types,
        cell_sign=cell_sign,
        cell_type_dimensions=cell_type_dimensions,
        cell_type_mask=cell_type_mask,
        state_dim=D
    )
    
    # Perturb parameters for initialization
    key_init = jax.random.PRNGKey(123)
    params_init = perturb_params(params_true, key_init, scale=0.3)
    
    return {
        'ctds': ctds,
        'params_true': params_true,
        'params_init': params_init,
        'observations': obs,
        'D': D,
        'N': N,
        'T': T
    }


def test_shapes_after_m_step(small_synthetic_problem):
    """Test that all parameter shapes are correct after M-step."""
    prob = small_synthetic_problem
    ctds = prob['ctds']
    params = prob['params_init']
    obs = prob['observations']
    D, N = prob['D'], prob['N']
    
    # Prepare batch observations
    batch_obs = jnp.expand_dims(obs, axis=0)  # (1, T, N)
    
    # Run one EM step
    from ctds.inference import DynamaxLGSSMBackend
    batch_stats, lls = jax.vmap(
        DynamaxLGSSMBackend.e_step, 
        in_axes=(None, 0, None)
    )(params, batch_obs, None)
    
    m_step_state = ctds.initialize_m_step_state(params, None)
    params_new, _ = ctds.m_step(params, None, batch_stats, m_step_state)
    
    # Check shapes
    assert params_new.dynamics.weights.shape == (D, D), f"A shape wrong: {params_new.dynamics.weights.shape}"
    assert params_new.dynamics.cov.shape == (D, D), f"Q shape wrong: {params_new.dynamics.cov.shape}"
    assert params_new.emissions.weights.shape == (N, D), f"C shape wrong: {params_new.emissions.weights.shape}"
    assert params_new.emissions.cov.shape == (N, N), f"R shape wrong: {params_new.emissions.cov.shape}"
    assert params_new.initial.mean.shape == (D,), f"initial mean shape wrong"
    assert params_new.initial.cov.shape == (D, D), f"initial cov shape wrong"


def test_Q_is_psd_after_m_step(small_synthetic_problem):
    """Test that Q remains PSD after M-step."""
    prob = small_synthetic_problem
    ctds = prob['ctds']
    params = prob['params_init']
    obs = prob['observations']
    
    batch_obs = jnp.expand_dims(obs, axis=0)
    
    from ctds.inference import DynamaxLGSSMBackend
    batch_stats, _ = jax.vmap(
        DynamaxLGSSMBackend.e_step,
        in_axes=(None, 0, None)
    )(params, batch_obs, None)
    
    m_step_state = ctds.initialize_m_step_state(params, None)
    params_new, _ = ctds.m_step(params, None, batch_stats, m_step_state)
    
    Q = params_new.dynamics.cov
    
    # Check Q is PSD
    assert_psd(Q, "Q", tol=-1e-8)


def test_R_is_psd_after_m_step(small_synthetic_problem):
    """Test that R remains PSD after M-step."""
    prob = small_synthetic_problem
    ctds = prob['ctds']
    params = prob['params_init']
    obs = prob['observations']
    
    batch_obs = jnp.expand_dims(obs, axis=0)
    
    from ctds.inference import DynamaxLGSSMBackend
    batch_stats, _ = jax.vmap(
        DynamaxLGSSMBackend.e_step,
        in_axes=(None, 0, None)
    )(params, batch_obs, None)
    
    m_step_state = ctds.initialize_m_step_state(params, None)
    params_new, _ = ctds.m_step(params, None, batch_stats, m_step_state)
    
    R = params_new.emissions.cov
    
    # Check R is PSD
    assert_psd(R, "R", tol=-1e-8)


def test_dale_constraints_on_A(small_synthetic_problem):
    """Test that Dale's law constraints are satisfied on A."""
    prob = small_synthetic_problem
    ctds = prob['ctds']
    params = prob['params_init']
    obs = prob['observations']
    
    batch_obs = jnp.expand_dims(obs, axis=0)
    
    from ctds.inference import DynamaxLGSSMBackend
    batch_stats, _ = jax.vmap(
        DynamaxLGSSMBackend.e_step,
        in_axes=(None, 0, None)
    )(params, batch_obs, None)
    
    m_step_state = ctds.initialize_m_step_state(params, None)
    params_new, _ = ctds.m_step(params, None, batch_stats, m_step_state)
    
    A = params_new.dynamics.weights
    dynamics_mask = params_new.dynamics.dynamics_mask
    
    # Check Dale constraints
    assert_dale_columns(A, dynamics_mask, tol=1e-5)


def test_C_nonnegative(small_synthetic_problem):
    """Test that C remains non-negative after M-step."""
    prob = small_synthetic_problem
    ctds = prob['ctds']
    params = prob['params_init']
    obs = prob['observations']
    
    batch_obs = jnp.expand_dims(obs, axis=0)
    
    from ctds.inference import DynamaxLGSSMBackend
    batch_stats, _ = jax.vmap(
        DynamaxLGSSMBackend.e_step,
        in_axes=(None, 0, None)
    )(params, batch_obs, None)
    
    m_step_state = ctds.initialize_m_step_state(params, None)
    params_new, _ = ctds.m_step(params, None, batch_stats, m_step_state)
    
    C = params_new.emissions.weights
    
    # Check C is non-negative
    assert_nonnegative(C, "C", tol=-1e-6)


def test_no_nans_after_m_step(small_synthetic_problem):
    """Test that no NaNs appear after M-step."""
    prob = small_synthetic_problem
    ctds = prob['ctds']
    params = prob['params_init']
    obs = prob['observations']
    
    batch_obs = jnp.expand_dims(obs, axis=0)
    
    from ctds.inference import DynamaxLGSSMBackend
    batch_stats, _ = jax.vmap(
        DynamaxLGSSMBackend.e_step,
        in_axes=(None, 0, None)
    )(params, batch_obs, None)
    
    m_step_state = ctds.initialize_m_step_state(params, None)
    params_new, _ = ctds.m_step(params, None, batch_stats, m_step_state)
    
    # Check for NaNs
    assert not jnp.any(jnp.isnan(params_new.dynamics.weights)), "A contains NaN"
    assert not jnp.any(jnp.isnan(params_new.dynamics.cov)), "Q contains NaN"
    assert not jnp.any(jnp.isnan(params_new.emissions.weights)), "C contains NaN"
    assert not jnp.any(jnp.isnan(params_new.emissions.cov)), "R contains NaN"


def test_no_infs_after_m_step(small_synthetic_problem):
    """Test that no Infs appear after M-step."""
    prob = small_synthetic_problem
    ctds = prob['ctds']
    params = prob['params_init']
    obs = prob['observations']
    
    batch_obs = jnp.expand_dims(obs, axis=0)
    
    from ctds.inference import DynamaxLGSSMBackend
    batch_stats, _ = jax.vmap(
        DynamaxLGSSMBackend.e_step,
        in_axes=(None, 0, None)
    )(params, batch_obs, None)
    
    m_step_state = ctds.initialize_m_step_state(params, None)
    params_new, _ = ctds.m_step(params, None, batch_stats, m_step_state)
    
    # Check for Infs
    assert not jnp.any(jnp.isinf(params_new.dynamics.weights)), "A contains Inf"
    assert not jnp.any(jnp.isinf(params_new.dynamics.cov)), "Q contains Inf"
    assert not jnp.any(jnp.isinf(params_new.emissions.weights)), "C contains Inf"
    assert not jnp.any(jnp.isinf(params_new.emissions.cov)), "R contains Inf"


def test_R_diagonal_has_minimum(small_synthetic_problem):
    """Test that diagonal of R has minimum values."""
    prob = small_synthetic_problem
    ctds = prob['ctds']
    params = prob['params_init']
    obs = prob['observations']
    
    batch_obs = jnp.expand_dims(obs, axis=0)
    
    from ctds.inference import DynamaxLGSSMBackend
    batch_stats, _ = jax.vmap(
        DynamaxLGSSMBackend.e_step,
        in_axes=(None, 0, None)
    )(params, batch_obs, None)
    
    m_step_state = ctds.initialize_m_step_state(params, None)
    params_new, _ = ctds.m_step(params, None, batch_stats, m_step_state)
    
    R = params_new.emissions.cov
    R_diag = jnp.diag(R)
    
    # Check minimum diagonal (should be at least 1e-3 based on code)
    min_noise = 1e-3
    assert jnp.all(R_diag >= min_noise * 0.9), f"R diagonal too small: min={jnp.min(R_diag)}"
