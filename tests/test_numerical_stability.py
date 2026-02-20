"""
Test numerical stability and edge cases.
"""
import pytest
import jax
import jax.numpy as jnp
from models import CTDS
from tests.test_helpers import generate_synthetic_ssm, perturb_params

jax.config.update("jax_enable_x64", True)


def test_Q_inverse_normalization():
    """Test that Q inverse normalization prevents overflow."""
    D = 5
    N = 15
    T = 100
    
    cell_types = jnp.array([0, 1])
    cell_sign = jnp.array([1, -1])
    cell_type_dimensions = jnp.array([3, 2])
    cell_type_mask = jnp.concatenate([
        jnp.zeros(8, dtype=jnp.int32),
        jnp.ones(7, dtype=jnp.int32)
    ])
    
    key = jax.random.PRNGKey(42)
    
    # Generate with very small Q (large Q_inv)
    params_true, latent, obs = generate_synthetic_ssm(
        D, N, T,
        cell_types, cell_sign, cell_type_dimensions, cell_type_mask,
        key,
        Q_scale=1e-4,  # Very small
        R_scale=0.5
    )
    
    ctds = CTDS(
        emission_dim=N,
        cell_types=cell_types,
        cell_sign=cell_sign,
        cell_type_dimensions=cell_type_dimensions,
        cell_type_mask=cell_type_mask,
        state_dim=D
    )
    
    key_init = jax.random.PRNGKey(123)
    params_init = perturb_params(params_true, key_init, scale=0.2)
    
    batch_obs = jnp.expand_dims(obs, axis=0)
    
    # Run EM - normalization should prevent overflow
    params_final, log_probs = ctds.fit_em(
        params_init,
        batch_obs,
        num_iters=10,
        verbose=False
    )
    
    # Check no NaNs or Infs
    assert not jnp.any(jnp.isnan(params_final.dynamics.weights)), "A contains NaN"
    assert not jnp.any(jnp.isinf(params_final.dynamics.weights)), "A contains Inf"
    assert not jnp.any(jnp.isnan(jnp.array(log_probs))), "LL contains NaN"


def test_nearly_singular_Q():
    """Test stability with nearly singular Q."""
    D = 4
    N = 10
    T = 80
    
    cell_types = jnp.array([0, 1])
    cell_sign = jnp.array([1, -1])
    cell_type_dimensions = jnp.array([2, 2])
    cell_type_mask = jnp.concatenate([
        jnp.zeros(5, dtype=jnp.int32),
        jnp.ones(5, dtype=jnp.int32)
    ])
    
    key = jax.random.PRNGKey(456)
    
    params_true, latent, obs = generate_synthetic_ssm(
        D, N, T,
        cell_types, cell_sign, cell_type_dimensions, cell_type_mask,
        key,
        Q_scale=1e-6,  # Nearly singular
        R_scale=0.5
    )
    
    ctds = CTDS(
        emission_dim=N,
        cell_types=cell_types,
        cell_sign=cell_sign,
        cell_type_dimensions=cell_type_dimensions,
        cell_type_mask=cell_type_mask,
        state_dim=D
    )
    
    key_init = jax.random.PRNGKey(789)
    params_init = perturb_params(params_true, key_init, scale=0.2)
    
    # Manually set Q to be nearly singular
    params_init.dynamics.cov = jnp.eye(D) * 1e-7
    
    batch_obs = jnp.expand_dims(obs, axis=0)
    
    # Run EM - regularization should handle it
    params_final, log_probs = ctds.fit_em(
        params_init,
        batch_obs,
        num_iters=5,
        verbose=False
    )
    
    # Check no NaNs
    assert not jnp.any(jnp.isnan(params_final.dynamics.cov)), "Q contains NaN"
    assert not jnp.any(jnp.isnan(params_final.dynamics.weights)), "A contains NaN"
    
    # Check Q is still PSD
    eigenvals = jnp.linalg.eigvalsh(params_final.dynamics.cov)
    assert jnp.all(eigenvals > -1e-8), f"Q not PSD: min eig={jnp.min(eigenvals)}"


def test_nearly_singular_R():
    """Test stability with nearly singular R."""
    D = 4
    N = 10
    T = 80
    
    cell_types = jnp.array([0, 1])
    cell_sign = jnp.array([1, -1])
    cell_type_dimensions = jnp.array([2, 2])
    cell_type_mask = jnp.concatenate([
        jnp.zeros(5, dtype=jnp.int32),
        jnp.ones(5, dtype=jnp.int32)
    ])
    
    key = jax.random.PRNGKey(111)
    
    params_true, latent, obs = generate_synthetic_ssm(
        D, N, T,
        cell_types, cell_sign, cell_type_dimensions, cell_type_mask,
        key,
        Q_scale=0.1,
        R_scale=1e-6  # Nearly singular
    )
    
    ctds = CTDS(
        emission_dim=N,
        cell_types=cell_types,
        cell_sign=cell_sign,
        cell_type_dimensions=cell_type_dimensions,
        cell_type_mask=cell_type_mask,
        state_dim=D
    )
    
    key_init = jax.random.PRNGKey(222)
    params_init = perturb_params(params_true, key_init, scale=0.2)
    
    batch_obs = jnp.expand_dims(obs, axis=0)
    
    # Run EM
    params_final, log_probs = ctds.fit_em(
        params_init,
        batch_obs,
        num_iters=5,
        verbose=False
    )
    
    # Check no NaNs
    assert not jnp.any(jnp.isnan(params_final.emissions.cov)), "R contains NaN"
    assert not jnp.any(jnp.isnan(params_final.emissions.weights)), "C contains NaN"
    
    # Check R diagonal has minimum
    R_diag = jnp.diag(params_final.emissions.cov)
    assert jnp.all(R_diag >= 1e-4), f"R diagonal too small: min={jnp.min(R_diag)}"


def test_all_excitatory():
    """Test with all excitatory cell types."""
    D = 6
    N = 15
    T = 100
    
    # All excitatory
    cell_types = jnp.array([0])
    cell_sign = jnp.array([1])
    cell_type_dimensions = jnp.array([6])
    cell_type_mask = jnp.zeros(N, dtype=jnp.int32)
    
    key = jax.random.PRNGKey(333)
    
    params_true, latent, obs = generate_synthetic_ssm(
        D, N, T,
        cell_types, cell_sign, cell_type_dimensions, cell_type_mask,
        key
    )
    
    ctds = CTDS(
        emission_dim=N,
        cell_types=cell_types,
        cell_sign=cell_sign,
        cell_type_dimensions=cell_type_dimensions,
        cell_type_mask=cell_type_mask,
        state_dim=D
    )
    
    key_init = jax.random.PRNGKey(444)
    params_init = perturb_params(params_true, key_init, scale=0.3)
    
    batch_obs = jnp.expand_dims(obs, axis=0)
    
    # Run EM
    params_final, log_probs = ctds.fit_em(
        params_init,
        batch_obs,
        num_iters=15,
        verbose=False
    )
    
    # Check constraints
    A = params_final.dynamics.weights
    D_dim = A.shape[0]
    
    # All off-diagonal entries should be non-negative
    for i in range(D_dim):
        for j in range(D_dim):
            if i != j:
                assert A[i, j] >= -1e-5, f"A[{i},{j}] is negative in all-excitatory case"
    
    # Check no NaNs
    assert not jnp.any(jnp.isnan(jnp.array(log_probs))), "NaN with all excitatory"


def test_all_inhibitory():
    """Test with all inhibitory cell types."""
    D = 6
    N = 15
    T = 100
    
    # All inhibitory
    cell_types = jnp.array([0])
    cell_sign = jnp.array([-1])
    cell_type_dimensions = jnp.array([6])
    cell_type_mask = jnp.zeros(N, dtype=jnp.int32)
    
    key = jax.random.PRNGKey(555)
    
    params_true, latent, obs = generate_synthetic_ssm(
        D, N, T,
        cell_types, cell_sign, cell_type_dimensions, cell_type_mask,
        key
    )
    
    ctds = CTDS(
        emission_dim=N,
        cell_types=cell_types,
        cell_sign=cell_sign,
        cell_type_dimensions=cell_type_dimensions,
        cell_type_mask=cell_type_mask,
        state_dim=D
    )
    
    key_init = jax.random.PRNGKey(666)
    params_init = perturb_params(params_true, key_init, scale=0.3)
    
    batch_obs = jnp.expand_dims(obs, axis=0)
    
    # Run EM
    params_final, log_probs = ctds.fit_em(
        params_init,
        batch_obs,
        num_iters=15,
        verbose=False
    )
    
    # Check constraints
    A = params_final.dynamics.weights
    D_dim = A.shape[0]
    
    # All off-diagonal entries should be non-positive
    for i in range(D_dim):
        for j in range(D_dim):
            if i != j:
                assert A[i, j] <= 1e-5, f"A[{i},{j}] is positive in all-inhibitory case"
    
    # Check no NaNs
    assert not jnp.any(jnp.isnan(jnp.array(log_probs))), "NaN with all inhibitory"


def test_mixed_cell_types():
    """Test with multiple mixed cell types."""
    D = 9
    N = 25
    T = 120
    
    # 3 cell types: exc, inh, exc
    cell_types = jnp.array([0, 1, 2])
    cell_sign = jnp.array([1, -1, 1])
    cell_type_dimensions = jnp.array([3, 3, 3])
    
    # Mix neurons across types
    cell_type_mask = jnp.array([0]*8 + [1]*9 + [2]*8, dtype=jnp.int32)
    
    key = jax.random.PRNGKey(777)
    
    params_true, latent, obs = generate_synthetic_ssm(
        D, N, T,
        cell_types, cell_sign, cell_type_dimensions, cell_type_mask,
        key
    )
    
    ctds = CTDS(
        emission_dim=N,
        cell_types=cell_types,
        cell_sign=cell_sign,
        cell_type_dimensions=cell_type_dimensions,
        cell_type_mask=cell_type_mask,
        state_dim=D
    )
    
    key_init = jax.random.PRNGKey(888)
    params_init = perturb_params(params_true, key_init, scale=0.3)
    
    batch_obs = jnp.expand_dims(obs, axis=0)
    
    # Run EM
    params_final, log_probs = ctds.fit_em(
        params_init,
        batch_obs,
        num_iters=20,
        verbose=False
    )
    
    # Check Dale constraints on each block
    A = params_final.dynamics.weights
    dynamics_mask = params_final.dynamics.dynamics_mask
    
    from tests.test_helpers import assert_dale_columns
    assert_dale_columns(A, dynamics_mask, tol=1e-5)
    
    # Check monotonicity
    for i in range(1, len(log_probs)):
        assert log_probs[i] >= log_probs[i-1] - 1e-3, \
            f"LL decreased with mixed types at iteration {i}"


def test_P_A_conditioning():
    """Test that P_A matrix is well-conditioned."""
    D = 5
    N = 12
    T = 100
    
    cell_types = jnp.array([0, 1])
    cell_sign = jnp.array([1, -1])
    cell_type_dimensions = jnp.array([3, 2])
    cell_type_mask = jnp.concatenate([
        jnp.zeros(6, dtype=jnp.int32),
        jnp.ones(6, dtype=jnp.int32)
    ])
    
    key = jax.random.PRNGKey(999)
    
    params_true, latent, obs = generate_synthetic_ssm(
        D, N, T,
        cell_types, cell_sign, cell_type_dimensions, cell_type_mask,
        key
    )
    
    # Compute P_A directly
    Q = params_true.dynamics.cov
    latent_second_moment = jnp.einsum('ti,tj->tij', latent, latent)
    Mt_1 = jnp.sum(latent_second_moment[:-1], axis=0) + 1e-6 * jnp.eye(D)
    
    Qinv = jnp.linalg.inv(Q)
    alpha = jnp.max(jnp.abs(Qinv))
    Qtil = Qinv / alpha
    L = jnp.linalg.cholesky(Qtil)
    
    P_A = 2.0 * jnp.kron(Mt_1, (L.T @ L))
    
    # Check conditioning
    cond_P_A = jnp.linalg.cond(P_A)
    print(f"\nCondition number of P_A: {cond_P_A:.2e}")
    
    # Should be well-conditioned
    assert cond_P_A < 1e10, f"P_A is ill-conditioned: cond={cond_P_A}"
    
    # Check no NaNs or Infs in P_A
    assert not jnp.any(jnp.isnan(P_A)), "P_A contains NaN"
    assert not jnp.any(jnp.isinf(P_A)), "P_A contains Inf"


def test_very_small_T():
    """Test stability with T=3 (minimum for dynamics)."""
    D = 3
    N = 8
    T = 3  # Minimum
    
    cell_types = jnp.array([0, 1])
    cell_sign = jnp.array([1, -1])
    cell_type_dimensions = jnp.array([2, 1])
    cell_type_mask = jnp.concatenate([
        jnp.zeros(4, dtype=jnp.int32),
        jnp.ones(4, dtype=jnp.int32)
    ])
    
    key = jax.random.PRNGKey(1111)
    
    params_true, latent, obs = generate_synthetic_ssm(
        D, N, T,
        cell_types, cell_sign, cell_type_dimensions, cell_type_mask,
        key
    )
    
    ctds = CTDS(
        emission_dim=N,
        cell_types=cell_types,
        cell_sign=cell_sign,
        cell_type_dimensions=cell_type_dimensions,
        cell_type_mask=cell_type_mask,
        state_dim=D
    )
    
    key_init = jax.random.PRNGKey(2222)
    params_init = perturb_params(params_true, key_init, scale=0.2)
    
    batch_obs = jnp.expand_dims(obs, axis=0)
    
    # Run EM - should not crash despite T=3
    params_final, log_probs = ctds.fit_em(
        params_init,
        batch_obs,
        num_iters=5,
        verbose=False
    )
    
    # Check no NaNs
    assert not jnp.any(jnp.isnan(jnp.array(log_probs))), "NaN with T=3"
    assert not jnp.any(jnp.isnan(params_final.dynamics.weights)), "A NaN with T=3"
    assert not jnp.any(jnp.isnan(params_final.emissions.weights)), "C NaN with T=3"
    
    print(f"\nSuccessfully ran EM with T={T}")


def test_regularization_prevents_singularity():
    """Test that regularization prevents matrices from becoming singular."""
    D = 4
    N = 10
    T = 40
    
    cell_types = jnp.array([0, 1])
    cell_sign = jnp.array([1, -1])
    cell_type_dimensions = jnp.array([2, 2])
    cell_type_mask = jnp.concatenate([
        jnp.zeros(5, dtype=jnp.int32),
        jnp.ones(5, dtype=jnp.int32)
    ])
    
    key = jax.random.PRNGKey(3333)
    
    params_true, latent, obs = generate_synthetic_ssm(
        D, N, T,
        cell_types, cell_sign, cell_type_dimensions, cell_type_mask,
        key,
        Q_scale=0.01,  # Small noise
        R_scale=0.01
    )
    
    ctds = CTDS(
        emission_dim=N,
        cell_types=cell_types,
        cell_sign=cell_sign,
        cell_type_dimensions=cell_type_dimensions,
        cell_type_mask=cell_type_mask,
        state_dim=D
    )
    
    key_init = jax.random.PRNGKey(4444)
    params_init = perturb_params(params_true, key_init, scale=0.2)
    
    batch_obs = jnp.expand_dims(obs, axis=0)
    
    # Run EM
    params_final, log_probs = ctds.fit_em(
        params_init,
        batch_obs,
        num_iters=10,
        verbose=False
    )
    
    # Check Q and R are invertible
    cond_Q = jnp.linalg.cond(params_final.dynamics.cov)
    cond_R = jnp.linalg.cond(params_final.emissions.cov)
    
    print(f"\nFinal Q condition number: {cond_Q:.2e}")
    print(f"Final R condition number: {cond_R:.2e}")
    
    # Should not be near-singular
    assert cond_Q < 1e10, f"Q is near-singular: cond={cond_Q}"
    assert cond_R < 1e10, f"R is near-singular: cond={cond_R}"
    
    # Check minimum eigenvalues
    Q_eigenvals = jnp.linalg.eigvalsh(params_final.dynamics.cov)
    R_eigenvals = jnp.linalg.eigvalsh(params_final.emissions.cov)
    
    print(f"Min Q eigenvalue: {jnp.min(Q_eigenvals):.2e}")
    print(f"Min R eigenvalue: {jnp.min(R_eigenvals):.2e}")
    
    assert jnp.min(Q_eigenvals) > 1e-6, "Q has too small eigenvalue"
    assert jnp.min(R_eigenvals) > 1e-4, "R has too small eigenvalue"
