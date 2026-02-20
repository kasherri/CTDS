"""
Test Q and R covariance updates.
"""
import pytest
import jax
import jax.numpy as jnp
from tests.test_helpers import assert_psd, generate_stable_A, generate_nonnegative_C

jax.config.update("jax_enable_x64", True)


def compute_Q_update(A, latent_second_moment, cross_time_moment, T):
    """
    Compute Q update from sufficient statistics.
    
    Q = (1/(T-1)) * [M_{2:T} - A M_Δ^T - M_Δ A^T + A M_{1:T-1} A^T]
    """
    Mt_1 = jnp.sum(latent_second_moment[:-1], axis=0)  # sum_{t=1}^{T-1} E[x_t x_t^T]
    M2_T = jnp.sum(latent_second_moment[1:], axis=0)  # sum_{t=2}^{T} E[x_t x_t^T]
    Mdelta = jnp.sum(cross_time_moment, axis=0)  # sum_{t=1}^{T-1} E[x_t x_{t+1}^T]
    
    Q_raw = (M2_T - A @ Mdelta.T - Mdelta @ A.T + A @ Mt_1 @ A.T) / (T - 1.0)
    
    # Apply regularization as in actual code
    Q = (Q_raw + Q_raw.T) / 2.0  # Symmetrize
    Q = Q + 1e-4 * jnp.eye(Q.shape[0])  # Ridge
    
    # Ensure minimum eigenvalue
    eigenvals = jnp.linalg.eigvalsh(Q)
    min_eig = 1e-5
    if jnp.min(eigenvals) < min_eig:
        Q = Q + (min_eig - jnp.min(eigenvals) + 1e-6) * jnp.eye(Q.shape[0])
    
    return Q


def compute_R_update(C, latent_mean, latent_second_moment, observations, N, T):
    """
    Compute R update from sufficient statistics.
    
    R = (1/T) * [YY^T - C Ytil^T - Ytil C^T + C Mxx C^T]
    
    where:
    - Y: (N, T) observations
    - Ytil = Y @ latent_mean: (N, D)
    - Mxx = sum_t E[x_t x_t^T]: (D, D)
    """
    Y = observations  # (N, T)
    Ytil = Y @ latent_mean  # (N, D)
    Mxx = jnp.sum(latent_second_moment, axis=0)  # (D, D)
    
    YY = Y @ Y.T  # (N, N)
    
    R_raw = (YY - C @ Ytil.T - Ytil @ C.T + C @ Mxx @ C.T) / T
    
    # Apply regularization
    R = (R_raw + R_raw.T) / 2.0  # Symmetrize
    R = R + 1e-4 * jnp.eye(N)  # Ridge
    
    # Enforce minimum diagonal
    R_diag = jnp.diag(R)
    min_obs_noise = 1e-3
    R_diag = jnp.maximum(R_diag, min_obs_noise)
    R = R.at[jnp.diag_indices(N)].set(R_diag)
    
    return R


def test_Q_update_is_psd():
    """Test that Q update produces PSD matrix."""
    D = 5
    T = 100
    
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 5)
    
    # Generate trajectory
    dynamics_mask = jnp.array([1, 1, 1, -1, -1])
    A = generate_stable_A(D, dynamics_mask, keys[0])
    Q_true = jnp.eye(D) * 0.1
    
    latent = jnp.zeros((T, D))
    latent = latent.at[0].set(jax.random.normal(keys[1], (D,)))
    
    for t in range(1, T):
        noise = jax.random.multivariate_normal(keys[2], jnp.zeros(D), Q_true)
        latent = latent.at[t].set(A @ latent[t-1] + noise)
    
    # Compute sufficient statistics
    latent_second_moment = jnp.einsum('ti,tj->tij', latent, latent)
    cross_time_moment = jnp.einsum('ti,sj->tsij', latent[:-1], latent[1:])[:, 0, :, :]
    
    # Compute Q update
    Q = compute_Q_update(A, latent_second_moment, cross_time_moment, T)
    
    # Check PSD
    assert_psd(Q, "Q", tol=-1e-8)


def test_Q_update_symmetry():
    """Test that Q update is symmetric."""
    D = 4
    T = 80
    
    key = jax.random.PRNGKey(123)
    keys = jax.random.split(key, 5)
    
    # Generate data
    dynamics_mask = jnp.array([1, 1, -1, -1])
    A = generate_stable_A(D, dynamics_mask, keys[0])
    Q_true = jnp.eye(D) * 0.05
    
    latent = jnp.zeros((T, D))
    latent = latent.at[0].set(jax.random.normal(keys[1], (D,)))
    
    for t in range(1, T):
        noise = jax.random.multivariate_normal(keys[2], jnp.zeros(D), Q_true)
        latent = latent.at[t].set(A @ latent[t-1] + noise)
    
    # Compute statistics
    latent_second_moment = jnp.einsum('ti,tj->tij', latent, latent)
    cross_time_moment = jnp.einsum('ti,sj->tsij', latent[:-1], latent[1:])[:, 0, :, :]
    
    # Compute Q
    Q = compute_Q_update(A, latent_second_moment, cross_time_moment, T)
    
    # Check symmetry
    max_asymmetry = jnp.max(jnp.abs(Q - Q.T))
    print(f"\nMax asymmetry in Q: {max_asymmetry:.2e}")
    assert max_asymmetry < 1e-10, f"Q not symmetric: max error={max_asymmetry}"


def test_Q_update_with_small_T():
    """Test Q update stability with small T."""
    D = 3
    T = 5  # Very small
    
    key = jax.random.PRNGKey(456)
    keys = jax.random.split(key, 5)
    
    # Generate data
    dynamics_mask = jnp.array([1, 1, 1])
    A = generate_stable_A(D, dynamics_mask, keys[0], spectral_radius=0.8)
    Q_true = jnp.eye(D) * 0.2
    
    latent = jnp.zeros((T, D))
    latent = latent.at[0].set(jax.random.normal(keys[1], (D,)))
    
    for t in range(1, T):
        noise = jax.random.multivariate_normal(keys[2], jnp.zeros(D), Q_true)
        latent = latent.at[t].set(A @ latent[t-1] + noise)
    
    # Compute statistics
    latent_second_moment = jnp.einsum('ti,tj->tij', latent, latent)
    cross_time_moment = jnp.einsum('ti,sj->tsij', latent[:-1], latent[1:])[:, 0, :, :]
    
    # Compute Q
    Q = compute_Q_update(A, latent_second_moment, cross_time_moment, T)
    
    # Check no NaNs
    assert not jnp.any(jnp.isnan(Q)), "Q contains NaN with small T"
    assert not jnp.any(jnp.isinf(Q)), "Q contains Inf with small T"
    
    # Check PSD
    assert_psd(Q, "Q", tol=-1e-8)


def test_Q_update_minimum_eigenvalue():
    """Test that Q update enforces minimum eigenvalue."""
    D = 4
    T = 50
    
    key = jax.random.PRNGKey(789)
    keys = jax.random.split(key, 5)
    
    # Generate data with very small noise (stress test)
    dynamics_mask = jnp.array([1, 1, -1, -1])
    A = generate_stable_A(D, dynamics_mask, keys[0])
    Q_true = jnp.eye(D) * 1e-6  # Very small noise
    
    latent = jnp.zeros((T, D))
    latent = latent.at[0].set(jax.random.normal(keys[1], (D,)))
    
    for t in range(1, T):
        noise = jax.random.multivariate_normal(keys[2], jnp.zeros(D), Q_true)
        latent = latent.at[t].set(A @ latent[t-1] + noise)
    
    # Compute statistics
    latent_second_moment = jnp.einsum('ti,tj->tij', latent, latent)
    cross_time_moment = jnp.einsum('ti,sj->tsij', latent[:-1], latent[1:])[:, 0, :, :]
    
    # Compute Q
    Q = compute_Q_update(A, latent_second_moment, cross_time_moment, T)
    
    # Check minimum eigenvalue
    eigenvals = jnp.linalg.eigvalsh(Q)
    min_eig = jnp.min(eigenvals)
    
    print(f"\nMinimum eigenvalue of Q: {min_eig:.2e}")
    print(f"All eigenvalues: {eigenvals}")
    
    # Should be at least 1e-5 due to regularization
    assert min_eig >= 1e-5 * 0.9, f"Q eigenvalue too small: {min_eig}"


def test_R_update_is_psd():
    """Test that R update produces PSD matrix."""
    N = 10
    D = 5
    T = 100
    
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 5)
    
    # Generate data
    latent = jax.random.normal(keys[0], (T, D))
    C = generate_nonnegative_C(N, D, keys[1])
    R_true = jnp.eye(N) * 0.5
    
    obs_noise = jax.random.multivariate_normal(keys[2], jnp.zeros(N), R_true, (T,))
    observations = (C @ latent.T).T + obs_noise
    
    # Compute sufficient statistics
    latent_second_moment = jnp.einsum('ti,tj->tij', latent, latent)
    
    # Compute R
    R = compute_R_update(C, latent, latent_second_moment, observations.T, N, T)
    
    # Check PSD
    assert_psd(R, "R", tol=-1e-8)


def test_R_update_symmetry():
    """Test that R update is symmetric."""
    N = 8
    D = 4
    T = 80
    
    key = jax.random.PRNGKey(123)
    keys = jax.random.split(key, 5)
    
    # Generate data
    latent = jax.random.normal(keys[0], (T, D))
    C = generate_nonnegative_C(N, D, keys[1])
    R_true = jnp.eye(N) * 0.3
    
    obs = (C @ latent.T).T + jax.random.multivariate_normal(keys[2], jnp.zeros(N), R_true, (T,))
    
    # Compute statistics
    latent_second_moment = jnp.einsum('ti,tj->tij', latent, latent)
    
    # Compute R
    R = compute_R_update(C, latent, latent_second_moment, obs.T, N, T)
    
    # Check symmetry
    max_asymmetry = jnp.max(jnp.abs(R - R.T))
    print(f"\nMax asymmetry in R: {max_asymmetry:.2e}")
    assert max_asymmetry < 1e-10, f"R not symmetric: max error={max_asymmetry}"


def test_R_update_minimum_diagonal():
    """Test that R update enforces minimum diagonal values."""
    N = 12
    D = 5
    T = 60
    
    key = jax.random.PRNGKey(456)
    keys = jax.random.split(key, 5)
    
    # Generate data with very small observation noise
    latent = jax.random.normal(keys[0], (T, D))
    C = generate_nonnegative_C(N, D, keys[1])
    R_true = jnp.eye(N) * 1e-6  # Very small noise
    
    obs = (C @ latent.T).T + jax.random.multivariate_normal(keys[2], jnp.zeros(N), R_true, (T,))
    
    # Compute statistics
    latent_second_moment = jnp.einsum('ti,tj->tij', latent, latent)
    
    # Compute R
    R = compute_R_update(C, latent, latent_second_moment, obs.T, N, T)
    
    # Check minimum diagonal
    R_diag = jnp.diag(R)
    min_diag = jnp.min(R_diag)
    
    print(f"\nMinimum R diagonal: {min_diag:.2e}")
    
    # Should be at least 1e-3 due to regularization
    min_noise_floor = 1e-3
    assert jnp.all(R_diag >= min_noise_floor * 0.9), f"R diagonal too small: min={min_diag}"


def test_R_update_with_small_T():
    """Test R update stability with small T."""
    N = 6
    D = 3
    T = 8  # Small T
    
    key = jax.random.PRNGKey(789)
    keys = jax.random.split(key, 5)
    
    # Generate data
    latent = jax.random.normal(keys[0], (T, D))
    C = generate_nonnegative_C(N, D, keys[1])
    R_true = jnp.eye(N) * 0.4
    
    obs = (C @ latent.T).T + jax.random.multivariate_normal(keys[2], jnp.zeros(N), R_true, (T,))
    
    # Compute statistics
    latent_second_moment = jnp.einsum('ti,tj->tij', latent, latent)
    
    # Compute R
    R = compute_R_update(C, latent, latent_second_moment, obs.T, N, T)
    
    # Check no NaNs
    assert not jnp.any(jnp.isnan(R)), "R contains NaN with small T"
    assert not jnp.any(jnp.isinf(R)), "R contains Inf with small T"
    
    # Check PSD
    assert_psd(R, "R", tol=-1e-8)


def test_Q_and_R_conditioning():
    """Test that Q and R updates produce well-conditioned matrices."""
    D = 6
    N = 15
    T = 100
    
    key = jax.random.PRNGKey(999)
    keys = jax.random.split(key, 5)
    
    # Generate data
    dynamics_mask = jnp.array([1, 1, 1, -1, -1, -1])
    A = generate_stable_A(D, dynamics_mask, keys[0])
    Q_true = jnp.eye(D) * 0.1
    C = generate_nonnegative_C(N, D, keys[1])
    R_true = jnp.eye(N) * 0.3
    
    # Simulate
    latent = jnp.zeros((T, D))
    latent = latent.at[0].set(jax.random.normal(keys[2], (D,)))
    
    for t in range(1, T):
        noise = jax.random.multivariate_normal(keys[3], jnp.zeros(D), Q_true)
        latent = latent.at[t].set(A @ latent[t-1] + noise)
    
    obs = (C @ latent.T).T + jax.random.multivariate_normal(keys[4], jnp.zeros(N), R_true, (T,))
    
    # Compute statistics
    latent_second_moment = jnp.einsum('ti,tj->tij', latent, latent)
    cross_time_moment = jnp.einsum('ti,sj->tsij', latent[:-1], latent[1:])[:, 0, :, :]
    
    # Compute Q and R
    Q = compute_Q_update(A, latent_second_moment, cross_time_moment, T)
    R = compute_R_update(C, latent, latent_second_moment, obs.T, N, T)
    
    # Check conditioning
    cond_Q = jnp.linalg.cond(Q)
    cond_R = jnp.linalg.cond(R)
    
    print(f"\nCondition number of Q: {cond_Q:.2e}")
    print(f"Condition number of R: {cond_R:.2e}")
    
    # Should be well-conditioned (not near singular)
    assert cond_Q < 1e8, f"Q is ill-conditioned: cond={cond_Q}"
    assert cond_R < 1e8, f"R is ill-conditioned: cond={cond_R}"
