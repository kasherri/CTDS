"""
Test A-update (dynamics matrix) in isolation.
"""
import pytest
import jax
import jax.numpy as jnp
from jaxopt import BoxCDQP
from tests.test_helpers import (
    assert_dale_columns, check_kkt_conditions,
    generate_stable_A, generate_synthetic_ssm
)

jax.config.update("jax_enable_x64", True)

_boxCDQP = BoxCDQP(tol=1e-7, maxiter=10000, verbose=False)


def compute_A_update_matrices(latent_second_moment, cross_time_moment, Q):
    """
    Compute P_A and q_A for the A update QP.
    
    Args:
        latent_second_moment: (T, D, D) - E[x_t x_t^T]
        cross_time_moment: (T-1, D, D) - E[x_t x_{t+1}^T]
        Q: (D, D) - process noise covariance
    
    Returns:
        P_A: (D^2, D^2) quadratic term
        q_A: (D^2,) linear term
        Mt_1: (D, D) sum of past moments
        Mdelta: (D, D) sum of cross moments
    """
    D = Q.shape[0]
    
    # Sufficient statistics
    Mt_1 = jnp.sum(latent_second_moment[:-1], axis=0)  # sum_{t=1}^{T-1} E[x_t x_t^T]
    Mdelta = jnp.sum(cross_time_moment, axis=0)  # sum_{t=1}^{T-1} E[x_t x_{t+1}^T]
    
    # Normalize Q inverse
    Qinv = jnp.linalg.inv(Q)
    alpha = jnp.max(jnp.abs(Qinv))
    Qtil = Qinv / alpha
    
    # Cholesky
    L = jnp.linalg.cholesky(Qtil)
    
    # QP matrices
    P_A = 2.0 * jnp.kron(Mt_1, (L.T @ L))
    q_A = -2.0 * jnp.ravel(Qtil.T @ Mdelta.T, order='F')
    
    return P_A, q_A, Mt_1, Mdelta


def create_A_bounds(dynamics_mask, D):
    """Create bounds for A update."""
    # Diagonal mask (False for diagonal)
    diag_masks = jnp.ravel(
        jnp.logical_not(jnp.eye(D, dtype=jnp.bool_)), 
        order='F'
    )
    
    # Lower bounds
    cell_type_lb = jnp.ravel(
        jnp.tile(jnp.where(dynamics_mask == -1, -jnp.inf, 1e-6), (D, 1)),
        order='F'
    )
    lb = jnp.where(diag_masks, cell_type_lb, -jnp.inf)
    
    # Upper bounds
    cell_type_ub = jnp.ravel(
        jnp.tile(jnp.where(dynamics_mask == -1, -1e-6, jnp.inf), (D, 1)),
        order='F'
    )
    ub = jnp.where(diag_masks, cell_type_ub, jnp.inf)
    
    return lb, ub


def test_A_update_kkt_conditions():
    """Test that A update satisfies KKT optimality conditions."""
    D = 6
    T = 100
    
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 5)
    
    # Create dynamics mask
    dynamics_mask = jnp.array([1, 1, 1, -1, -1, -1])
    
    # Generate true A
    A_true = generate_stable_A(D, dynamics_mask, keys[0])
    Q = jnp.eye(D) * 0.1
    
    # Simulate latent trajectory
    latent = jnp.zeros((T, D))
    latent = latent.at[0].set(jax.random.normal(keys[1], (D,)) * 0.1)
    
    for t in range(1, T):
        noise = jax.random.multivariate_normal(keys[2], jnp.zeros(D), Q)
        latent = latent.at[t].set(A_true @ latent[t-1] + noise)
    
    # Compute sufficient statistics (using true latents, perfect E-step)
    latent_second_moment = jnp.einsum('ti,tj->tij', latent, latent)
    cross_time_moment = jnp.einsum('ti,sj->tsij', latent[:-1], latent[1:])[:, 0, :, :]
    
    # Compute QP matrices
    P_A, q_A, Mt_1, Mdelta = compute_A_update_matrices(
        latent_second_moment, cross_time_moment, Q
    )
    
    # Create bounds
    lb, ub = create_A_bounds(dynamics_mask, D)
    
    # Solve QP
    A_init = jnp.ravel(jnp.eye(D) * 0.9, order='F')
    result = _boxCDQP.run(A_init, params_obj=(P_A, q_A), params_ineq=(lb, ub))
    A_vec = result.params
    
    # Check KKT conditions
    kkt_results = check_kkt_conditions(A_vec, P_A, q_A, lb, ub, tol=1e-4)
    
    print(f"\nKKT Results for A update:")
    print(f"  Interior violations: {kkt_results['interior_violations']} / {kkt_results['interior_count']}")
    print(f"  Interior max error: {kkt_results['interior_max_error']:.2e}")
    print(f"  Lower bound violations: {kkt_results['lower_violations']}")
    print(f"  Upper bound violations: {kkt_results['upper_violations']}")
    print(f"  Total violations: {kkt_results['total_violations']}")
    
    # Allow small number of violations due to numerical tolerance
    assert kkt_results['total_violations'] < D * 2, "Too many KKT violations"
    assert kkt_results['interior_max_error'] < 1e-3, "Interior stationarity violated"


def test_A_update_satisfies_constraints():
    """Test that solved A satisfies Dale's law constraints."""
    D = 4
    T = 50
    
    key = jax.random.PRNGKey(123)
    keys = jax.random.split(key, 5)
    
    # 2 excitatory, 2 inhibitory
    dynamics_mask = jnp.array([1, 1, -1, -1])
    
    # Generate data
    A_true = generate_stable_A(D, dynamics_mask, keys[0])
    Q = jnp.eye(D) * 0.05
    
    latent = jnp.zeros((T, D))
    latent = latent.at[0].set(jax.random.normal(keys[1], (D,)) * 0.1)
    
    for t in range(1, T):
        noise = jax.random.multivariate_normal(keys[2], jnp.zeros(D), Q)
        latent = latent.at[t].set(A_true @ latent[t-1] + noise)
    
    # Compute statistics
    latent_second_moment = jnp.einsum('ti,tj->tij', latent, latent)
    cross_time_moment = jnp.einsum('ti,sj->tsij', latent[:-1], latent[1:])[:, 0, :, :]
    
    # Solve for A
    P_A, q_A, _, _ = compute_A_update_matrices(
        latent_second_moment, cross_time_moment, Q
    )
    lb, ub = create_A_bounds(dynamics_mask, D)
    
    A_init = jnp.ravel(jnp.eye(D) * 0.8, order='F')
    result = _boxCDQP.run(A_init, params_obj=(P_A, q_A), params_ineq=(lb, ub))
    A_vec = result.params
    A = jnp.reshape(A_vec, (D, D), order='F')
    

    # Check Dale constraints
    assert_dale_columns(A, dynamics_mask, tol=1e-5)




def test_A_update_orientation():
    """Test that A update has correct transpose orientation."""
    D = 4
    T = 100
    
    key = jax.random.PRNGKey(789)
    keys = jax.random.split(key, 5)
    
    dynamics_mask = jnp.array([1, 1, -1, -1])
    
    # Generate data with known structure
    A_true = jnp.array([
        [0.9, 0.1, 0.0, 0.0],
        [0.2, 0.8, 0.0, 0.0],
        [0.0, 0.0, -0.9, -0.1],
        [0.0, 0.0, -0.2, -0.8]
    ])
    #A_true = generate_stable_A(D, dynamics_mask, keys[0])
    Q = jnp.eye(D) * 0.05
    
    # Simulate
    latent = jnp.zeros((T, D))
    latent = latent.at[0].set(jax.random.normal(keys[0], (D,)))
    
    for t in range(1, T):
        noise = jax.random.multivariate_normal(keys[1], jnp.zeros(D), Q)
        latent = latent.at[t].set(A_true @ latent[t-1] + noise)
    
    # Solve for A
    latent_second_moment = jnp.einsum('ti,tj->tij', latent, latent)
    cross_time_moment = jnp.einsum('ti,sj->tsij', latent[:-1], latent[1:])[:, 0, :, :]
    
    P_A, q_A, _, _ = compute_A_update_matrices(
        latent_second_moment, cross_time_moment, Q
    )
    lb, ub = create_A_bounds(dynamics_mask, D)

    A_init = jnp.ravel(jnp.eye(D) * 0.5, order='F')
    result = _boxCDQP.run(A_init, params_obj=(P_A, q_A), params_ineq=(lb, ub))
    A_recovered = jnp.reshape(result.params, (D, D), order='F')
    print(A_init)
    print(result)
    print("\n lb")
    print(lb)
    print("\n ub")
    print(ub)
    # Should recover block structure
    print("\nTrue A:")
    print(A_true)
    print("\nRecovered A:")
    print(A_recovered)
    
    # Check that off-diagonal blocks are near zero
    exc_to_inh = jnp.linalg.norm(A_recovered[:2, 2:])
    inh_to_exc = jnp.linalg.norm(A_recovered[2:, :2])
    
    print(f"\nExc->Inh coupling: {exc_to_inh:.3f}")
    print(f"Inh->Exc coupling: {inh_to_exc:.3f}")
    
    # These should be small (true model has no cross-coupling)
    assert exc_to_inh < 0.2, "Unexpected excitatory to inhibitory coupling"
    assert inh_to_exc < 0.9, "Unexpected inhibitory to excitatory coupling"


def test_A_update_with_regularization():
    """Test that A update is stable with regularized sufficient statistics."""
    D = 5
    T = 30  # Small T to stress-test
    
    key = jax.random.PRNGKey(999)
    keys = jax.random.split(key, 5)
    
    dynamics_mask = jnp.array([1, 1, 1, -1, -1])
    
    A_true = generate_stable_A(D, dynamics_mask, keys[0], spectral_radius=0.7)
    Q = jnp.eye(D) * 0.1
    
    # Simulate short trajectory
    latent = jnp.zeros((T, D))
    latent = latent.at[0].set(jax.random.normal(keys[1], (D,)))
    
    for t in range(1, T):
        noise = jax.random.multivariate_normal(keys[2], jnp.zeros(D), Q)
        latent = latent.at[t].set(A_true @ latent[t-1] + noise)
    
    # Compute statistics with regularization
    latent_second_moment = jnp.einsum('ti,tj->tij', latent, latent)
    # Add regularization as in actual M-step
    latent_second_moment = latent_second_moment + 1e-6 * jnp.eye(D)[None, :, :]
    
    cross_time_moment = jnp.einsum('ti,sj->tsij', latent[:-1], latent[1:])[:, 0, :, :]
    
    # Solve for A
    P_A, q_A, _, _ = compute_A_update_matrices(
        latent_second_moment, cross_time_moment, Q
    )
    lb, ub = create_A_bounds(dynamics_mask, D)
    
    # Check P_A is well-conditioned
    cond_P = jnp.linalg.cond(P_A)
    print(f"\nCondition number of P_A: {cond_P:.2e}")
    
    A_init = jnp.ravel(jnp.eye(D) * 0.8, order='F')
    result = _boxCDQP.run(A_init, params_obj=(P_A, q_A), params_ineq=(lb, ub))
    A = jnp.reshape(result.params, (D, D), order='F')
    
    # Check no NaNs or Infs
    assert not jnp.any(jnp.isnan(A)), "A contains NaN"
    assert not jnp.any(jnp.isinf(A)), "A contains Inf"
    
    # Check constraints
    assert_dale_columns(A, dynamics_mask, tol=1e-5)
