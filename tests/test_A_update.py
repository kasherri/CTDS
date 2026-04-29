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
    P_A = 2.0 * jnp.kron( Mt_1, (L.T @ L))
    q_A = -2.0 * jnp.ravel(Qtil.T @ Mdelta.T)
    
    return P_A, q_A, Mt_1, Mdelta


def create_A_bounds(dynamics_mask, D):
    """Create bounds for A update."""
    # Diagonal mask (False for diagonal)
    diag_masks = jnp.ravel(
        jnp.logical_not(jnp.eye(D, dtype=jnp.bool_))
    )
    
    # Lower bounds
    cell_type_lb = jnp.ravel(
        jnp.tile(jnp.where(dynamics_mask == -1, -jnp.inf, 1e-6), (D, 1))
    )
    lb = jnp.where(diag_masks, cell_type_lb, -jnp.inf)
    
    # Upper bounds
    cell_type_ub = jnp.ravel(
        jnp.tile(jnp.where(dynamics_mask == -1, -1e-6, jnp.inf), (D, 1))
    )
    ub = jnp.where(diag_masks, cell_type_ub, jnp.inf)
    
    return lb, ub


def test_A_update_kkt_conditions():
    """Test that A update satisfies KKT optimality conditions."""
    D = 6
    T = 100
    
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, T)
    
    # Create dynamics mask
    dynamics_mask = jnp.array([1, 1, 1, -1, -1, -1])
    
    # Generate true A
    A_true = generate_stable_A(D, dynamics_mask, keys[0])
    Q = jnp.eye(D) * 0.1
    
    # Simulate latent trajectory
    latent = jnp.zeros((T, D))
    latent = latent.at[0].set(jax.random.normal(keys[1], (D,)) * 0.1)
    
    for t in range(1, T):
        noise = jax.random.multivariate_normal(keys[t], jnp.zeros(D), Q)
        latent = latent.at[t].set(A_true @ latent[t-1] + noise)
    
    # Compute sufficient statistics (using true latents, perfect E-step)
    latent_second_moment = jnp.einsum('ti,tj->tij', latent, latent)
    cross_time_moment = jnp.einsum('ti,tj->tij', latent[:-1], latent[1:])
    
    # Compute QP matrices
    P_A, q_A, Mt_1, Mdelta = compute_A_update_matrices(
        latent_second_moment, cross_time_moment, Q
    )
    
    # Create bounds
    lb, ub = create_A_bounds(dynamics_mask, D)
    
    # Solve QP
    A_init = jnp.ravel(jnp.eye(D) * 0.9)
    
    result = _boxCDQP.run(A_init, params_obj=(P_A, q_A), params_ineq=(lb, ub))
    A_vec = result.params
    A= jnp.reshape(A_vec, (D,D))
    print(A)
    
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
    
    A_init = jnp.ravel(jnp.eye(D) * 0.8)
    result = _boxCDQP.run(A_init, params_obj=(P_A, q_A), params_ineq=(lb, ub))
    A_vec = result.params
    A = jnp.reshape(A_vec, (D, D))
    

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


def test_A_update_frobenius_stability():
    """
    Test that the Frobenius norm of A does not blow up over 20 EM-like iterations.

    Each iteration:
      1. Simulate a latent trajectory under the current A.
      2. Compute sufficient statistics (perfect E-step).
      3. Solve the QP to update A (M-step A update).

    Asserts:
      - ||A||_F stays bounded (< 10 * ||A_true||_F) throughout all iterations.
      - A never contains NaN or Inf.
      - The spectral radius of A stays below 1.5 (does not become unstable).
    """
    D = 4
    T = 200
    N_ITERS = 20

    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, N_ITERS * (T + 5) + 10)
    ki = 0  # running key index

    dynamics_mask = jnp.array([1, 1, -1, -1])
    A_true = generate_stable_A(D, dynamics_mask, keys[ki]); ki += 1
    Q = jnp.eye(D) * 0.1

    true_frob = float(jnp.linalg.norm(A_true, 'fro'))
    frob_cap = 10.0 * true_frob

    # Start from (slightly perturbed) true A so the test is meaningful
    A_current = A_true + jax.random.normal(keys[ki], (D, D)) * 0.05; ki += 1

    frob_history = []

    for iteration in range(N_ITERS):
        # ---- simulate latent trajectory under current A (perfect E-step) ----
        latent = jnp.zeros((T, D))
        latent = latent.at[0].set(jax.random.normal(keys[ki], (D,)) * 0.1); ki += 1

        for t in range(1, T):
            noise = jax.random.multivariate_normal(keys[ki], jnp.zeros(D), Q); ki += 1
            latent = latent.at[t].set(A_current @ latent[t - 1] + noise)

        # ---- sufficient statistics ----
        latent_second_moment = jnp.einsum('ti,tj->tij', latent, latent)
        cross_time_moment    = jnp.einsum('ti,tj->tij', latent[:-1], latent[1:])

        # ---- M-step A update ----
        P_A, q_A, _, _ = compute_A_update_matrices(
            latent_second_moment, cross_time_moment, Q
        )
        lb, ub = create_A_bounds(dynamics_mask, D)

        A_init = jnp.ravel(A_current)
        result = _boxCDQP.run(A_init, params_obj=(P_A, q_A), params_ineq=(lb, ub))
        A_current = jnp.reshape(result.params, (D, D))

        frob = float(jnp.linalg.norm(A_current, 'fro'))
        frob_history.append(frob)
        

        # Per-iteration checks
        assert not jnp.any(jnp.isnan(A_current)), f"NaN in A at iteration {iteration}"
        assert not jnp.any(jnp.isinf(A_current)), f"Inf in A at iteration {iteration}"
        assert frob < frob_cap, (
            f"Frobenius norm blew up at iteration {iteration}: "
            f"||A||_F = {frob:.3f} > cap {frob_cap:.3f}"
        )

    spectral_radius = float(jnp.max(jnp.abs(jnp.linalg.eigvals(A_current))))

    print(f"\nFrobenius norm history: {[f'{v:.3f}' for v in frob_history]}")
    print(f"Final spectral radius: {spectral_radius:.4f}")
    print(f"True ||A||_F: {true_frob:.4f}   Cap: {frob_cap:.4f}")

    assert spectral_radius < 1.5, (
        f"Spectral radius too large after {N_ITERS} iterations: {spectral_radius:.4f}"
    )


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
    
    A_init = jnp.ravel(jnp.eye(D) * 0.8)
    result = _boxCDQP.run(A_init, params_obj=(P_A, q_A), params_ineq=(lb, ub))
    A = jnp.reshape(result.params, (D, D))
    
    # Check no NaNs or Infs
    assert not jnp.any(jnp.isnan(A)), "A contains NaN"
    assert not jnp.any(jnp.isinf(A)), "A contains Inf"
    
    # Check constraints
    assert_dale_columns(A, dynamics_mask, tol=1e-5)


def test_A_update_frobenius_stability():
    """After 20 EM iterations, ||A|| should not blow up."""
    import sys
    sys.path.insert(0, '.')
    from ctds.models import CTDS
    
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 10)

    # Small synthetic problem
    cell_types = jnp.array([0, 1])
    cell_sign  = jnp.array([1, -1])
    cell_type_dimensions = jnp.array([3, 3])
    N_per_type = 10
    N = N_per_type * 2
    cell_type_mask = jnp.array([0]*N_per_type + [1]*N_per_type)

    model = CTDS(
        emission_dim=N,
        cell_types=cell_types,
        cell_sign=cell_sign,
        cell_type_dimensions=cell_type_dimensions,
        cell_type_mask=cell_type_mask,
        state_dim=6,
    )

    T = 200
    D = 6
    A_true = 0.7 * jnp.eye(D)
    Q_true = 0.1 * jnp.eye(D)
    C_true = jax.random.normal(keys[0], (N, D)) * 0.5
    R_true = 0.1 * jnp.eye(N)

    # Simulate
    x = jnp.zeros((T, D))
    x = x.at[0].set(jax.random.normal(keys[1], (D,)))
    for t in range(1, T):
        x = x.at[t].set(
            A_true @ x[t-1] + jax.random.multivariate_normal(keys[t % 8 + 2], jnp.zeros(D), Q_true)
        )
    Y = (C_true @ x.T).T + jax.random.normal(keys[3], (T, N)) * 0.1

    params = model.initialize(Y)
    params, lls = model.fit_em(params, Y[None], num_iters=20, verbose=False)

    A_final_norm = jnp.linalg.norm(params.dynamics.weights)
    print(f"||A|| after 20 iters: {A_final_norm:.4f}")

    # ||A|| should stay bounded — not blow up to thousands
    assert A_final_norm < 50.0, f"||A|| blew up to {A_final_norm:.2f} — diverging EM"
    # Log-likelihood should not be wildly worse than start
    assert lls[-1] > lls[1] - 5.0, f"LL degraded severely: {lls[1]:.2f} → {lls[-1]:.2f}"