"""
Test full EM algorithm convergence and parameter recovery.
"""
import pytest
import jax
import jax.numpy as jnp
from models import CTDS
from tests.test_helpers import (
    generate_synthetic_ssm, perturb_params,
    subspace_distance, assert_psd, assert_dale_columns, assert_nonnegative
)

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def parameter_recovery_problem():
    """Generate problem for parameter recovery test."""
    D = 6
    N = 20
    T = 500  # Long trajectory for better recovery
    
    # 2 cell types: excitatory and inhibitory
    cell_types = jnp.array([0, 1])
    cell_sign = jnp.array([1, -1])
    cell_type_dimensions = jnp.array([3, 3])
    cell_type_mask = jnp.concatenate([
        jnp.zeros(10, dtype=jnp.int32),
        jnp.ones(10, dtype=jnp.int32)
    ])
    
    key = jax.random.PRNGKey(42)
    
    # Generate with diagonal Q and R for cleaner recovery
    params_true, latent, obs = generate_synthetic_ssm(
        D, N, T,
        cell_types, cell_sign, cell_type_dimensions, cell_type_mask,
        key,
        Q_scale=0.1,
        R_scale=0.5
    )
    
    # Create model
    ctds = CTDS(
        emission_dim=N,
        cell_types=cell_types,
        cell_sign=cell_sign,
        cell_type_dimensions=cell_type_dimensions,
        cell_type_mask=cell_type_mask,
        state_dim=D
    )
    
    # Perturb for initialization
    key_init = jax.random.PRNGKey(123)
    params_init = perturb_params(params_true, key_init, scale=0.4)
    
    return {
        'ctds': ctds,
        'params_true': params_true,
        'params_init': params_init,
        'observations': obs,
        'latent_true': latent,
        'D': D,
        'N': N,
        'T': T
    }


def test_em_monotonicity(parameter_recovery_problem):
    """Test that EM log-likelihood increases monotonically."""
    prob = parameter_recovery_problem
    ctds = prob['ctds']
    params = prob['params_init']
    obs = prob['observations']
    
    # Prepare batch
    batch_obs = obs[None,:,:]  # (1, T, N)
    print(batch_obs.shape)
    
    # Run EM for 20 iterations
    params_final, log_probs = ctds.fit_em(
        params,
        batch_obs,
        num_iters=30,
        verbose=False
    )
    
    print(f"\nLog-likelihood trajectory:")
    for i, ll in enumerate(log_probs):
        print(f"  Iter {i}: {ll:.2f}")
    
    # Check monotonicity
    for i in range(1, len(log_probs)):
        assert log_probs[i] >= log_probs[i-1] - 1e-3, \
            f"LL decreased at iteration {i}: {log_probs[i-1]:.2f} -> {log_probs[i]:.2f}"
    
    # Check no NaNs
    assert not jnp.any(jnp.isnan(jnp.array(log_probs))), "NaN in log-likelihoods"
    
    # Check improvement
    improvement = log_probs[-1] - log_probs[0]
    print(f"\nTotal improvement: {improvement:.2f}")
    assert improvement > 0, "EM did not improve log-likelihood"


def test_em_no_nans(parameter_recovery_problem):
    """Test that EM never produces NaNs."""
    prob = parameter_recovery_problem
    ctds = prob['ctds']
    params = prob['params_init']
    obs = prob['observations']
    
    batch_obs = jnp.expand_dims(obs, axis=0)
    
    # Run EM
    params_final, log_probs = ctds.fit_em(
        params,
        batch_obs,
        num_iters=15,
        verbose=False
    )
    
    # Check no NaNs in parameters
    assert not jnp.any(jnp.isnan(params_final.dynamics.weights)), "A contains NaN"
    assert not jnp.any(jnp.isnan(params_final.dynamics.cov)), "Q contains NaN"
    assert not jnp.any(jnp.isnan(params_final.emissions.weights)), "C contains NaN"
    assert not jnp.any(jnp.isnan(params_final.emissions.cov)), "R contains NaN"
    
    # Check no Infs
    assert not jnp.any(jnp.isinf(params_final.dynamics.weights)), "A contains Inf"
    assert not jnp.any(jnp.isinf(params_final.dynamics.cov)), "Q contains Inf"
    assert not jnp.any(jnp.isinf(params_final.emissions.weights)), "C contains Inf"
    assert not jnp.any(jnp.isinf(params_final.emissions.cov)), "R contains Inf"


def test_em_maintains_constraints(parameter_recovery_problem):
    """Test that EM maintains all constraints throughout."""
    prob = parameter_recovery_problem
    ctds = prob['ctds']
    params = prob['params_init']
    obs = prob['observations']
    
    batch_obs = jnp.expand_dims(obs, axis=0)
    
    # Run EM
    params_final, _ = ctds.fit_em(
        params,
        batch_obs,
        num_iters=25,
        verbose=False
    )
    
    # Check Dale's law on A
    assert_dale_columns(
        params_final.dynamics.weights,
        params_final.dynamics.dynamics_mask,
        tol=1e-5
    )
    
    # Check C non-negativity
    assert_nonnegative(params_final.emissions.weights, "C", tol=-1e-6)
    
    # Check Q and R are PSD
    assert_psd(params_final.dynamics.cov, "Q", tol=-1e-8)
    assert_psd(params_final.emissions.cov, "R", tol=-1e-8)


def test_em_parameter_recovery(parameter_recovery_problem):
    """Test that EM recovers parameters up to reasonable error."""
    prob = parameter_recovery_problem
    ctds = prob['ctds']
    params_true = prob['params_true']
    params_init = prob['params_init']
    obs = prob['observations']
    
    batch_obs = jnp.expand_dims(obs, axis=0)
    
    # Run EM for many iterations
    params_final, log_probs = ctds.fit_em(
        params_init,
        batch_obs,
        num_iters=30,
        verbose=False
    )
    
    print(f"\nFinal log-likelihood: {log_probs[-1]:.2f}")
    
    # Check subspace recovery for C
    # (Can't compare matrices directly due to rotation/scaling ambiguity)
    C_true = params_true.emissions.weights
    C_final = params_final.emissions.weights
    
    # Compare column spaces via principal angles
    angle = subspace_distance(C_true, C_final)
    print(f"\nMax principal angle between C subspaces: {jnp.rad2deg(angle):.2f} degrees")
    
    # Should be reasonably close (allow some error)
    assert angle < jnp.deg2rad(30), f"C subspace poorly recovered: angle={jnp.rad2deg(angle):.1f}°"
    
    # Check that learned parameters have right properties
    A_final = params_final.dynamics.weights
    spectral_radius = jnp.max(jnp.abs(jnp.linalg.eigvals(A_final)))
    print(f"Final A spectral radius: {spectral_radius:.3f}")
    
    # Should be stable
    assert spectral_radius < 1.0, f"A is unstable: spectral_radius={spectral_radius}"


def test_em_predictive_performance(parameter_recovery_problem):
    """Test that EM improves predictive performance."""
    prob = parameter_recovery_problem
    ctds = prob['ctds']
    params_init = prob['params_init']
    params_true = prob['params_true']
    obs = prob['observations']
    latent_true = prob['latent_true']
    
    batch_obs = jnp.expand_dims(obs, axis=0)
    
    # Compute initial prediction error
    from inference import DynamaxLGSSMBackend
    stats_init, _ = DynamaxLGSSMBackend.e_step(params_init, obs, None)
    pred_init = params_init.emissions.weights @ stats_init.latent_mean.T
    mse_init = jnp.mean((pred_init.T - obs)**2)
    
    # Run EM
    params_final, _ = ctds.fit_em(
        params_init,
        batch_obs,
        num_iters=25,
        verbose=False
    )
    
    # Compute final prediction error
    stats_final, _ = DynamaxLGSSMBackend.e_step(params_final, obs, None)
    pred_final = params_final.emissions.weights @ stats_final.latent_mean.T
    mse_final = jnp.mean((pred_final.T - obs)**2)
    
    print(f"\nInitial MSE: {mse_init:.4f}")
    print(f"Final MSE: {mse_final:.4f}")
    print(f"Improvement: {(mse_init - mse_final) / mse_init * 100:.1f}%")
    
    # Should improve prediction
    assert mse_final < mse_init, "EM did not improve prediction"


def test_em_with_small_T():
    """Test EM stability with small number of timesteps."""
    D = 4
    N = 10
    T = 30  # Small T
    
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
    
    key_init = jax.random.PRNGKey(789)
    params_init = perturb_params(params_true, key_init, scale=0.3)
    
    batch_obs = jnp.expand_dims(obs, axis=0)
    
    # Run EM - should not crash
    params_final, log_probs = ctds.fit_em(
        params_init,
        batch_obs,
        num_iters=10,
        verbose=False
    )
    
    # Check no NaNs
    assert not jnp.any(jnp.isnan(jnp.array(log_probs))), "NaN with small T"
    assert not jnp.any(jnp.isnan(params_final.dynamics.weights)), "A NaN with small T"
    assert not jnp.any(jnp.isnan(params_final.emissions.weights)), "C NaN with small T"


def test_em_with_multiple_sequences():
    """Test EM with batch of multiple sequences."""
    D = 4
    N = 12
    T = 80
    num_sequences = 5
    
    cell_types = jnp.array([0, 1])
    cell_sign = jnp.array([1, -1])
    cell_type_dimensions = jnp.array([2, 2])
    cell_type_mask = jnp.concatenate([
        jnp.zeros(6, dtype=jnp.int32),
        jnp.ones(6, dtype=jnp.int32)
    ])
    
    key = jax.random.PRNGKey(999)
    keys = jax.random.split(key, num_sequences + 1)
    
    # Generate multiple sequences from same model
    params_true, _, _ = generate_synthetic_ssm(
        D, N, T,
        cell_types, cell_sign, cell_type_dimensions, cell_type_mask,
        keys[0]
    )
    
    # Generate observations for each sequence
    observations_list = []
    for i in range(num_sequences):
        _, _, obs = generate_synthetic_ssm(
            D, N, T,
            cell_types, cell_sign, cell_type_dimensions, cell_type_mask,
            keys[i+1],
            Q_scale=0.1,
            R_scale=0.5
        )
        observations_list.append(obs)
    
    batch_obs = jnp.stack(observations_list, axis=0)  # (num_sequences, T, N)
    
    print(f"\nBatch observations shape: {batch_obs.shape}")
    
    ctds = CTDS(
        emission_dim=N,
        cell_types=cell_types,
        cell_sign=cell_sign,
        cell_type_dimensions=cell_type_dimensions,
        cell_type_mask=cell_type_mask,
        state_dim=D
    )
    
    key_init = jax.random.PRNGKey(111)
    params_init = perturb_params(params_true, key_init, scale=0.3)
    
    # Run EM on batch
    params_final, log_probs = ctds.fit_em(
        params_init,
        batch_obs,
        num_iters=15,
        verbose=False
    )
    
    print(f"\nLog-likelihood improvement: {log_probs[-1] - log_probs[0]:.2f}")
    
    # Check monotonicity
    for i in range(1, len(log_probs)):
        assert log_probs[i] >= log_probs[i-1] - 1e-3, \
            f"LL decreased at iteration {i}"
    
    # Check no NaNs
    assert not jnp.any(jnp.isnan(jnp.array(log_probs))), "NaN with multiple sequences"


def test_em_convergence_detection():
    """Test that EM can detect convergence."""
    D = 5
    N = 15
    T = 200
    
    cell_types = jnp.array([0, 1])
    cell_sign = jnp.array([1, -1])
    cell_type_dimensions = jnp.array([3, 2])
    cell_type_mask = jnp.concatenate([
        jnp.zeros(8, dtype=jnp.int32),
        jnp.ones(7, dtype=jnp.int32)
    ])
    
    key = jax.random.PRNGKey(222)
    
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
    
    # Start close to truth for fast convergence
    key_init = jax.random.PRNGKey(333)
    params_init = perturb_params(params_true, key_init, scale=0.1)
    
    batch_obs = jnp.expand_dims(obs, axis=0)
    
    # Run with many iterations but should converge early
    params_final, log_probs = ctds.fit_em(
        params_init,
        batch_obs,
        num_iters=100,
        verbose=False
    )
    
    print(f"\nNumber of EM iterations run: {len(log_probs)}")
    
    # Check that it converged (stopped early)
    assert len(log_probs) < 100, "EM did not detect convergence"
    
    # Check final change is small
    if len(log_probs) > 1:
        final_change = jnp.abs(log_probs[-1] - log_probs[-2]) / jnp.abs(log_probs[-2])
        print(f"Final relative change: {final_change:.2e}")
        assert final_change < 1e-3, "EM did not converge"
