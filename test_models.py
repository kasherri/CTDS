import jax
import pytest
from params import ParamsCTDS, SufficientStats
from models import CTDS

# test_models.py

import jax.numpy as jnp


@pytest.fixture
def dummy_data():
    N, T, D_E, D_I = 5, 10, 2, 2
    D = D_E + D_I
    key = jax.random.PRNGKey(0)
    observations = jax.random.normal(key, (N, T))
    mask = jnp.array([True, True, False, False, True])
    return observations, mask, D_E, D_I, D, N, T

@pytest.fixture
def dummy_stats(dummy_data):
    _, _, _, _, D, _, T = dummy_data
    latent_mean = jax.random.normal(jax.random.PRNGKey(1), (T, D))
    latent_second_moment = jax.random.normal(jax.random.PRNGKey(2), (T, D, D))
    cross_time_moment = jax.random.normal(jax.random.PRNGKey(3), (T-1, D, D))
    return SufficientStats(
        latent_mean=latent_mean,
        latent_second_moment=latent_second_moment,
        cross_time_moment=cross_time_moment,
        loglik=0.1,
        T=T
    )

def test_initialize_params_shapes_and_types(dummy_data):
    observations, mask, D_E, D_I, D, N, T = dummy_data
    model = CTDS(observations, mask, mask, D)
    params = model.initialize_params(observations, mask, D_E=D_E, D_I=D_I)
    assert isinstance(params, ParamsCTDS)
    assert params.initial.mean.shape == (D,)
    assert params.initial.cov.shape == (D, D)
    assert params.dynamics.A.shape == (D, D)
    assert params.dynamics.Q.shape == (D, D)
    assert params.emissions.C.shape == (N, D)
    assert params.emissions.R.shape == (N, N)

def test_m_step_shapes_and_types(dummy_data, dummy_stats):
    observations, mask, D_E, D_I, D, N, T = dummy_data
    model = CTDS(observations, mask, mask, D)
    params = model.initialize_params(observations, mask, D_E=D_E, D_I=D_I)
    # Add cell_types_mask to params for m_step
    params = ParamsCTDS(
        initial=params.initial,
        dynamics=params.dynamics,
        emissions=params.emissions,
        cell_types_mask=mask
    )
    new_params = model.m_step(params, dummy_stats)
    assert isinstance(new_params, ParamsCTDS)
    assert new_params.initial.mean.shape == (D,)
    assert new_params.initial.cov.shape == (D, D)
    assert new_params.dynamics.A.shape == (D, D)
    assert new_params.dynamics.Q.shape == (D, D)
    assert new_params.emissions.C.shape == (N, D)
    assert new_params.emissions.R.shape == (N, N)


def test_m_step_updates_dynamics_and_emissions(dummy_data, dummy_stats):
    """
    Test that m_step updates the dynamics and emissions parameters and that their values change.
    """
    observations, mask, D_E, D_I, D, N, T = dummy_data
    model = CTDS(observations, mask, mask, D)
    params = model.initialize_params(observations, mask, D_E=D_E, D_I=D_I)
    params = ParamsCTDS(
        initial=params.initial,
        dynamics=params.dynamics,
        emissions=params.emissions,
        cell_types_mask=mask
    )
    new_params = model.m_step(params, dummy_stats)
    # Check that at least one value in A, Q, C, R has changed
    assert not jnp.allclose(params.dynamics.A, new_params.dynamics.A)
    assert not jnp.allclose(params.dynamics.Q, new_params.dynamics.Q)
    assert not jnp.allclose(params.emissions.C, new_params.emissions.C)
    assert not jnp.allclose(params.emissions.R, new_params.emissions.R)

def test_m_step_preserves_shapes(dummy_data, dummy_stats):
    """
    Test that m_step output shapes match input parameter shapes.
    """
    observations, mask, D_E, D_I, D, N, T = dummy_data
    model = CTDS(observations, mask, mask, D)
    params = model.initialize_params(observations, mask, D_E=D_E, D_I=D_I)
    params = ParamsCTDS(
        initial=params.initial,
        dynamics=params.dynamics,
        emissions=params.emissions,
        cell_types_mask=mask
    )
    new_params = model.m_step(params, dummy_stats)
    assert new_params.dynamics.A.shape == (D, D)
    assert new_params.dynamics.Q.shape == (D, D)
    assert new_params.emissions.C.shape == (N, D)
    assert new_params.emissions.R.shape == (N, N)
    assert new_params.initial.mean.shape == (D,)
    assert new_params.initial.cov.shape == (D, D)

def test_m_step_nonnegativity_of_C(dummy_data, dummy_stats):
    """
    Test that the emission matrix C returned by m_step is non-negative (NNLS constraint).
    """
    observations, mask, D_E, D_I, D, N, T = dummy_data
    model = CTDS(observations, mask, mask, D)
    params = model.initialize_params(observations, mask, D_E=D_E, D_I=D_I)
    params = ParamsCTDS(
        initial=params.initial,
        dynamics=params.dynamics,
        emissions=params.emissions,
        cell_types_mask=mask
    )
    new_params = model.m_step(params, dummy_stats)
    assert jnp.all(new_params.emissions.C >= 0)

def test_m_step_returns_new_object(dummy_data, dummy_stats):
    """
    Test that m_step returns a new ParamsCTDS object, not the same as input.
    """
    observations, mask, D_E, D_I, D, N, T = dummy_data
    model = CTDS(observations, mask, mask, D)
    params = model.initialize_params(observations, mask, D_E=D_E, D_I=D_I)
    params = ParamsCTDS(
        initial=params.initial,
        dynamics=params.dynamics,
        emissions=params.emissions,
        cell_types_mask=mask
    )
    new_params = model.m_step(params, dummy_stats)
    assert new_params is not params
    assert isinstance(new_params, ParamsCTDS)


@pytest.fixture
def large_dummy_data():
    N, T, D_E, D_I = 20, 50, 5, 5  # Larger N and T
    D = D_E + D_I
    key = jax.random.PRNGKey(42)
    observations = jax.random.normal(key, (N, T))
    mask = jnp.array([True] * (N // 2) + [False] * (N - N // 2))
    return observations, mask, D_E, D_I, D, N, T

@pytest.fixture
def large_dummy_stats(large_dummy_data):
    _, _, _, _, D, _, T = large_dummy_data
    latent_mean = jax.random.normal(jax.random.PRNGKey(101), (T, D))
    latent_second_moment = jax.random.normal(jax.random.PRNGKey(102), (T, D, D))
    cross_time_moment = jax.random.normal(jax.random.PRNGKey(103), (T-1, D, D))
    return SufficientStats(
        latent_mean=latent_mean,
        latent_second_moment=latent_second_moment,
        cross_time_moment=cross_time_moment,
        loglik=0.1,
        T=T
    )

def test_m_step_large_inputs(large_dummy_data, large_dummy_stats):
    observations, mask, D_E, D_I, D, N, T = large_dummy_data
    model = CTDS(observations, mask, mask, D)
    params = model.initialize_params(observations, mask, D_E=D_E, D_I=D_I)
    params = ParamsCTDS(
        initial=params.initial,
        dynamics=params.dynamics,
        emissions=params.emissions,
        cell_types_mask=mask
    )
    new_params = model.m_step(params, large_dummy_stats)
    assert isinstance(new_params, ParamsCTDS)
    assert new_params.initial.mean.shape == (D,)
    assert new_params.initial.cov.shape == (D, D)
    assert new_params.dynamics.A.shape == (D, D)
    assert new_params.dynamics.Q.shape == (D, D)
    assert new_params.emissions.C.shape[1] == D or new_params.emissions.C.shape[0] == D  # Accept either (N, D) or (D, N)
    assert new_params.emissions.R.shape
