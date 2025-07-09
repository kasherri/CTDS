
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from models.ctds_model import CTDSModel


def simulate_ctds_data(
    model: CTDSModel,
    T: int,
    num_trials: int,
    seed: int = 0
) -> Tuple[List[jnp.ndarray], List[jnp.ndarray]]:
    """
    simulate multiple trials of data from the CTDS model.

    Args:
        model: Instantiated CTDSModel object
        T: Number of time steps
        num_trials: Number of independent trials
        seed: PRNG seed

    Returns:
        states_list: List of length `num_trials`, each (T, D) latent state trajectory
        data_list: List of length `num_trials`, each (T, N) observed neural activity
    """
    key = jax.random.PRNGKey(seed)
    states_list, data_list = [], []

    for i in range(num_trials):
        key, subkey = jax.random.split(key)
        states, observations = model.sample(T, prefix=None, rng=subkey)
        states_list.append(states)
        data_list.append(observations)

    return states_list, data_list



