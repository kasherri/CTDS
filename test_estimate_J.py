
import jax
import pytest
import jax.numpy as jnp
import numpy as np
from functools import partial
from models.components.intialize import estimate_J
#python -m pytest test_estimate_J.py

# Seed for reproducibility
key = jax.random.PRNGKey(0)

def generate_dale_J(N, frac_excit=0.7, key=None):
    """
    Generate a ground truth J matrix with Dale constraints.
    Excitatory neurons: non-negative outgoing weights.
    Inhibitory neurons: non-positive outgoing weights.
    """
    key_EI, key_vals = jax.random.split(key)
    mask = jax.random.bernoulli(key_EI, p=frac_excit, shape=(N,))  # True = excitatory
    J = jnp.zeros((N, N))

    def generate_col(j, J):
        is_excit = mask[j]
        key_j = jax.random.fold_in(key_vals, j)
        vals = jax.random.normal(key_j, (N,))
        # Apply Dale constraint
        col = jnp.where(is_excit, jnp.abs(vals), -jnp.abs(vals)) * 0.2  # small weights
        return J.at[:, j].set(col)

    J = jax.lax.fori_loop(0, N, generate_col, J)
    return J, mask

def simulate_activity(J_true, T, noise_scale=0.05, key=None):
    """
    Simulate activity Y with AR(1) dynamics: Y_t = J Y_{t-1} + ε_t
    """
    N = J_true.shape[0]
    Y = jnp.zeros((N, T))
    key_noise = key

    def step(t, Y):
        key_t = jax.random.fold_in(key_noise, t)
        eps = noise_scale * jax.random.normal(key_t, (N,))
        y_next = J_true @ Y[:, t - 1] + eps
        return Y.at[:, t].set(y_next)

    Y = Y.at[:, 0].set(jax.random.normal(key_noise, (N,)))  # Init
    Y = jax.lax.fori_loop(1, T, step, Y)
    return Y

# ------------------------------
# 1. Generate true J and mask
N, T = 5, 5
J_true, mask = generate_dale_J(N=N, key=key)

# 2. Simulate dynamics
Y = simulate_activity(J_true, T=T, key=key)
print("Y",Y.shape)
# 3. Estimate J
J_est = estimate_J(Y, mask)
print("True J:\n", J_true)
print("Estimated J:\n", J_est)

# 4. Evaluate accuracy
def evaluate(J_true, J_est, mask):
    error = jnp.linalg.norm(J_true - J_est) / jnp.linalg.norm(J_true)
    print(f"Relative Frobenius Error: {error:.3f}")
    # Sign checks
    excit_error = jnp.sum((J_est[:, mask] < -1e-5))  # should be ≥ 0
    inhib_error = jnp.sum((J_est[:, ~mask] > 1e-5))  # should be ≤ 0
    print(f"Violations: Excitatory: {excit_error}, Inhibitory: {inhib_error}")

evaluate(J_true, J_est, mask)
