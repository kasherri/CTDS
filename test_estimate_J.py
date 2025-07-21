
import jax
import pytest
import cvxpy as cp
from jax.numpy.linalg import eigvals
import matplotlib.pyplot as plt
from tqdm import tqdm
import jax.numpy as jnp
import numpy as np
from functools import partial
from models.components.intialize import estimate_J, solve_dale_QP
#python -m pytest test_estimate_J.py
import mosek


# Seed for reproducibility
key = jax.random.PRNGKey(0)

#unconstrained J matrix generation to test if solve_dale_qp was the problem
def generate_unconstrained_J(N, key, weight_scale=1.0, max_spectral_radius=0.95):
    """
    Generate an unconstrained J matrix spectral radius so values dont explode
    """
    J = weight_scale * jax.random.normal(key, shape=(N, N))
    eigs = jnp.abs(eigvals(J))
    spectral_radius = jnp.max(eigs)
    J_scaled = J * (max_spectral_radius / spectral_radius)
    return J_scaled


def estimate_unconstrained_J(Y: jnp.ndarray) -> jnp.ndarray:
    """
    uncontrained J estimation via unconstrained least squares
    """
    Y_past = Y[:, :-1]   # Shape (N, T-1)
    Y_future = Y[:, 1:]  # Shape (N, T-1)

    #  least squares: J @ Y_past ≈ Y_future
    # equiv to J = Y_future @ Y_past.T @ inv(Y_past @ Y_past.T)
    Xt = Y_past
    Yt = Y_future

    XtXt_inv = jnp.linalg.inv(Xt @ Xt.T + 1e-6 * jnp.eye(Xt.shape[0]))  # regularize
    J = Yt @ Xt.T @ XtXt_inv

    return J

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
N, T = 10, 1000
J_true, mask = generate_dale_J(N=N, key=key)


# 2. Simulate dynamics
Y = simulate_activity(J_true, T=T, key=key)
# 3. Estimate J
J_est = estimate_J(Y, mask)
#print("True J:\n", J_true.shape)
#print("Estimated J:\n", J_est)

# 4. Evaluate accuracy
def evaluate(J_true, J_est, mask):
    error = jnp.linalg.norm(J_true - J_est) / jnp.linalg.norm(J_true)
    print(f"Relative Frobenius Error: {error:.3f}")
    # Sign checks
    excit_error = jnp.sum((J_est[:, mask] < -1e-5))  # should be ≥ 0
    inhib_error = jnp.sum((J_est[:, ~mask] > 1e-5))  # should be ≤ 0
    print(f"Violations: Excitatory: {excit_error}, Inhibitory: {inhib_error}")

evaluate(J_true, J_est, mask)
J_unconstrained = generate_unconstrained_J(N, key)
Y_unconstrained = simulate_activity(J_unconstrained, T=T, key=key)
J_est_unconstrained = estimate_unconstrained_J(Y_unconstrained)

error= jnp.linalg.norm(J_unconstrained - J_est_unconstrained) / jnp.linalg.norm(J_unconstrained)
print(f"Relative Frobenius Error (unconstrained): {error}")



#comparing with Aditis implementation
def learn_J_from_data(datas, N_e=0, signs=None):
    """use linear regression to learn an estimate of the connectivity matrix J"""

    y = np.concatenate(datas, axis = 0)
    y_data = y[:-1,:]
    y_next_data = y[1:,:]

    # J should obey Dale's law so solve for J using constrained linear regression
    if y_data.shape[1] < 100:
        # learn this matrix in a go when # of neurons <=100
        J = cp.Variable((y_data.shape[1],y_data.shape[1]))
        # N_e columns of J should be positive and the rest should be negative
        if signs is None:
            constraints = [J[:,0:N_e] >= 0, J[:,N_e:] <= 0]
        else:
            e_cells = np.where(signs==1)[0]
            i_cells = np.where(signs==2)[0]
            constraints = [J[:,e_cells] >= 0, J[:,i_cells] <= 0]
        # define the objective``
        objective = cp.Minimize(cp.norm(y_next_data.T - J@y_data.T, 'fro'))
        # define the problem
        problem = cp.Problem(objective, constraints)
        # solve the problem
        problem.solve(cp.MOSEK, verbose = False)
        return J.value
    else:
        # learn every row of J separately
        J = np.empty((y_data.shape[1], y_data.shape[1]))
        for i in tqdm(range(y_data.shape[1])):
            J_i = cp.Variable((y_data.shape[1]))
            # N_e columns of J should be positive and the rest should be negative
            if signs is None:
                constraints = [J_i[0:N_e] >= 0, J_i[N_e:] <= 0]
            else:
                e_cells = np.where(signs==1)[0]
                i_cells = np.where(signs==2)[0]
                constraints = [J_i[e_cells] >= 0, J_i[i_cells] <= 0]
            # define the objective
            objective = cp.Minimize(cp.norm(y_next_data.T[i,:] - J_i@y_data.T, 'fro'))
            # define the problem
            problem = cp.Problem(objective, constraints)
            # solve the problem
            problem.solve(cp.MOSEK, verbose = False)
            # put this solution in J
            J[i,:] = J_i.value
        return J
    


#convery Y to numpy array
signs=jnp.where(mask, 1, 2)  # 1 for excitatory, 2 for inhibitory
Y_np = Y.astype(np.float64)
J_est_aditi = learn_J_from_data([Y_np.T], N_e=2, signs=signs.astype(np.int32))
#print("Estimated J (Aditi's method):\n", J_est_aditi.shape)
#convert to jax array
J_est_aditi = jnp.array(J_est_aditi)
evaluate(J_true, J_est_aditi, mask)





# Plot difference matrices
fig, axs = plt.subplots(1, 3, figsize=(12, 5))
im0 = axs[0].imshow(J_true - J_est, cmap="bwr")
axs[0].set_title("J_true - J_est (JAX Constrained)")
plt.colorbar(im0, ax=axs[0])



im1 = axs[1].imshow(J_unconstrained - J_est_unconstrained, cmap="bwr")
axs[1].set_title("J_unconstrained - J_est_unconstrained")
plt.colorbar(im1, ax=axs[1])

im2 = axs[2].imshow(J_true - J_est_aditi, cmap="bwr")
axs[2].set_title("J_true - J_est_aditi (CVPXY Method)")
plt.colorbar(im2, ax=axs[2])

plt.tight_layout()
plt.show()
