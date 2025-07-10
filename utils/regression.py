import jax
import jax.numpy as jnp
from jaxopt import OSQP

import jax.numpy as jnp
from jax import vmap

def solve_single_constrained_regression(Q, c, mask):
    """
    Solves a single constrained quadratic program using OSQP:
        min_x 0.5 * x^T Q x + c^T x
        s.t. x_i >= 0 if mask[i] is True
             x_i == 0 if mask[i] is False
    """
    lower = jnp.where(mask, 0.0, 0.0)
    upper = jnp.where(mask, jnp.inf, 0.0)
    osqp = OSQP()
    sol = osqp.run(Q, c, jnp.eye(Q.shape[0]), lower, upper)
    return sol.params

##batched and refactored version of ssms fit_constrained_linear_regression NEEDS TO BE TESTED MORE!!
def fit_constrained_linear_regression_batched(
    ExxT: jnp.ndarray,               # (D + 1, D + 1)
    ExyT: jnp.ndarray,               # (D + 1, N)
    list_of_dims: jnp.ndarray,      # (R, C): dims per region/cell type
    region_identity: jnp.ndarray,   # (N,)
    cell_identity: jnp.ndarray,     # (N,)
    fit_intercept: bool = True
):
    """
    Batched constrained regression using OSQP.

    Returns:
        C: (N, D) or (N, D + 1) if fit_intercept
    """
    N = ExyT.shape[1]
    D = ExxT.shape[0] - int(fit_intercept)
    output_dim = D + int(fit_intercept)
    C_out = jnp.zeros((N, output_dim))

    def solve_neuron(i):
        region_id = region_identity[i]
        cell = cell_identity[i]

        dims_this_region = int(jnp.sum(list_of_dims[region_id]))
        dims_prev_regions = int(jnp.sum(list_of_dims[:region_id, :])) if region_id > 0 else 0

        block_start = dims_prev_regions
        block_end = block_start + dims_this_region

        # Extract Q, c
        Q_block = ExxT[block_start:block_end, block_start:block_end]
        c_block = -2 * ExyT[block_start:block_end, i]

        if fit_intercept:
            # Add intercept row and column
            intercept_col = ExxT[block_start:block_end, -1][:, None]
            intercept_row = jnp.hstack((ExxT[-1, block_start:block_end], ExxT[-1, -1]))
            Q_block = jnp.hstack((Q_block, intercept_col))
            Q_block = jnp.vstack((Q_block, intercept_row[None, :]))
            c_block = jnp.concatenate([c_block, jnp.array([ExyT[-1, i]])])
            dim = Q_block.shape[0]
        else:
            dim = Q_block.shape[0]

        # Build constraint mask
        if cell == 0:
            # Unknown cell type → all entries allowed
            mask = jnp.ones(dim, dtype=bool)
        else:
            block_idx = cell - 1
            block_dim = list_of_dims[region_id, block_idx]
            start = sum(list_of_dims[region_id, :block_idx])
            end = start + block_dim
            mask = jnp.zeros(dim, dtype=bool)
            mask = mask.at[start:end].set(True)
            if fit_intercept:
                mask = mask.at[-1].set(True)

        return solve_single_constrained_regression(2 * Q_block, c_block, mask)

    C_out = vmap(solve_neuron)(jnp.arange(N))  # (N, D + 1) or (N, D)

    if fit_intercept:
        return C_out[:, :-1], C_out[:, -1]
    else:
        return C_out, None



def _compute_statistics_for_m_step(self, ys, inputs, continuous_expectations, discrete_expectations):
    """
    JAX-compatible: compute all statistics needed for M step when a closed-form update is feasible.
    Returns:
        ExxT: (K, D+1, D+1)
        ExyT: (K, D+1, N)
        EyyT: (K, N, N)
        weight_sum: scalar
    """
    K = self.K
    N = self.N
    D = self.D

    # Initialize accumulators
    EyyT = jnp.zeros((K, N, N))
    ExxT = jnp.zeros((K, D+1, D+1))
    ExyT = jnp.zeros((K, D+1, N))
    weight_sum = 0.0

    for y, input, (_, Ex, smoothed_sigmas, _), (Ez, _, _) in zip(ys, inputs, continuous_expectations, discrete_expectations):
        y = jnp.asarray(y)
        Ex = jnp.asarray(Ex)
        smoothed_sigmas = jnp.asarray(smoothed_sigmas)
        Ez = jnp.asarray(Ez)
        for k in range(K):
            w = Ez[:, k]  # (T,)
            # EyyT
            EyyT = EyyT.at[k].add(jnp.einsum('t,ti,tj->ij', w, y, y))
            # ExxT
            mumuT = jnp.einsum('ti,tj->tij', Ex, Ex) + smoothed_sigmas  # (T, D, D)
            ExxT = ExxT.at[k, :D, :D].add(jnp.einsum('t, tij->ij', w, mumuT))
            ExxT = ExxT.at[k, -1, :D].add(jnp.einsum('t, ti->i', w, Ex))
            ExxT = ExxT.at[k, :D, -1].add(jnp.einsum('t, ti->i', w, Ex))
            ExxT = ExxT.at[k, -1, -1].add(jnp.sum(w))
            # ExyT
            ExyT = ExyT.at[k, :D, :].add(jnp.einsum('t,ti,tj->ij', w, Ex, y))
            ExyT = ExyT.at[k, -1, :].add(jnp.einsum('t,ti->i', w, y))
            weight_sum += jnp.sum(w)
    return ExxT, ExyT, EyyT, weight_sum