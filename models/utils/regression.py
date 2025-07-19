import jax
import jax.numpy as jnp
from jaxopt import BoxOSQP
from jax import vmap


_boxosqp = BoxOSQP() 

#might change args to matvec functions
@jax.jit
def solve_dale_QP(Q, c, mask):
    """
    Solve a sign-constrained quadratic program with cell-type-specific constraints.

    Problem form:
        minimize    (1/2) * xᵀ Q x + cᵀ x
        subject to:
            x[i] >= 0      if mask[i] is True  (excitatory neuron)
            x[i] == 0      if mask[i] is False (inhibitory neuron)

    This is a box-constrained QP where:
        - Excitatory (E) neurons can have positive weights
        - Inhibitory (I) neurons are clamped to zero

    Args:
        Q (jnp.ndarray): Positive semi-definite matrix of shape (D, D).
        c (jnp.ndarray): Linear coefficient vector of shape (D,).
        mask (jnp.ndarray of bools): Shape (D,). 
            `True` → E-cell (allow ≥0),
            `False` → I-cell (force to 0).

    Returns:
        jnp.ndarray: Solution vector x ∈ ℝ^D satisfying constraints.
    """
    D = Q.shape[0]
    A= jnp.eye(D)
   
    #if mask is true cell is E cell
    #else cell is I cell
    #.where returns 0.0 for I cells and inf for E cells
    lower = jnp.where(mask, 0.0, -jnp.inf) #E cell lower bound is 0.0, I cell lower bound is -inf
    upper = jnp.where(mask, jnp.inf, 0.0) # I cell upper bound is 0.0, E cell upper bound is inf
    
    
    # Run the OSQP solver
    sol = _boxosqp.run( params_obj=(Q, c), params_eq=A,  params_ineq=(lower, upper))
    
    return sol.params.primal[0]  # Return the optimal parameters (solution vector)








"""
Example usage:
if __name__ == "__main__":
    # Simulate data
    key = jax.random.PRNGKey(0)
    T, M, N = 100, 10, 5
    X = jax.random.normal(key, (T, M))
    true_W = jax.random.uniform(key, (M, N))
    Y = X @ true_W + 0.1 * jax.random.normal(key, (T, N))
    
    # Estimate W via NNLS-QP
    W_est = nnls_qp(X, Y, maxiter=300, tol=1e-6)
    print("Estimated W (non-negative):\n", W_est)
"""
