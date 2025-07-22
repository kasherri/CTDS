import jax
import jax.numpy as jnp
#from jaxopt import BoxOSQP
from typing import Optional
from jaxopt import BoxCDQP
import jax.random as jr


_boxCDQP = BoxCDQP(tol=1e-6) 

#might change args to matvec functions
#need to change so it constrains diagonal to 0
@jax.jit
def solve_dale_QP(Q, c, mask):
    """
    Solve a sign-constrained quadratic program with cell-type-specific constraints.

    Problem form:
        minimize    (1/2) * xᵀ Q x + cᵀ x
        subject to:
            x[i] >= 0      if mask[i] is True  (excitatory neuron)
            x[i] <= 0      if mask[i] is False (inhibitory neuron)

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
  
   
    #if mask is true cell is E cell
    #else cell is I cell
    #.where returns 0.0 for I cells and inf for E cells
    lower = jnp.where(mask, 1e-3, -jnp.inf) #E cell lower bound is 0.0, I cell lower bound is -inf
    upper = jnp.where(mask, jnp.inf, -1e-3) # I cell upper bound is 0.0, E cell upper bound is inf

    init_x = jax.random.normal(jax.random.PRNGKey(42), (D,))
    
    # Run the OSQP solver
    sol = _boxCDQP.run( init_x, params_obj=(Q, c),   params_ineq=(lower, upper))
  
    #return sol.params.primal[0]  # Return the optimal parameters (solution vector)
    return sol.params



#initialize factors for NMF
@jax.jit
def init_factors(key, shape_u, shape_v, method='uniform', J_plus=None):
    """
    Initialize non-negative matrix factors U and V.

    Args:
        key: jax.random.PRNGKey
        shape_u: tuple, shape of U (e.g., (N_E, K1))
        shape_v: tuple, shape of V (e.g., (N, K1))
        method: str, one of {'uniform', 'exp', 'nndsvd'}
        J_plus: jnp.ndarray, required for 'nndsvd'; shape should match (shape_u[0], shape_v[0])

    Returns:
        U (jnp.ndarray), V (jnp.ndarray): initialized factors
    """
    if method == 'uniform':
        k1, k2 = jr.split(key)
        U = jr.uniform(k1, shape_u)
        V = jr.uniform(k2, shape_v)

    elif method == 'exp':
        k1, k2 = jr.split(key)
        U = jr.exponential(k1, shape_u)
        V = jr.exponential(k2, shape_v)

    elif method == 'nndsvd':
        assert J_plus is not None, "J_plus must be provided for nndsvd initialization"
        # shape_u = (M, K), shape_v = (N, K) implies J_plus ∈ ℝ^{M × N}
        U_full, S, VT_full = jnp.linalg.svd(J_plus, full_matrices=False)
        K = shape_u[1]

        U_trunc = U_full[:, :K]
        S_trunc = S[:K]
        V_trunc = VT_full[:K, :].T

        U = jnp.maximum(U_trunc, 0.0) * jnp.sqrt(S_trunc)
        V = jnp.maximum(V_trunc, 0.0) * jnp.sqrt(S_trunc)

    else:
        raise ValueError(f"Unknown init method: {method}")

    # Normalize (helps stability)
    U = U / (jnp.linalg.norm(U, axis=1, keepdims=True) + 1e-8)
    V = V / (jnp.linalg.norm(V, axis=0, keepdims=True) + 1e-8)

    return U, V


"""
# Example usage:
# Random init
U, V = init_factors(jr.PRNGKey(0), (N_E, K1), (N, K1), method='uniform')

# NNDSVD init
J_E_plus = jnp.abs(J[:N_E, :])
U, V = init_factors(None, (N_E, K1), (N, K1), method='nndsvd', J_plus=J_E_plus)

"""

# Step 1: Estimate J from neural activity Y
#   J is the matrix of weights predicting each neuron from all others.
#   J is constrained to have non-negative weights for excitatory neurons and non-positive for inhibitory neurons.
#   This is done via a constrained quadratic program (QP) for each column of J
#   using the BoxCDQP solver from JAXOPT.
@jax.jit
def estimate_J( Y: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Estimate the Dale matrix J from neural activity Y.
    Args:
        Y: (N, T) array of neural activity, where N is number of neurons and T is timepoints.
        mask: (N,) boolean array indicating excitatory (True) vs inhibitory (False) neurons.
    Returns:
        J: (N, N) estimated Dale matrix, where J[i, j] is the weight from neuron j to neuron i.
    """

    N, T = Y.shape
    J = jnp.zeros((N, N))
   
    Y_past = Y[:, :-1]           # ( N, T-1)  # Past observations
    Y_future = Y[:, 1:]    # (N, T-1)  # Future observations

    # X= Y_pastᵀ∈ ℝ^{(T-1) × N}
    X = Y_past.T  # (T-1, N)

    epsilon = 1e-3  # Small regularization constant (can tune this)
   #Q = 2 Xᵀ X∈ ℝ^{N × N}, positive semidefinite
    Q = 2.0 * (X.T @ X) + epsilon * jnp.eye(N)

    C = -2.0 * (X.T @ Y_future.T)            # shape (N, N) → column c_j is C[:, j]
    masks = jnp.tile(mask[None, :], (N, 1)) 

    vmap_solver = jax.vmap(solve_dale_QP, in_axes=(None, 1, 0))
    J=vmap_solver(Q, C, masks)  # shape (N, N)
    return J


#TO DO: Multiregion estimate_J


"""

# Dummy data: 3 neurons, 5 timepoints
Y = jnp.array([
    [1., 2., 3., 4., 5.],    # neuron 0
    [0., 1., 0., 1., 0.],    # neuron 1
    [5., 4., 3., 2., 1.],    # neuron 2
])


# Neuron types: [E, I, E]
mask = jnp.array([True, False, False])  # True = excitatory, False = inhibitory


# Run the estimator
#J_est = estimate_J(Y, mask)

#print("Estimated J:\n", J_est)
"""





# Step 2: J⁺ = |J|

def compute_J_plus(J: jnp.ndarray) -> jnp.ndarray:
    """Element‐wise absolute value of J."""
    return jnp.abs(J)


# Step 3: multiplicative‐update NMF
def nmf(
    key: jax.random.PRNGKey,
    M: jnp.ndarray,
    rank: int,
    num_iters: int = 100,
    eps: float = 1e-8
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Non-negative matrix factorization via multiplicative updates.
    Solves M ≈ U @ V^T with U >= 0, V >= 0.
    Args:
      key: PRNGKey for initialization.
      M: matrix to factor, shape (m, n).
      rank: number of components.
      num_iters: number of update iterations.
      eps: small constant to avoid division by zero.
    Returns:
      U: (m, rank), V: (n, rank)
    """
    m, n = M.shape
    k1, k2 = jax.random.split(key)
    # initialize U, V > 0
    U = jax.random.uniform(k1, (m, rank), minval=0.1, maxval=1.0)
    V = jax.random.uniform(k2, (n, rank), minval=0.1, maxval=1.0)

    def body(_, UV):
        """
        
        """
        U, V = UV
        # update U
        numerator_U = M @ V                     # (m, rank)
        denominator_U = (U @ (V.T @ V)) + eps   # (m, rank)
        U = U * (numerator_U / denominator_U)
        # update V
        numerator_V = M.T @ U                   # (n, rank)
        denominator_V = (V @ (U.T @ U)) + eps   # (n, rank)
        V = V * (numerator_V / denominator_V)
        return (U, V)

    U_final, V_final = jax.lax.fori_loop(0, num_iters, body, (U, V))
    return U_final, V_final


# Step 4–5: block-wise NMF and build U, V_dale
#   U is block-diagonal, V_dale = [V1, -V2] where V1, V2 are NMF factors for excitatory/inhibitory blocks.
#   U is used to initialize the latent dynamics matrix A₀.
#   V_dale is used to apply Dale's principle to the latent dynamics.
@jax.jit
def blockwise_nmf(
    key: jax.random.PRNGKey,
    J_plus: jnp.ndarray,
    NE: int,
    NI: int,
    DE: int,
    DI: int,
    num_iters: int = 100,
    eps: float = 1e-8
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Splits J⁺ into excitatory/inhibitory blocks, runs NMF, and constructs:
      U     shape (N, DE+DI)  (block-diagonal)
      V_dale shape (N, DE+DI)  ([ V1, -V2 ])
    """
    N = NE + NI
    # Split keys
    k1, k2 = jax.random.split(key)

    # Block-wise NMF
    U1, V1 = nmf(k1, J_plus[:NE, :], DE, num_iters, eps)   # U1: NE×DE, V1: N×DE
    U2, V2 = nmf(k2, J_plus[NE:, :], DI, num_iters, eps)   # U2: NI×DI, V2: N×DI

    # Build U = block_diag(U1, U2)  → N×(DE+DI)
    top    = jnp.concatenate([U1, jnp.zeros((NE, DI))], axis=1)
    bottom = jnp.concatenate([jnp.zeros((NI, DE)), U2], axis=1)
    U = jnp.concatenate([top, bottom], axis=0)             # (N, DE+DI)

    # Build V_dale = [ V1, -V2 ]  → N×(DE+DI)
    V_dale = jnp.concatenate([V1, -V2], axis=1)

    return U, V_dale


# Step 6: initial A₀ = V_daleᵀ @ U
#C=U
@jax.jit
def build_A(U: jnp.ndarray, V_dale: jnp.ndarray) -> jnp.ndarray:
    """
    Compute initial latent dynamics matrix:
      A = V_dale^T @ U
    """
    return V_dale.T @ U

