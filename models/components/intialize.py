import jax
import jax.numpy as jnp
from jaxopt import BoxOSQP


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





# Step 1: Estimate J from neural activity Y
#   J is the matrix of weights predicting each neuron from all others.
#   J is constrained to have non-negative weights for excitatory neurons and non-positive for inhibitory neurons.
#   This is done via a constrained quadratic program (QP) for each column of J
#   using the BoxCDQP solver from JAXOPT.
@jax.jit
def estimate_J( Y: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """
    Estimate Dale-constrained J from neural activity Y.
    
    Args:
      Y: array of shape (N, T), each row is one timepoint across N neurons.
      excitatory_mask: bool array of shape (N,), True if neuron j is excitatory.
      maxiter: max CD iterations in BoxCDQP.
      tol: convergence tolerance for BOXCDQP    .
    
    Returns:
      J: array of shape (N, N), column j = weights predicting neuron j.
    """


    N, T = Y.shape
    J = jnp.zeros((N, N))
   
    Y_past = Y[:, :-1]           # ( N, T-1)  # Past observations
    Y_future = Y[:, 1:]    # (N, T-1)  # Future observations

    # X= Y_pastᵀ∈ ℝ^{(T-1) × N}
    X = Y_past.T  # (T-1, N)

    """
      For each column j ∈ {1, ..., N}:
          Let y_future_j = Y[j, :]ᵀ      ∈ ℝ^{T-1}
              X         = Y_pastᵀ        ∈ ℝ^{(T-1) × N}  
    """
    Q = 2.0 * jnp.dot(X.T, X) #Q = 2 Xᵀ X∈ ℝ^{N × N}, positive semidefinite
    #implent with fori
    def body_fn(j, J):
       y_future_j = Y_future[j, :].T  # (T-1,)
       c = -2.0 * jnp.dot(X.T, y_future_j)  #c = -2 Xᵀ y_future_j ∈ ℝ^N
       w_mask = jnp.full((N,), mask[j])
# (N,) if mask[j] is True, then w_mask is filled with N True, else filled with N False
       w_j= solve_dale_QP(Q, c, w_mask)  # Solve the constrained QP for column j
       # Update J with the solution for column j
       #assert w_j.shape == (N,), f"w_j.shape = {w_j.shape}, expected {(N,)}"

       return J.at[:, j].set(w_j)  # set column j

    J_init = jnp.zeros((N, N))
    J = jax.lax.fori_loop(0, N, body_fn, J_init)
    return J

"""
# Dummy data: 3 neurons, 5 timepoints
Y = jnp.array([
    [1., 2., 3., 4., 5.],    # neuron 0
    [0., 1., 0., 1., 0.],    # neuron 1
    [5., 4., 3., 2., 1.],    # neuron 2
])

# Neuron types: [E, I, E]
mask = jnp.array([True, False, True])  # True = excitatory, False = inhibitory

# Run the estimator
J_est = estimate_J(Y, mask)

print("Estimated J:\n", J_est)
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

