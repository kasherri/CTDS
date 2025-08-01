import jax
import jax.numpy as jnp
#from jaxopt import BoxOSQP
from typing import Optional
from jaxopt import BoxCDQP
import jax.random as jr
from dynamax.linear_gaussian_ssm import PosteriorGSSMSmoothed
from params import SufficientStats


_boxCDQP = BoxCDQP(tol=1e-7, maxiter=10000, verbose=False) 

#might change args to matvec functions
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

    init_x = jax.random.normal(jax.random.PRNGKey(42), (D,)) #TODO: find better way of intializing
    
    # Run the OSQP solver
    sol = _boxCDQP.run( init_x, params_obj=(Q, c),   params_ineq=(lower, upper))
  
    #return sol.params.primal[0]  # Return the optimal parameters (solution vector)
    return sol.params


@jax.jit
def solve_constrained_QP(Q, c, mask, isExcitatory, key=jax.random.PRNGKey(0)):
    def true_fn(args):
        """
        all entries are nonnegative except diagonals. True=Non-neggative, False=unconstrained diagonal
        """
        Q, c, mask, key = args
        D = Q.shape[0]
        lower = jnp.where(mask, 0, -jnp.inf)
        upper = jnp.where(mask, jnp.inf, jnp.inf)
        init_x = jax.random.normal(key, (D,))
        sol = _boxCDQP.run(init_x, params_obj=(Q, c), params_ineq=(lower, upper))
        return sol.params

    def false_fn(args):
        """
        all entries are nonpositive except diagonals. True=Non-positive, False=unconstrained diagonal
        """
        Q, c, mask, key = args
        D = Q.shape[0]
        lower = jnp.where(mask, -jnp.inf, -jnp.inf)
        upper = jnp.where(mask, 0, jnp.inf)
        init_x = jax.random.normal(key, (D,))
        sol = _boxCDQP.run(init_x, params_obj=(Q, c), params_ineq=(lower, upper))
        return sol.params

    return jax.lax.cond(isExcitatory, 
                    (Q, c, mask, key), true_fn, 
                    (Q, c, mask, key), false_fn)



@jax.jit
def NNLS(Q, c):
    lower = jnp.zeros(c.shape[0])  # Non-negative constraint
    upper = jnp.inf * jnp.ones(c.shape[0])  # No upper bound
    init_x = jax.random.normal(jax.random.PRNGKey(42), (c.shape[0],))
    sol = _boxCDQP.run(init_x, params_obj=(Q, c), params_ineq=(lower, upper))
    return sol.params


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

    C = -2.0 * (X.T @ Y_future.T)            # shape (N, N) → column c_j is C[:, j] with shape (N,)
    masks = jnp.tile(mask[None, :], (N, 1))  #shape (N, N), tile mask for each neuron 

    vmap_solver = jax.vmap(solve_dale_QP, in_axes=(None, 1, 0)) # vmap over each column of C and each row of Q
    J=vmap_solver(Q, C, masks)  # shape (N, N)
    return J


#TO DO: Multiregion estimate_J


def blockwise_nmf(J,mask, D_E,D_I):
    """
    TODO: docustring
    """
    J_plus = jnp.abs(J)  #J⁺ = |J|,  where J⁺_{ij} = |J_{ij}|

    # Split J⁺ into excitatory and inhibitory blocks
    idx_E = jnp.nonzero(jnp.where(mask, 1, 0))[0]
    idx_I = jnp.nonzero(jnp.where(~mask, 1, 0))[0]

    # Extract excitatory and inhibitory parts
    J_E = jnp.take(J_plus, idx_E, axis=0)  # shape: (N_E, N)
    J_I = jnp.take(J_plus, idx_I, axis=0)  # shape: (N_I, N)
    

    N_E = J_E.shape[0]
    N_I = J_I.shape[0]

    # Initialize factors for excitatory and inhibitory blocks
    U_E=jax.random.uniform(jax.random.PRNGKey(0), (N_E, D_E), minval=0.1, maxval=1.0)
    U_I=jax.random.uniform(jax.random.PRNGKey(1), (N_I, D_I), minval=0.1, maxval=1.0)
    V_E=jax.random.uniform(jax.random.PRNGKey(2), (J_plus.shape[0], D_E), minval=0.1, maxval=1.0) #shape: (N, D_E)
    V_I=jax.random.uniform(jax.random.PRNGKey(3), (J_plus.shape[0], D_I), minval=0.1, maxval=1.0) #shape: (N, D_I)

    #U_E, V_E = init_factors(jax.random.PRNGKey(0),shape_u=(N_E, D_E), shape_v=(J_plus.shape[0], D_E))
    #U_I, V_I = init_factors(jax.random.PRNGKey(1),shape_u=(N_I, D_I), shape_v=(J_plus.shape[0], D_I))

    # We apply NMF to each block using alternating non-negative least squares (NNLS).
    U_E, V_E = NMF(U_E, V_E, J_E)  # shape: (N_E, D_E), (N, D_E)
    U_I, V_I = NMF(U_I, V_I, J_I) # shape: (N_I, D_I), (N, D_I)

    return U_E, V_E, U_I, V_I

    


# ------------------------------------------------------------------------------
# Jittable Non-Negative Matrix Factorization via Alternating NNLS
# ------------------------------------------------------------------------------
@jax.jit
def NMF(U_init, V_init, J, max_iterations=100000, relative_error=1e-8):

    def cond_fun(state):
        """
        Condition for while loop termination:
          - max_iterations reached
          - relative error below threshold
          - change in error below threshold
        """
        i, U, V = state
        numerator = jnp.linalg.norm(J- U @ V.T, ord='fro')**2
        denominator = jnp.linalg.norm(J, ord='fro')**2
        return (i==max_iterations) | (numerator / denominator < relative_error)


    def body_fun(state):
        i, U, V = state

        #--------------U Step---------------
        #  Fix V, solve for each col uᵢ ∈ {1,...,N_E}:
        #  min_{uᵢ ≥ 0} ||VU.T[:, i] -J[:, i] ||_2²
        
        Q1= V.T@V  #shape (D_E, D_E)
        C1 = -2.0 * (J @ V)     # shape (N_E, DE) 

        #masks all True since doing non-negative least squares
        #masks = jnp.full((J.shape[0],Q1.shape[0]), True, dtype=bool) #shape (N_E, D_E)
        
        vmap_solver = jax.vmap(NNLS, in_axes=(None, 1)) #vmap over each column of C1.T which is shape (D_E, 1)
        U_new=vmap_solver(Q1, C1.T)  # shape (N_E, D_E)

        #--------------V Step---------------
        #  Fix U, solve for each col vⱼ ∈ {1,...,N}:
        #  min_{vⱼ ≥ 0} ||UV[:, j] - J[:, j] ||_2²
        Q2 = U_new.T @ U_new #shape (D_E, D_E)
        C2 = -2.0 * ( J.T @ U_new)  # shape  (N, D_E)
       
        #masks = jnp.full((J.shape[1], Q2.shape[0]), True, dtype=bool) # shape (N, D_E)
        vmap_solver = jax.vmap(NNLS, in_axes=(None, 1)) #Vmap over each column of C2.T which is shape (D_E, 1)
        V_new = vmap_solver(Q2, C2.T)  # shape (N, D_E)

        return (i + 1, U_new, V_new)
    
    # Initialize state
    init_state = (0, U_init, V_init)

    final_state = jax.lax.while_loop(cond_fun, body_fun, init_state)
    _, U_final, V_final = final_state
    return U_final, V_final


def build_v_dale(V1: jnp.ndarray, V2: jnp.ndarray) -> jnp.ndarray:
    """
    Constructs the latent feedback matrix V_Dale = [ V1  |  -V2 ],
    where:
      - V1 contains feedback weights for excitatory latents (non-negative),
      - V2 contains feedback weights for inhibitory latents (flipped to be non-positive).
    
    Args:
        V1: jnp.ndarray of shape (N, K1), excitatory feedback matrix (non-negative)
        V2: jnp.ndarray of shape (N, K2), inhibitory feedback matrix (non-negative)
    
    Returns:
        V_dale: jnp.ndarray of shape (N, K1 + K2)
    """
    V2_neg = -V2                           # Flip sign: inhibitory → non-positive
    V_dale = jnp.concatenate([V1, V2_neg], axis=1)  # Horizontal concatenation
    return V_dale


def build_U(U1, U2):
    """
    Constructs block-diagonal emission matrix:
        U = [[U1,  0 ],
             [ 0,  U2]]
    
    Args:
        U1: shape (NE, K1), excitatory emission matrix
        U2: shape (NI, K2), inhibitory emission matrix

    Returns:
        U: shape (NE + NI, K1 + K2), block-diagonal emission matrix
    """
    NE, K1 = U1.shape
    NI, K2 = U2.shape

    top    = jnp.concatenate([U1, jnp.zeros((NE, K2))], axis=1)
    bottom = jnp.concatenate([jnp.zeros((NI, K1)), U2], axis=1)

    U = jnp.concatenate([top, bottom], axis=0)
    return U



# Step 6: initial A₀ = V_daleᵀ @ U
def build_A(U, V_dale):
    """
    Compute initial latent dynamics matrix:
      A = V_dale^T @ U
    """
    return V_dale.T @ U








def compute_sufficient_statistics(posterior) -> SufficientStats:
    """
    Compute sufficient statistics from smoothed posterior for EM updates.

    Args:
        posterior: PosteriorGSSMSmoothed
            Must have attributes:
            - smoothed_means:        (T, K)
            - smoothed_covariances:  (T, K, K)
            - smoothed_cross_covariances: (T-1, K, K)
            - marginal_loglik:       scalar

    Returns:
        SufficientStats: a NamedTuple of fixed-shape JAX arrays
    """
    mu = posterior.smoothed_means
    Sigma = posterior.smoothed_covariances
    cross = posterior.smoothed_cross_covariances

    # E[z_t z_t^T] = smoothed_covariances + outer(smoothed_means, smoothed_means)
    EzzT = Sigma + jnp.einsum("ti,tj->tij", mu, mu)

    return SufficientStats(
        latent_mean=mu,
        latent_second_moment=EzzT,
        cross_time_moment=cross,
        loglik=posterior.marginal_loglik,
        T=mu.shape[0]
    )
