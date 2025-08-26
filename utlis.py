import jax
import jax.numpy as jnp
from jaxopt import BoxOSQP
from typing import Optional, List, Tuple
from jaxtyping import Float, Array
from jaxopt import BoxCDQP
from params import SufficientStats, ParamsCTDSConstraints
import optax
from functools import partial

_boxCDQP = BoxCDQP(tol=1e-7, maxiter=10000, verbose=False) 
_boxOSQP = BoxOSQP(tol=1e-8, maxiter=50000, verbose=False, primal_infeasible_tol=1e-10, dual_infeasible_tol=1e-10)


#might change args to matvec functions
@jax.jit
def solve_dale_QP(Q, c, mask,key):
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
    lower = jnp.where(mask, 1e-12, -jnp.inf) #E cell lower bound is 0.0, I cell lower bound is -inf
    upper = jnp.where(mask, jnp.inf, -1e-12) # I cell upper bound is 0.0, E cell upper bound is inf

    init_x = jax.random.normal(key, (D,)) #TODO: find better way of intializing
    A = jnp.eye(D)  # Identity matrix as a placeholder for the linear constraints
    # Run the OSQP solver
    init_params= _boxOSQP.init_params(init_x=init_x,params_obj=(Q, c),  params_eq=A, params_ineq=(lower, upper))
    result = _boxOSQP.run( init_params=init_params, params_obj=(Q, c),  params_eq=A, params_ineq=(lower, upper))
    sol = result.params.primal[0]
    return sol


@jax.jit
def solve_constrained_QP(Q, c, mask, isExcitatory, init_x, key=jax.random.PRNGKey(0)):
    """Solve a constrained quadratic program with excitatory/inhibitory constraints.
    Args:
        Q (jnp.ndarray): Positive semi-definite matrix of shape (D, D).
        c (jnp.ndarray): Linear coefficient vector of shape (D,).
        mask (jnp.ndarray of bools): Shape (D,). 
            'False' → if diagonal element
            'True' → if off-diagonal element
        isExcitatory (bool): If True, solve for excitatory cells; if False, for inhibitory cells.
        key (jax.random.PRNGKey): Random key for initialization."""
    def true_fn(args):
        """
        all entries are nonnegative except diagonals. True=Non-negative, False=unconstrained diagonal
        """
        Q, c, mask, key, init_x = args
        D = Q.shape[0]
        lower = jnp.where(mask, 0, -jnp.inf)
        upper = jnp.where(mask, jnp.inf, jnp.inf)
        #init_x = jax.random.normal(key, (D,))
        A = jnp.eye(D)  # Identity matrix as a placeholder for the linear constraints
        init_params= _boxOSQP.init_params(init_x=init_x,params_obj=(Q, c),  params_eq=A, params_ineq=(lower, upper))
        sol = _boxOSQP.run(init_params=init_params, params_obj=(Q, c),  params_eq=A, params_ineq=(lower, upper)).params.primal[0]
        return sol
        #sol = _boxCDQP.run(init_x, params_obj=(Q, c), params_ineq=(lower, upper))
        #return sol.params


    def false_fn(args):
        """
        all entries are nonpositive except diagonals. True=Non-positive, False=unconstrained diagonal
        """
        Q, c, mask, key, init_x = args
        D = Q.shape[0]
        lower = jnp.where(mask, -jnp.inf, -jnp.inf)
        upper = jnp.where(mask, 0, jnp.inf)
        #init_x = jax.random.normal(key, (D,))
        #sol = _boxCDQP.run(init_x, params_obj=(Q, c), params_ineq=(lower, upper))
        #return sol.params
        A = jnp.eye(D)  # Identity matrix as a placeholder for the linear constraints
        init_params= _boxOSQP.init_params(init_x=init_x,params_obj=(Q, c),  params_eq=A, params_ineq=(lower, upper))
        sol = _boxOSQP.run(init_params=init_params, params_obj=(Q, c),  params_eq=A, params_ineq=(lower, upper))
        return sol.params.primal[0]

    return jax.lax.cond(isExcitatory, 
                    (Q, c, mask, key, init_x), true_fn, 
                    (Q, c, mask, key, init_x), false_fn)




@jax.jit
def jaxOpt_NNLS(Q, c, init_x):
    lower = jnp.zeros(c.shape[0])  # Non-negative constraint
    upper = jnp.inf * jnp.ones(c.shape[0])  # No upper bound
    #init_x = jax.random.uniform(jax.random.PRNGKey(42), (c.shape[0],), minval=0.0, maxval=1.0)
    #sol = _boxCDQP.run(init_x, params_obj=(Q, c), params_ineq=(lower, upper))
    #return sol.params
    A = jnp.eye(c.shape[0])  # Identity matrix as a placeholder for the linear constraints
    init_params= _boxOSQP.init_params(init_x=init_x,params_obj=(Q, c),  params_eq=A, params_ineq=(lower, upper))
    sol = _boxOSQP.run(init_params=init_params, params_obj=(Q, c),  params_eq=A, params_ineq=(lower, upper)).params.primal[0]
    return sol

@jax.jit
def Optax_NNLS(A, b, iters: Optional[int] = 1000):
    vec=optax.nnls(A, b, iters=iters)
    return vec



def blockwise_NNLS(Y, X, left_paddings, right_paddings, emission_dims, cell_type_dims, C_prev):
    C_blocks=[]
    for i in range(left_paddings.shape[0]):
        emission_dim=emission_dims[i]
        left_padding=left_paddings[i]
        right_padding=right_paddings[i]
        cell_type_dim=cell_type_dims[i]
        Y_type=Y[ :, emission_dim[0] : emission_dim[1]] #(T, N_type)
        X_type=X[:, left_padding[1] :left_padding[1]+ cell_type_dim ] #(T, D_type)
        #C_prev_block = C_prev[emission_dim[0] : emission_dim[1], left_padding[1] : left_padding[1] + cell_type_dim]  # shape (N, D_type)
        #Q = X_type.T @ X_type  # shape (D_type, D_type)
        #F = X_type.T @ Y_type  # shape (D_type, N_type)
        
        #Vmap over N columns of F. Each column corresponds to f= X @ y_i ∈ ℝ^D
        #vmap_solver2 = jax.vmap(jaxOpt_NNLS, in_axes=(None, 1, 1))
        #C = vmap_solver2(Q, F, C_prev_block.T)
        #C_blocks.append(jnp.concatenate([jnp.zeros(left_padding), C, jnp.zeros(right_padding)], axis=1))

        C_T=Optax_NNLS(X_type, Y_type)
        C_blocks.append(jnp.concatenate([jnp.zeros(left_padding), jnp.transpose(C_T), jnp.zeros(right_padding)], axis=1))
    
    C = jnp.concatenate(C_blocks, axis=0)

    return C

        








"""
#trying to use lax.scan for blockwise NNLS
def blockwise_NNLS(carry, index):
    Y, X, left_paddings, right_paddings, emission_dims, cell_type_dims =carry
    emission_dim=emission_dims[index]
    left_padding=left_paddings[index]
    right_padding=right_paddings[index]
    cell_type_dim=cell_type_dims[index]
    jax.lax.dynamic_slice
    Y_type=Y[ :, emission_dim[0] : emission_dim[1]] #(T, N_type)
    X_type=X[:, left_padding[1] :left_padding[1]+ cell_type_dim ] #(T, D_type)
    C_T=Optax_NNLS(X_type, Y_type)
    C=jnp.transpose(C_T)
    return carry, C
#partial(jax.jit, static_argnames=("left_padding", "right_padding", "emission_dim", "cell_type_dim"))
def blockwise_NNLS(Y, X, left_padding, right_padding, emission_dim, cell_type_dim ):

    Y_type=Y[ :, emission_dim[0] : emission_dim[1]] #(T, N_type)
    X_type=X[:, left_padding[1] :left_padding[1]+ cell_type_dim ] #(T, D_type)
    C_T=Optax_NNLS(X_type, Y_type)
    C=jnp.transpose(C_T)
    C_block = jnp.concatenate([jnp.zeros(left_padding), C, jnp.zeros(right_padding)], axis=1)

"""





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
#@jax.jit
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
    keys = jax.random.split(jax.random.PRNGKey(0), num=N)

    epsilon = 1e-3  # Small regularization constant (can tune this)
   #Q = 2 Xᵀ X∈ ℝ^{N × N}, positive semidefinite
    Q = 2.0 * (X.T @ X) + epsilon * jnp.eye(N)
    assert is_positive_semidefinite(Q), "Q is not positive semidefinite."
    assert check_qp_condition(Q), "Q is not well conditioned. Consider regularizing the model or checking the data."

    C = -2.0 * (X.T @ Y_future.T)            # shape (N, N) → column c_j is C[:, j] with shape (N,)
    #masks = jnp.tile(mask[None, :], (1, ))  #shape (N, N), tile mask for each neuron 
    masks = jnp.tile(mask[:, None], (1, N))  # shape (N, N)

    vmap_solver = jax.vmap(solve_dale_QP, in_axes=(None, 1, 1,0)) # vmap over each column of C and each row of Q
    J=vmap_solver(Q, C, masks, keys)  # shape (N, N)
  
    return J


#does not need to be jittable since it iterates over small number of cell types
def blockwise_NMF(J, cell_constraints:ParamsCTDSConstraints):
    """
    Perform block-wise Non-negative Matrix Factorization (NMF) on the absolute Dale matrix |J|.
    Each block corresponds to a specific cell type, and NMF is applied separately to each block.
    Args:
        J: (N, N) Dale matrix where J[i, j] is the weight from neuron j to neuron i.
        cell_constraints: ParamsCellConstraints containing cell type information.
            - cell_types: Array of shape (k,) with unique cell type labels.
            - cell_type_dimensions: Array of shape (k,) with latent dimensions per cell type.
            - cell_type_mask: Array of shape (N,) with cell type label for each neuron.
    Returns:
        List of (U, V) tuples for each cell type, where:
            - U: jnp.ndarray(N_type, D_type) matrix of basis vectors for this cell type.
            - V: jnp.ndarray(N, D_type) matrix of coefficients for this cell type.
    """
    J_plus = jnp.abs(J)  # J⁺ = |J|, where J⁺_{ij} = |J_{ij}|
    
    cell_types = cell_constraints.cell_types
    cell_type_dimensions = cell_constraints.cell_type_dimensions
    cell_type_mask = cell_constraints.cell_type_mask 
    
    N = J_plus.shape[0]
    num_cell_types = len(cell_types)

    # Initialize list to store (U, V) tuples for each cell type
    block_factors = []

    # using for loop since number of cell types is small and fixed
    for i, cell_type in enumerate(cell_types):
        # get all the indices for this cell type
        idx_type = jnp.nonzero(jnp.where(cell_type_mask == cell_type, 1, 0))[0] #array of indices where cell_type_mask matches current cell type
        
        # return the submatrix of J_plus corresponding to this cell type
        J_type = jnp.take(J_plus, idx_type, axis=0)  # shape: (N_type, N)
        
        N_type = J_type.shape[0]
        D_type = int(cell_type_dimensions[i])
        
        # Initialize factors for this cell type block
        U_type = jax.random.uniform(jax.random.PRNGKey(i), (N_type, D_type), minval=0.1, maxval=1.0)
        V_type = jax.random.uniform(jax.random.PRNGKey(i + num_cell_types), (N, D_type), minval=0.1, maxval=1.0)
        
        # Apply NMF to this block using alternating non-negative least squares (NNLS)
        U_type, V_type = NMF(U_type, V_type, J_type)
        
        # Store the (U, V) tuple for this cell type
        block_factors.append((U_type, V_type))
    
    return block_factors



@jax.jit
def NMF(U_init, V_init, J, max_iterations=1000, relative_error=1e-4):
    # J shape: (N_type, N)
    def cond_fun(state):
        """
        Condition for while loop termination:
          - max_iterations reached
          - relative error below threshold
          - change in error below threshold
        """
        i, U, V = state
        numerator = jnp.linalg.norm(J- U @ V.T, ord='fro')**2
        error = numerator / jnp.linalg.norm(J, ord='fro')**2
        return (i!=max_iterations) & ( relative_error <= error)


    def body_fun(state):
        i, U, V = state

        #--------------U Step---------------
        #  Fix V, solve for each col uᵢ ∈ {1,...,N_E}:
        #  min_{uᵢ ≥ 0} ||VU.T[:, i] -J[:, i] ||_2²
        
        Q1= 2.0 * V.T@V  #shape (D_E, D_E)
        C_T = -2.0 * (J @ V)     # shape (N_E, DE) 
        C1=C_T.T  # shape (DE, N_E)

        #masks all True since doing non-negative least squares
        #masks = jnp.full((J.shape[0],Q1.shape[0]), True, dtype=bool) #shape (N_E, D_E)

        vmap_solver = jax.vmap(jaxOpt_NNLS, in_axes=(None, 1,0)) #vmap over each column of C1.T which is shape (D_E, 1)
        U_new=vmap_solver(Q1, C1, U)  # shape (N_E, D_E)

        #--------------V Step---------------
        #  Fix U, solve for each col vⱼ ∈ {1,...,N}:
        #  min_{vⱼ ≥ 0} ||UV[:, j] - J[:, j] ||_2²
        Q2 = 2.0 * U_new.T @ U_new #shape (D_E, D_E)
        C2 = -2.0 * ( J.T @ U_new)  # shape  (N, D_E)
       #masks = jnp.full((J.shape[1], Q2.shape[0]), True, dtype=bool) # shape (N, D_E)
        vmap_solver = jax.vmap(jaxOpt_NNLS, in_axes=(None, 1,0)) #Vmap over each column of C2.T which is shape (D_E, 1)
        V_new = vmap_solver(Q2, C2.T, V)  # shape (N, D_E)

        return (i + 1, U_new, V_new)
    
    # Initialize state
    init_state = (0, U_init, V_init)

    final_state = jax.lax.while_loop(cond_fun, body_fun, init_state)
    _, U_final, V_final = final_state
    return U_final, V_final



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
        loglik=jnp.array(posterior.marginal_loglik, float),
        T=mu.shape[0]
    )




#TO DO: Multiregion estimate_J

def simple_blockwise_nmf(J,mask, D_E,D_I):
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



def check_qp_condition(Q: jnp.ndarray) -> bool:
    """
    Check the quadratic programming (QP) condition for the given matrix.
    """
    eigs=jnp.linalg.eigvalsh(Q)
    cond=jnp.max(eigs) / jnp.min(eigs)
    return cond < 1e8

def is_positive_semidefinite(Q: jnp.ndarray) -> bool:
    """
    Check if the given matrix is positive semidefinite.
    """
    eigs = jnp.linalg.eigvalsh(Q)
    return jnp.all(eigs >= -1e-10)

#def update_emissions_blockwise(C_blocks, Y_obs, X):

@jax.jit
def linreg_to_quadratic_form(A: jnp.ndarray, b: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Convert linear regression problem ||Ax - b||² to quadratic form ½x^T P x + q^T x + r
    
    The linear regression objective ||Ax - b||² expands to:
    (Ax - b)^T (Ax - b) = x^T A^T A x - 2 b^T A x + b^T b
    
    In standard quadratic form ½x^T P x + q^T x + r:
    - P = 2 * A^T A  (factor of 2 because of the ½ in standard form)
    - q = -2 * A^T b
    - r = b^T b
    
    Args:
        A: Design matrix of shape (m, n)
        b: Target vector of shape (m,)
        
    Returns:
        P: Quadratic coefficient matrix of shape (n, n)
        q: Linear coefficient vector of shape (n,)
        r: Constant scalar
    """
    # Compute A^T A for the quadratic term
    AtA = A.T @ A
    
    # Compute A^T b for the linear term  
    Atb = A.T @ b
    
    # Compute b^T b for the constant term
    btb = b.T @ b
    
    # Convert to standard quadratic form ½x^T P x + q^T x + r
    P = 2.0 * AtA
    q = -2.0 * Atb  
    r = btb
    
    return P, q, r