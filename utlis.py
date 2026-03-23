import jax
import jax.numpy as jnp
#from jaxopt import BoxOSQP
from typing import Optional, List, Tuple
from jaxtyping import Float, Array
from jaxopt import BoxCDQP, BoxOSQP
from params import SufficientStats, ParamsCTDSConstraints
import optax

_boxCDQP = BoxCDQP(tol=1e-7, maxiter=10000, verbose=False) 
_boxOSQP = BoxOSQP(tol=1e-8, maxiter=50000, verbose=False, primal_infeasible_tol=1e-10, dual_infeasible_tol=1e-10)

#might change args to matvec functions
#@jax.jit
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


#@jax.jit
def solve_constrained_QP(Q, c, mask, isExcitatory, init_x):
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
        all entries are nonnegative except diagonals. True=Non-neggative, False=unconstrained diagonal
        """
        Q, c, mask, key = args
        D = Q.shape[0]
        lower = jnp.where(mask, 0, -jnp.inf)
        upper = jnp.where(mask, jnp.inf, jnp.inf)
        #init_x = jax.random.normal(key, (D,))
        sol = _boxCDQP.run(init_x, params_obj=(Q, c), params_ineq=(lower, upper))
        return sol.params

    def false_fn(args):
        """
        all entries are nonpositive except diagonals. True=Non-positive, False=unconstrained diagonal
        """
        Q, c, mask, init_x = args
        D = Q.shape[0]
        lower = jnp.where(mask, -jnp.inf, -jnp.inf)
        upper = jnp.where(mask, 0, jnp.inf)
        #init_x = jax.random.normal(key, (D,))
        sol = _boxCDQP.run(init_x, params_obj=(Q, c), params_ineq=(lower, upper))
        return sol.params

    return jax.lax.cond(isExcitatory, 
                    (Q, c, mask, init_x), true_fn, 
                    (Q, c, mask, init_x), false_fn)



@jax.jit
def jaxOpt_NNLS(Q, c, init_x):
    """
    Solve non-negative least squares using quadratic programming.

    Solves the constrained quadratic program:
        minimize    (1/2) * x^T Q x + c^T x
        subject to:  x >= 0

    Parameters
    ----------
    Q : Array, shape (D, D)
        Positive semi-definite Hessian matrix.
    c : Array, shape (D,)
        Linear coefficient vector.
    init_x : Array, shape (D,)
        Initial solution guess.

    Returns
    -------
    solution : Array, shape (D,)
        Non-negative optimal solution.

    Notes
    -----
    Uses BoxCDQP solver with non-negativity constraints.
    Equivalent to solving min ||Ax - b||^2 s.t. x >= 0
    where Q = A^T A and c = -A^T b.
    """
    lower = jnp.zeros(c.shape[0])  # Non-negative constraint
    upper = jnp.inf * jnp.ones(c.shape[0])  # No upper bound
    #A = jnp.eye(c.shape[0])  # Identity matrix placeholder
    #init_params = _boxOSQP.init_params(init_x=init_x, params_obj=(Q, c), params_eq=A, params_ineq=(lower, upper))
    #sol = _boxOSQP.run(init_params=init_params, params_obj=(Q, c), params_eq=A, params_ineq=(lower, upper)).params.primal[0]
    sol = _boxCDQP.run(init_x, params_obj=(Q, c), params_ineq=(lower, upper)).params
    return sol


@jax.jit
def Optax_NNLS(A, b, iters: Optional[int] = 1000):
    """
    Solve non-negative least squares using Optax.

    Solves: minimize ||Ax - b||^2 subject to x >= 0

    Parameters
    ----------
    A : Array, shape (M, N)
        Design matrix.
    b : Array, shape (M,)
        Target vector.
    iters : int, optional
        Maximum number of iterations (default: 1000).

    Returns
    -------
    solution : Array, shape (N,)
        Non-negative least squares solution.

    Notes
    -----
    Uses Optax's built-in NNLS solver which implements
    an active-set algorithm.
    """
    vec = optax.nnls(A, b, iters=iters)
    return vec

def blockwise_NNLS(Y, X, left_paddings, right_paddings, emission_dims, cell_type_dims, C_prev):
    C_blocks=[]
    for i in range(left_paddings.shape[0]):
        emission_dim=emission_dims[i]
        left_padding=left_paddings[i]
        right_padding=right_paddings[i]
        cell_type_dim=cell_type_dims[i]
        #Y_type=Y[ :, emission_dim[0] : emission_dim[1]] #(T, N_type)
        #X_type=X[:, left_padding[1] :left_padding[1]+ cell_type_dim ] #(T, D_type)
        Y_type=Y[  left_padding[1] :left_padding[1]+ cell_type_dim, emission_dim[0] : emission_dim[1] ] #(D_type, N_type)
        X_type=X[left_padding[1] :left_padding[1]+ cell_type_dim , left_padding[1] :left_padding[1]+ cell_type_dim ]  #(D_type, D_type)


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

    epsilon = 1e-3  # Small regularization constant (can tune this)
   #Q = 2 Xᵀ X∈ ℝ^{N × N}, positive semidefinite
    Q = 2.0 * (X.T @ X) + epsilon * jnp.eye(N)
    assert is_positive_semidefinite(Q), "Q is not positive semidefinite."
    assert check_qp_condition(Q), "Q is not well conditioned. Consider regularizing the model or checking the data."

    C = -2.0 * (X.T @ Y_future.T)            # shape (N, N) → column c_j is C[:, j] with shape (N,)
    masks = jnp.tile(mask[None, :], (N, 1))  #shape (N, N), tile mask for each neuron 

    vmap_solver = jax.vmap(solve_dale_QP, in_axes=(None, 1, 0)) # vmap over each column of C and each row of Q
    J=vmap_solver(Q, C, masks)  # shape (N, N)
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




#@jax.jit
def blockwise_NMF_jit(J, cell_constraints:ParamsCTDSConstraints):
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
        Tuple of (U_factors, V_factors) where:
            - U_factors: jnp.ndarray of shape (max_N_type, max_D_type, K) containing all U matrices
            - V_factors: jnp.ndarray of shape (N, max_D_type, K) containing all V matrices
            K is the number of cell types, padded with zeros for smaller dimensions
    """
    J_plus = jnp.abs(J)  # J⁺ = |J|, where J⁺_{ij} = |J_{ij}|
    
    cell_types = cell_constraints.cell_types
    cell_type_dimensions = cell_constraints.cell_type_dimensions
    cell_type_mask = cell_constraints.cell_type_mask 
    
    N = J_plus.shape[0]
    K = len(cell_types)  # Number of cell types
    
    # Find maximum dimensions for padding
    max_D_type = jnp.max(cell_type_dimensions).astype(int)
    
    # Calculate number of neurons per cell type
    cell_type_counts = jnp.array([
        jnp.sum(cell_type_mask == cell_type) for cell_type in cell_types
    ])
    max_N_type = jnp.max(cell_type_counts).astype(int)
    
    # Pre-allocate arrays for all factors (padded)
    U_factors = jnp.zeros((max_N_type, max_D_type, K))
    V_factors = jnp.zeros((N, max_D_type, K))
    
    def process_cell_type(i, arrays):
        U_factors, V_factors = arrays
        cell_type = cell_types[i]
        D_type = cell_type_dimensions[i].astype(int)
        
        # Create mask for this cell type
        type_mask = (cell_type_mask == cell_type)
        
        # Get indices and extract submatrix
        idx_type = jnp.where(type_mask, size=max_N_type, fill_value=0)[0]
        N_type = jnp.sum(type_mask).astype(int)
        
        # Extract relevant rows using advanced indexing
        J_type = J_plus[idx_type[:N_type], :]  # shape: (N_type, N)
        
        # Initialize factors for this cell type block
        key_u = jax.random.PRNGKey(i)
        key_v = jax.random.PRNGKey(i + K)
        
        U_type = jax.random.uniform(key_u, (N_type, D_type), minval=0.1, maxval=1.0)
        V_type = jax.random.uniform(key_v, (N, D_type), minval=0.1, maxval=1.0)
        
        # Apply NMF to this block
        U_type, V_type = NMF(U_type, V_type, J_type)
        
        # Store in padded arrays
        U_factors = U_factors.at[:N_type, :D_type, i].set(U_type)
        V_factors = V_factors.at[:, :D_type, i].set(V_type)
        
        return U_factors, V_factors
    
    # Use fori_loop for JAX-compatible iteration
    initial_arrays = (U_factors, V_factors)
    final_U, final_V = jax.lax.fori_loop(0, K, process_cell_type, initial_arrays)
    
    return final_U, final_V


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

def compute_sufficient_statistics(posterior, emissions) -> SufficientStats:
    """
    Compute sufficient statistics from smoothed posterior for EM algorithm.

    Transforms Dynamax posterior results into the sufficient statistics
    format required for CTDS M-step parameter updates.

    Parameters
    ----------
    posterior : PosteriorGSSMSmoothed
        Dynamax smoother output containing:
        - smoothed_means : Array, shape (T, K)
            Posterior state means E[x_t | y_{1:T}]
        - smoothed_covariances : Array, shape (T, K, K)  
            Posterior state covariances Cov[x_t | y_{1:T}]
        - smoothed_cross_covariances : Array, shape (T-1, K, K)
            Cross-time moment E[x_t, x_{t+1} | y_{1:T}]
        - marginal_loglik : float
            Marginal log-likelihood p(y_{1:T})

    Returns
    -------
    stats : SufficientStats
        Sufficient statistics containing:
        - latent_mean : Array, shape (T, K)
            E[x_t | y_{1:T}]
        - latent_second_moment : Array, shape (T, K, K)
            E[x_t x_t^T | y_{1:T}] = Cov + mean ⊗ mean  
        - cross_time_moment : Array, shape (T-1, K, K)
            E[x_t x_{t-1}^T | y_{1:T}]
        - loglik : float
            Marginal log-likelihood
        - T : int
            Sequence length

    Notes
    -----
    Second moments are computed using the identity:
    E[x_t x_t^T] = Cov[x_t] + E[x_t] E[x_t]^T
    
    These statistics are sufficient for maximum likelihood estimation
    in the M-step of the EM algorithm.
    """
    mu = posterior.smoothed_means
    Sigma = posterior.smoothed_covariances
    cross = posterior.smoothed_cross_covariances

    # E[x_t x_t^T] = smoothed_covariances + outer(smoothed_means, smoothed_means)
    ExxT = Sigma + jnp.einsum("ti,tj->tij", mu, mu)
    Mxx=jnp.sum(ExxT, axis=0) # (K, K) sum over time of E[x_t x_t^T]
    Mdelta = jnp.sum(cross, axis=0) # (K, K) sum over time of E[x_t x_{t-1}^T]
    Ytil=emissions.T @ mu # (N, K) sum over time of y_t E[x_t]^T
    Mt_1=jnp.sum(ExxT[-1,:,:], axis=0)
    M2_T=jnp.sum(ExxT[1:,:,:], axis=0)


    return SufficientStats(
        latent_mean=mu,
        latent_second_moment=ExxT,
        cross_time_moment=cross,
        loglik=jnp.array(posterior.marginal_loglik, float),
        T=emissions.shape[0],
        Mxx=Mxx,
        Mdelta=Mdelta,
        Mt_1=Mt_1,
        M2_T=M2_T,
        Ytil=Ytil)
    


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


def _fit_A_from_pairs(
    X_prev: jnp.ndarray,       # (M, D)
    X_next: jnp.ndarray,       # (M, D)
    dynamics_mask: jnp.ndarray, # (D,)
    ridge: float = 1e-5,
) -> jnp.ndarray:
    """
    Fit A: (D, D) from pairs (X_prev, X_next) then project to Dale cone.

    LS:   A.T = (X_prev.T X_prev + ridge I)^{-1} X_prev.T X_next   → A.T is (D,D)
    Dale: project off-diagonal signs per column.
    Scale spectral radius to 0.95.

    Returns: (D, D).
    """
    D = X_prev.shape[1]
    XtX = X_prev.T @ X_prev + ridge * jnp.eye(D)    # (D, D)
    XtXnext = X_prev.T @ X_next                      # (D, D)
    A_T = jnp.linalg.solve(XtX, XtXnext)             # A.T : (D, D)
    A = A_T.T

    A = _project_A_dale(A, dynamics_mask)

    sr = jnp.max(jnp.abs(jnp.linalg.eigvals(A)))
    A = jnp.where(sr > 1e-8, A * (0.95 / (sr + 1e-8)), A)
    return A


def _fit_Q_from_pairs(
    X_prev: jnp.ndarray,   # (M, D)
    X_next: jnp.ndarray,   # (M, D)
    A: jnp.ndarray,        # (D, D)
    ridge: float = 1e-5,
) -> jnp.ndarray:
    """
    Compute Q from residual covariance.

    resid = X_next - (A @ X_prev.T).T     (M, D)
    Q     = resid.T @ resid / M  +  ridge I

    Returns: (D, D) PSD.
    """
    resid = X_next - (A @ X_prev.T).T     # (M, D)
    Q_raw = resid.T @ resid / resid.shape[0] + ridge * jnp.eye(A.shape[0])
    return _psd_project(Q_raw, floor=ridge)

#Functions for Initilization
def single_PCA(Y_n, d_n):
    # Y shape is ( N_type, T)
    U_n,S_n,V_n=jnp.linalg.svd(Y_n, full_matrices=False)
    C_n=U_n[:,:d_n] #( N_type,D_type)
    X_n=S_n[:d_n, None]*V_n[:d_n, :] #(D_type, T)
    return C_n, X_n.T

def Blockwise_PCA(Y, cell_type_dimensions, cell_type_mask,cell_types):
    #Y shape (N,T)
    N=Y.shape[0]
    C_blocks=[]
    X_blocks=[]
    Y_blocks=[]

    # using for loop since number of cell types is small and fixed
    for i, cell_type in enumerate(cell_types):
        # get all the indices for this cell type
        idx_type = jnp.nonzero(jnp.where(cell_type_mask == cell_type, 1, 0))[0] # (N_type) array of indices where cell_type_mask matches current cell type
        
        
        Y_type = jnp.take(Y, idx_type, axis=0)  # shape: (N_type, T)
        
        N_type = Y_type.shape[0]
        D_type = int(cell_type_dimensions[i])

        C_type,X_type=single_PCA(Y_type, D_type)
        C_blocks.append(C_type)
        X_blocks.append(X_type)
        Y_blocks.append(Y_type.T)
    
    return jnp.array(C_blocks), jnp.array(X_blocks), jnp.array(Y_blocks)



def _symmetrize(M: jnp.ndarray) -> jnp.ndarray:
    """Force matrix to be symmetric: (M + M.T) / 2."""
    return 0.5 * (M + M.T)


def _psd_project(M: jnp.ndarray, floor: float = 1e-6) -> jnp.ndarray:
    """
    Project (D, D) matrix to PSD via eigenvalue flooring.

    Returns: (D, D) symmetric PSD matrix.
    """
    M = _symmetrize(M)
    w, V = jnp.linalg.eigh(M)          # w: (D,), V: (D,D)
    w = jnp.maximum(w, floor)
    return (V * w) @ V.T               # (D, D)


def _project_A_dale(
    A: jnp.ndarray,                    # (D, D)
    dynamics_mask: jnp.ndarray,        # (D,)  +1 excitatory, -1 inhibitory
    eps: float = 1e-6,
) -> jnp.ndarray:
    """
    Project A to satisfy CTDS Dale's law (column-wise sign constraints).

    Off-diagonal column j:
        dynamics_mask[j] == +1  →  A[i,j] >= 0   (excitatory column)
        dynamics_mask[j] == -1  →  A[i,j] <= 0   (inhibitory column)
    Diagonal entries are left unconstrained.

    Args:
        A             : (D, D) raw dynamics matrix.
        dynamics_mask : (D,)   +1 or -1 per latent dim.
        eps           : small non-negative slack kept for off-diag constraint.

    Returns: (D, D) projected dynamics matrix.
    """
    D = A.shape[0]
    diag_mask = jnp.eye(D, dtype=bool)   # (D, D)  True on diagonal

    # For each column j, off-diagonal entries must have sign dynamics_mask[j].
    # excitatory (mask==+1): clamp negative off-diag values to 0
    # inhibitory (mask==-1): clamp positive off-diag values to 0
    col_signs = dynamics_mask[None, :]   # (1, D)  broadcast over rows

    # off-diagonal excitatory: max(A, 0)
    A_exc = jnp.maximum(A, 0.0)
    # off-diagonal inhibitory: min(A, 0)
    A_inh = jnp.minimum(A, 0.0)

    A_constrained = jnp.where(col_signs == 1, A_exc, A_inh)   # (D, D)

    # Restore unconstrained diagonal
    A_out = jnp.where(diag_mask, A, A_constrained)
    return A_out


def _normalize_latents(
    X_flat: jnp.ndarray,   # (BT, D)
    C: jnp.ndarray,        # (N, D)
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Normalise each latent dim to unit variance; rescale C accordingly.

    scale[d] = std(X_flat[:, d]) + eps
    X_norm   = X_flat / scale             (BT, D)
    C_norm   = C * scale                  (N, D)  — preserves C x = C_norm x_norm

    Returns: X_norm (BT, D), C_norm (N, D).
    """
    scale = jnp.std(X_flat, axis=0) + 1e-8   # (D,)
    X_norm = X_flat / scale[None, :]
    C_norm = C * scale[None, :]
    return X_norm, C_norm
def update_C_block(X, Y):
    """
    X: (TB, D)
    Y: (TB, N)
    """
    C= Optax_NNLS(X,Y) #(D, N)
    return C.T

def update_X_block(C, Y):
    X = jnp.linalg.lstsq(C, Y.T)[0].T  # ( T, D)
    X,C=_normalize_latents(X,C)
    return X,C

def init_emissions(iters, C_list, X_list, Y_list):
    def body_fun(i, params):
        C_list, X_list=params
        C_list=jax.vmap(update_C_block, in_axes=(0,0))(X_list, Y_list)
        X_list, C_list=jax.vmap(update_X_block, in_axes=(0,0))(C_list, Y_list)
        return C_list, X_list

    init = (C_list, X_list)  # initial carry
    C_final, X_final = jax.lax.fori_loop(0, iters, body_fun, init)
    return C_final, X_final



def pad_C(C_list, state_dim):
    C_blocks=[]
    col_start = 0
    emission_dims = []
    left_padding_dims = []
    right_padding_dims = []
    emiss_start = 0
    
    for C in C_list:
        N_type, K_type = C.shape
        
        # Create padded block: [zeros_left, U, zeros_right]
        left_pad = jnp.zeros((N_type, col_start))
        right_pad = jnp.zeros((N_type, state_dim - col_start - K_type))
        
        left_padding_dims.append(left_pad.shape)
        right_padding_dims.append(right_pad.shape)

        C_block = jnp.concatenate([left_pad, C, right_pad], axis=1)
        C_blocks.append(C_block)

        emission_dims.append((emiss_start, emiss_start + N_type))

        emiss_start += N_type
     
        # Concatenate vertically to form complete emission matrix
    
    C = jnp.concatenate(C_blocks, axis=0)  # Shape: (N, D)

    return C, emission_dims, left_padding_dims, right_padding_dims
from typing import Tuple, Dict
def pca_initialize_ctds(
    Y: jnp.ndarray,                     # (B, T, N)
    e_mask: jnp.ndarray,                # (N,) bool
    i_mask: jnp.ndarray,                # (N,) bool
    D: jnp.ndarray,
    cell_types: jnp.ndarray,            # (K,)
    cell_sign: jnp.ndarray,             # (K,)
    cell_type_dimensions: jnp.ndarray,  # (K,)
    cell_type_mask: jnp.ndarray,        # (N,)
    key: jax.random.PRNGKey = jax.random.PRNGKey(0),
    pgd_iters: int = 400,
    pgd_ridge: float = 1e-4,
) -> Dict[str, jnp.ndarray]:
    """
    PCA-based CTDS initialization.

    Args:
        Y                   : (B, T, N)
        e_mask              : (N,) bool
        i_mask              : (N,) bool
        D_e                 : int
        D_i                 : int
        cell_types          : (K,)
        cell_sign           : (K,)
        cell_type_dimensions: (K,)
        cell_type_mask      : (N,)
        key                 : PRNG key (used only for PGD init)
        pgd_iters           : PGD iters for C regression
        pgd_ridge           : ridge for XtX

    Returns dict with keys:
        X0 (B,T,D), C0 (N,D), d0 (N,), R0 (N,), A0 (D,D), Q0 (D,D)
    """
    B, T, N = Y.shape
    # ---- Step 1: Preprocess ----
    Y_flat = Y.reshape(B * T, N)          # (BT, N)
    d0 = jnp.mean(Y_flat, axis=0)         # (N,)
    Y_centered = Y_flat - d0[None, :]     # (BT, N)

    C_list, X_list, Y_list=Blockwise_PCA(Y_centered.T, cell_type_dimensions, cell_type_mask,cell_types)
    C_list, X_list= init_emissions(1000, C_list, X_list, Y_list)
    C0, emission_dims, left_padding_dims, right_padding_dims=pad_C(C_list, D)
    X_flat_norm=X_list.reshape(-1, D)

    # ---- Step 5: R from residual variance ----
    Y_pred = (C0 @ X_flat_norm.T).T      # (BT, N)
    resid_Y = Y_centered - Y_pred        # (BT, N)
    R0 = jnp.mean(resid_Y ** 2, axis=0) + 1e-6    # (N,)

    # ---- Step 6: A and Q ----
    X0 = X_flat_norm.reshape(B, T, D)
    dynamics_mask = jnp.repeat(cell_sign, cell_type_dimensions)   # (D,)
    X_prev_list = X0[:, :-1, :].reshape(B * (T - 1), D)
    X_next_list = X0[:, 1:, :].reshape(B * (T - 1), D)

    A0 = _fit_A_from_pairs(X_prev_list, X_next_list, dynamics_mask)
    Q0 = _fit_Q_from_pairs(X_prev_list, X_next_list, A0)

    return dict(X0=X0, C0=C0, d0=d0, R0=R0, A0=A0, Q0=Q0)