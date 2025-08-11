import jax
import jax.numpy as jnp
#from jaxopt import BoxOSQP
from typing import Optional, List, Tuple
from jaxopt import BoxCDQP
import jax.random as jr
from dynamax.linear_gaussian_ssm import PosteriorGSSMSmoothed
from params import SufficientStats, ParamsCTDSConstraints


_boxCDQP = BoxCDQP(tol=1e-7, maxiter=10000, verbose=False) 

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
def solve_constrained_QP(Q, c, mask, isExcitatory, key=jax.random.PRNGKey(0)):
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




#TODO: change to implementing with optax
#@jax.jit
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


def build_v_dale(V_list:List) -> jnp.ndarray:
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


def generate_synthetic_cell_type_data(
    N: int = 50,  # Total number of neurons
    T: int = 500,  # Number of time points
    D: int = 8,   # Total latent dimensions
    K: int = 2,   # Number of cell types
    excitatory_fraction: float = 0.7,  # Fraction of excitatory neurons
    noise_level: float = 0.1,  # Observation noise level
    dynamics_strength: float = 0.8,  # Dynamics eigenvalue magnitude
    seed: int = 42
) -> Tuple[jnp.ndarray, ParamsCTDSConstraints]:
    """
    Generate synthetic neural data with cell-type structure and Dale's law constraints.
    
    Args:
        N: Number of neurons
        T: Number of time points
        D: Total latent state dimensions
        K: Number of cell types (default 2: excitatory/inhibitory)
        excitatory_fraction: Fraction of neurons that are excitatory
        noise_level: Standard deviation of observation noise
        dynamics_strength: Maximum eigenvalue magnitude for stable dynamics
        seed: Random seed for reproducibility
        
    Returns:
        observations: (T, N) array of neural activity
        constraints: ParamsCTDSConstraints object with cell type information
    """
    
    key = jr.PRNGKey(seed)
    keys = jr.split(key, 10)
    
    # Define cell types and properties
    cell_types = jnp.arange(K)
    
    if K == 2:
        # Standard excitatory/inhibitory setup
        cell_sign = jnp.array([1, -1])  # +1 for excitatory, -1 for inhibitory
        cell_type_dimensions = jnp.array([D//2, D - D//2])  # Split dimensions
    else:
        # General multi-type setup
        cell_sign = jnp.concatenate([jnp.ones(K//2), -jnp.ones(K - K//2)])
        dims_per_type = D // K
        cell_type_dimensions = jnp.full(K, dims_per_type)
        # Distribute remaining dimensions
        remaining = D - dims_per_type * K
        cell_type_dimensions = cell_type_dimensions.at[:remaining].add(1)
    
    # Assign neurons to cell types
    n_excitatory = int(N * excitatory_fraction)
    n_inhibitory = N - n_excitatory
    
    if K == 2:
        cell_type_mask = jnp.concatenate([
            jnp.zeros(n_excitatory, dtype=int),  # Excitatory = type 0
            jnp.ones(n_inhibitory, dtype=int)    # Inhibitory = type 1
        ])
    else:
        # Distribute neurons across all cell types
        neurons_per_type = N // K
        cell_type_mask = jnp.repeat(jnp.arange(K), neurons_per_type)
        # Handle remainder
        remaining_neurons = N - neurons_per_type * K
        if remaining_neurons > 0:
            extra_assignments = jnp.arange(remaining_neurons)
            cell_type_mask = jnp.concatenate([cell_type_mask, extra_assignments])
    
    # Shuffle neuron assignments
    cell_type_mask = jr.permutation(keys[0], cell_type_mask)
    
    # Create stable dynamics matrix
    A_raw = jr.normal(keys[1], (D, D)) * 0.3
    eigenvals, eigenvecs = jnp.linalg.eig(A_raw)
    eigenvals_scaled = eigenvals * (dynamics_strength / jnp.max(jnp.abs(eigenvals)))
    A = jnp.real(eigenvecs @ jnp.diag(eigenvals_scaled) @ jnp.linalg.inv(eigenvecs))
    
    # Create emission matrix with Dale's law structure
    C = jr.uniform(keys[2], (N, D), minval=0.1, maxval=1.0)
    
    # Apply Dale's law constraints
    dynamics_mask = jnp.concatenate([
        jnp.ones(cell_type_dimensions[0]),  # Excitatory dimensions
        -jnp.ones(sum(cell_type_dimensions[1:]))  # Inhibitory dimensions
    ])
    
    # Modify emission matrix based on cell types and latent dimensions
    for i in range(N):
        neuron_type = cell_type_mask[i]
        neuron_sign = cell_sign[neuron_type]
        
        for j in range(D):
            dim_sign = dynamics_mask[j]
            
            # Apply Dale's law: inhibitory neurons have negative connections to inhibitory dims
            if neuron_type > 0 and dim_sign < 0:  # Inhibitory neuron to inhibitory dimension
                C = C.at[i, j].set(-C[i, j])
    
    # Process and observation noise
    Q = jnp.diag(jr.uniform(keys[3], (D,), minval=0.05, maxval=0.15))
    R = jnp.diag(jr.uniform(keys[4], (N,), minval=noise_level*0.5, maxval=noise_level*1.5))
    
    # Generate latent states
    x = jnp.zeros((T, D))
    x = x.at[0].set(jr.normal(keys[5], (D,)) * 0.5)  # Initial state
    
    for t in range(1, T):
        # x_t = A @ x_{t-1} + process_noise
        process_noise = jr.multivariate_normal(keys[6], jnp.zeros(D), Q)
        x = x.at[t].set(A @ x[t-1] + process_noise)
        keys = jr.split(keys[6], 2)  # Update key for next iteration
        keys = jnp.concatenate([keys, jr.split(keys[0], 8)])
    
    # Generate observations
    observations = jnp.zeros((T, N))
    for t in range(T):
        # y_t = C @ x_t + observation_noise  
        obs_noise = jr.multivariate_normal(keys[7], jnp.zeros(N), R)
        observations = observations.at[t].set(C @ x[t] + obs_noise)
        keys = jr.split(keys[7], 2)
        keys = jnp.concatenate([keys, jr.split(keys[0], 8)])
    
    # Create constraints object
    constraints = ParamsCTDSConstraints(
        cell_types=cell_types,
        cell_sign=cell_sign,
        cell_type_dimensions=cell_type_dimensions,
        cell_type_mask=cell_type_mask
    )
    
    return observations.T, constraints


def generate_small_demo_data(seed: int = 42) -> Tuple[jnp.ndarray, ParamsCTDSConstraints]:
    """Generate small dataset for quick demos and testing."""
    return generate_synthetic_cell_type_data(
        N=20, T=100, D=4, K=2, 
        excitatory_fraction=0.7, 
        noise_level=0.1, 
        seed=seed
    )


def generate_medium_realistic_data(seed: int = 42) -> Tuple[jnp.ndarray, ParamsCTDSConstraints]:
    """Generate medium-sized dataset with realistic parameters."""
    return generate_synthetic_cell_type_data(
        N=50, T=500, D=8, K=2,
        excitatory_fraction=0.65,
        noise_level=0.08,
        seed=seed
    )


def generate_large_dataset(seed: int = 42) -> Tuple[jnp.ndarray, ParamsCTDSConstraints]:
    """Generate large dataset for performance testing."""
    return generate_synthetic_cell_type_data(
        N=100, T=1000, D=12, K=3,
        excitatory_fraction=0.6,
        noise_level=0.05,
        seed=seed
    )


def generate_multi_type_data(n_types: int = 3, seed: int = 42) -> Tuple[jnp.ndarray, ParamsCTDSConstraints]:
    """Generate data with multiple cell types (beyond excitatory/inhibitory)."""
    return generate_synthetic_cell_type_data(
        N=60, T=400, D=12, K=n_types,
        excitatory_fraction=0.5,  # Will be adjusted for multi-type
        noise_level=0.08,
        seed=seed
    )