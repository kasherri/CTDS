import jax
import jax.numpy as jnp
import jax.random as jr
from typing import Optional, List, Tuple
from jaxtyping import Float, Array
from params import ParamsCTDSConstraints,ParamsCTDS,ParamsCTDSEmissions,ParamsCTDSDynamics,ParamsCTDSInitial
from models import CTDS
import os

#Plotting Utilis
def save_figure(exp_group_number, fig, name, section):
    """Save figure (PNG only) into Experiment Group #/Section <section>/."""
    folder = os.path.join(f"Exp group {exp_group_number}", f"Section {section}")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{name}.png")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {path}")



# ============================================================================
# ASSERTIONS
# ============================================================================

def assert_psd(matrix: jnp.ndarray, name: str = "matrix", tol: float = -1e-8):
    """Assert matrix is symmetric and positive semi-definite."""
    assert matrix.ndim == 2, f"{name} must be 2D"
    assert matrix.shape[0] == matrix.shape[1], f"{name} must be square"
    
    # Check symmetry
    symmetric_error = jnp.max(jnp.abs(matrix - matrix.T))
    assert symmetric_error < 1e-6, f"{name} not symmetric: max error = {symmetric_error}"
    
    # Check eigenvalues
    eigenvals = jnp.linalg.eigvalsh(matrix)
    min_eig = jnp.min(eigenvals)
    assert min_eig >= tol, f"{name} not PSD: min eigenvalue = {min_eig}"


def assert_dale_columns(A: jnp.ndarray, dynamics_mask: jnp.ndarray, tol: float = 1e-6):
    """
    Assert Dale's law constraints on A matrix.
    
    For each column j:
    - If dynamics_mask[j] == 1 (excitatory): off-diagonal A[i,j] >= -tol
    - If dynamics_mask[j] == -1 (inhibitory): off-diagonal A[i,j] <= tol
    """
    D = A.shape[0]
    assert A.shape == (D, D), f"A must be square, got {A.shape}"
    assert dynamics_mask.shape == (D,), f"dynamics_mask must have length D={D}"
    
    for j in range(D):
        col = A[:, j]
        sign = dynamics_mask[j]
        
        # Get off-diagonal entries
        off_diag_mask = jnp.arange(D) != j
        off_diag = col[off_diag_mask]
        
        if sign == 1:  # Excitatory
            violations = off_diag < -tol
            if jnp.any(violations):
                violating_vals = off_diag[violations]
                raise AssertionError(
                    f"Excitatory column {j} has negative off-diagonal entries: {violating_vals}"
                )
        elif sign == -1:  # Inhibitory
            violations = off_diag > tol
            if jnp.any(violations):
                violating_vals = off_diag[violations]
                raise AssertionError(
                    f"Inhibitory column {j} has positive off-diagonal entries: {violating_vals}"
                )


def assert_nonnegative(C: jnp.ndarray, name: str = "C", tol: float = -1e-6):
    """Assert matrix has all non-negative entries (within tolerance)."""
    min_val = jnp.min(C)
    if min_val < tol:
        violations = jnp.sum(C < tol)
        raise AssertionError(
            f"{name} has negative entries: min={min_val}, {violations} violations"
        )


def check_kkt_conditions(x: jnp.ndarray, P: jnp.ndarray, q: jnp.ndarray, 
                         lb: jnp.ndarray, ub: jnp.ndarray, tol: float = 1e-4) -> dict:
    """
    Check KKT optimality conditions for box-constrained QP:
    min 0.5 x^T P x - q^T x  s.t. lb <= x <= ub
    
    Returns dict with violation counts and max violations.
    """
    g = P @ x - q  # Gradient at solution
    
    # Classify variables
    at_lower = jnp.abs(x - lb) < tol
    at_upper = jnp.abs(x - ub) < tol
    interior = ~at_lower & ~at_upper
    
    # KKT conditions:
    # Interior: g ≈ 0
    # At lower bound: g >= -tol (multiplier for lb is non-negative)
    # At upper bound: g <= tol (multiplier for ub is non-negative)
    
    interior_violations = interior & (jnp.abs(g) > tol)
    lower_violations = at_lower & (g < -tol)
    upper_violations = at_upper & (g > tol)
    
    return {
        'interior_count': jnp.sum(interior),
        'interior_violations': jnp.sum(interior_violations),
        'interior_max_error': jnp.max(jnp.where(interior, jnp.abs(g), 0.0)),
        'lower_violations': jnp.sum(lower_violations),
        'lower_max_error': jnp.max(jnp.where(lower_violations, -g, 0.0)),
        'upper_violations': jnp.sum(upper_violations),
        'upper_max_error': jnp.max(jnp.where(upper_violations, g, 0.0)),
        'total_violations': jnp.sum(interior_violations) + jnp.sum(lower_violations) + jnp.sum(upper_violations)
    }


# ============================================================================
# SYNTHETIC DATA GENERATION
# ============================================================================

def generate_stable_A(D: int, dynamics_mask: jnp.ndarray, key: jax.random.PRNGKey, 
                     spectral_radius: float = 0.95) -> jnp.ndarray:
    """
    Generate stable A matrix satisfying Dale's law constraints.
    
    Strategy:
    1. Generate random matrix
    2. Apply Dale constraints column-wise
    3. Scale to desired spectral radius
    """
    key1, key2 = jax.random.split(key)
    
    # Start with random matrix
    A = jax.random.normal(key1, (D, D)) * 0.1
    
    # Apply Dale constraints column by column
    for j in range(D):
        if dynamics_mask[j] == 1:  # Excitatory
            # Set off-diagonal to positive
            off_diag_mask = jnp.arange(D) != j
            A = A.at[off_diag_mask, j].set(jnp.abs(A[off_diag_mask, j]))
        elif dynamics_mask[j] == -1:  # Inhibitory
            # Set off-diagonal to negative
            off_diag_mask = jnp.arange(D) != j
            A = A.at[off_diag_mask, j].set(-jnp.abs(A[off_diag_mask, j]))
    
    # Scale to desired spectral radius
    current_radius = jnp.max(jnp.abs(jnp.linalg.eigvals(A)))
    if current_radius > 1e-8:
        A = A * (spectral_radius / current_radius)
    
    return A


def generate_nonnegative_C(N: int, D: int, key: jax.random.PRNGKey, 
                          sparsity: float = 0.3) -> jnp.ndarray:
    """Generate non-negative emission matrix with some sparsity."""
    key1, key2 = jax.random.split(key)
    
    # Generate positive entries
    C = jnp.abs(jax.random.normal(key1, (N, D))) * 0.5
    
    # Apply sparsity
    mask = jax.random.bernoulli(key2, 1 - sparsity, (N, D))
    C = C * mask
    
    return C


def generate_synthetic_ssm(D: int, N: int, T: int, 
                           cell_types: jnp.ndarray,
                           cell_sign: jnp.ndarray,
                           cell_type_dimensions: jnp.ndarray,
                           cell_type_mask: jnp.ndarray,
                           key: jax.random.PRNGKey,
                           Q_scale: float = 1e-2,
                           R_scale: float = 1e-3) -> Tuple[ParamsCTDS, jnp.ndarray, jnp.ndarray]:
    """
    Generate complete synthetic SSM with data.
    
    Returns:
        params: True parameters
        latent_states: (T, D) latent trajectory
        observations: (T, N) observations
    """
    keys = jax.random.split(key, 8)
    
    # Create dynamics mask
    dynamics_mask = jnp.repeat(cell_sign, cell_type_dimensions)
    
    # Generate parameters
    #A =jnp.array(create_dynamics_matrix(cell_type_dimensions, D))
    A=make_A_true(key[0], cell_type_dimensions, cell_sign, target_cond=10.0, spectral_radius=0.95)
    Q=jr.normal(keys[1], (D, D))
    Q = Q.T@Q + jnp.identity(D)
    Q=Q/(jnp.max(Q)*1000)
    Q=make_Q_true(key[1], D, target_cond=5.0, scale=6e-3)
    #C = generate_nonnegative_C(N, D, keys[1])
    R = jnp.array(np.diag(np.random.rand( N) + 0.1)/1000)

    # Create emission matrix with Dale's law structure
    keysC = jr.split(keys[2], len(cell_types))
    C_blocks = []
    col_start = 0
    for i in range(len(cell_types)):
        #get the number of neurons for this cell type
        N_type = jnp.sum(jnp.where(cell_type_mask == cell_types[i], 1, 0))
        D_type = cell_type_dimensions[i]
        C_type=jr.uniform(keysC[i], (N_type, D_type), minval=0.0, maxval=1.0)
        #C_type = generate_nonneg_matrix(keysC[i], N_type, D_type, noise=0.05, col_scale=1.0) *cell_sign[i]
        
        # create padded block: [zeros_left, U, zeros_right]
        left_pad = jnp.zeros((N_type, col_start))
        right_pad = jnp.zeros((N_type, D - col_start - D_type))
        C_blocks.append(jnp.concatenate([left_pad, C_type, right_pad], axis=1))
        col_start += D_type

    C = jnp.concatenate(C_blocks, axis=0)
    
    # Initial state
    initial_mean = jnp.zeros(D)
    initial_cov = jnp.eye(D) * 0.1
    
    # Sample trajectory
    latent_states = jnp.zeros((T, D))
    latent_states = latent_states.at[0].set(
        jax.random.multivariate_normal(keys[2], initial_mean, initial_cov)
    )
    
    for t in range(1, T):
        noise = jax.random.multivariate_normal(keys[3 + (t % 4)], jnp.zeros(D), Q)
        latent_states = latent_states.at[t].set(A @ latent_states[t-1] + noise)
    
    # Generate observations
    
    observation_noise = jax.random.multivariate_normal(keys[7], jnp.zeros(N), R, (T))
    print(C.shape, latent_states.shape, observation_noise.shape)
    observations = (C @ latent_states.T).T + observation_noise
   
    ctds=CTDS(
        emission_dim=N,
        cell_types=cell_types,
        cell_sign=cell_sign,
        cell_type_dimensions=cell_type_dimensions,
        cell_type_mask=cell_type_mask,
        region_identity=None,
        inputs_dim=None,
        state_dim= D,
    )
    
    
    # Create parameter object
    constraints = ParamsCTDSConstraints(
        cell_types=cell_types,
        cell_sign=cell_sign,
        cell_type_dimensions=cell_type_dimensions,
        cell_type_mask=cell_type_mask
    )
    bias=jnp.zeros(N)
    
    initial = ParamsCTDSInitial(mean=initial_mean, cov=initial_cov)
    dynamics = ParamsCTDSDynamics(weights=A, cov=Q, dynamics_mask=dynamics_mask)
    emissions = ParamsCTDSEmissions(weights=C, cov=R, bias=bias)
    
    params = ParamsCTDS(
        initial=initial,
        dynamics=dynamics,
        emissions=emissions,
        constraints=constraints,
        observations=observations
    )
    latent_states, observations = ctds.sample(params, key, T)
    #params.observations=observations
    
    
    return ctds, params, latent_states, observations

def generate_full_rank_matrix(key, T, N):
    """
    Generate an n x m matrix with linearly independent columns
    and reasonable conditioning.
    """
    A = jr.uniform(key, (T, N), minval=0.0001, maxval=1.0)
    # Orthonormalize columns with QR
    Q, _ = jnp.linalg.qr(A)
    return Q  
def generate_nonneg_matrix(key, n, p, noise=0.1, col_scale=1.0):
    """
    n x p nonnegative matrix with independent columns, suitable for emissions/design.
    """
    key, kN = jr.split(key)
    A = jnp.zeros((n, p))
    r = min(n, p)
    A = A.at[jnp.arange(r), jnp.arange(r)].set(0.85)  # identity block
    N = jnp.abs(jr.exponential(kN, (n, p))) * noise
    A = (A + N) * col_scale
    return A

def generate_CTDS_Params(
    N: int = 50,  # Total number of neurons
    T: int = 500,  # Number of time points
    D: int = 8,   # Total latent dimensions
    K: int = 2,   # Number of cell types
    excitatory_fraction: float = 0.5,  # Fraction of excitatory neurons
    noise_level: float = 0.1,  # Observation noise level
    dynamics_strength: float = 0.8,  # Dynamics eigenvalue magnitude
    seed: jax.random.PRNGKey = jr.PRNGKey(42)
) -> ParamsCTDS:
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
    
    keys = jr.split(seed, 10)
    
    # Define cell types and properties
    cell_types = jnp.arange(K)
    
    # Assign neurons to cell types
    n_excitatory = int(N * excitatory_fraction)
    n_inhibitory = N - n_excitatory
    
    if K == 2:
        # Standard excitatory/inhibitory setup
        cell_sign = jnp.array([1, -1])  # +1 for excitatory, -1 for inhibitory
        cell_type_dimensions = jnp.array([D//2, D - D//2])  # Split dimensions
        cell_type_mask = jnp.concatenate([
            jnp.zeros(n_excitatory, dtype=int),  # Excitatory = type 0
            jnp.ones(n_inhibitory, dtype=int)    # Inhibitory = type 1
        ])
    else:
        # General multi-type setup
        cell_sign = jnp.concatenate([jnp.ones(K//2), -jnp.ones(K - K//2)]) #shape: (K,)
        dims_per_type = D // K
        cell_type_dimensions = jnp.full(K, dims_per_type) #shape: (K,)
        
        # Distribute remaining dimensions
        remaining = D - dims_per_type * K
        cell_type_dimensions = cell_type_dimensions.at[:remaining].add(1)
        
        # Distribute neurons across all cell types
        neurons_per_type = N // K
        cell_type_mask = jnp.repeat(cell_sign[0], neurons_per_type)
        
        for i in range(1, K):
            if len(cell_type_mask) <= N:
                cell_type_mask = jnp.concatenate([cell_type_mask, jnp.repeat(cell_sign[i], neurons_per_type)])
            
        remaining_neurons = N%K
        if remaining_neurons > 0:
            cell_type_mask = jnp.concatenate([cell_type_mask, jnp.repeat(cell_sign[-1], remaining_neurons)])
    
    # Apply Dale's law constraints to create dynamics mask
    dynamics_list = []
    for i in range(K):
        dynamics_list.append(jnp.full(cell_type_dimensions[i], cell_sign[i]))
    dynamics_mask = jnp.concatenate(dynamics_list)
    
    #A=generate_nonneg_matrix(keys[1], D, D)  # Ensure non-negative dynamics
    A= make_A_true(keys[1], cell_type_dimensions, cell_sign, target_cond=10.0, spectral_radius=0.95) 

    # Create emission matrix with Dale's law structure
    keysC = jr.split(keys[2], len(cell_types))
    C_blocks = []
    col_start = 0
    for i in range(len(cell_types)):
        #get the number of neurons for this cell type
        N_type = jnp.sum(jnp.where(cell_type_mask == cell_types[i], 1, 0))
        D_type = cell_type_dimensions[i]
        C_type=jr.uniform(keysC[i], (N_type, D_type), minval=0.0, maxval=1.0)        
        # create padded block: [zeros_left, U, zeros_right]
        left_pad = jnp.zeros((N_type, col_start))
        right_pad = jnp.zeros((N_type, D - col_start - D_type))
        C_blocks.append(jnp.concatenate([left_pad, C_type, right_pad], axis=1))
        col_start += D_type

    C = jnp.concatenate(C_blocks, axis=0)

    # Process and observation noise
    #Q = jnp.diag(jr.normal(keys[3], (D,)))
    row_norms=jnp.linalg.norm(A, axis=1)
    diag_vals=jnp.maximum(1e-4, row_norms**2 )
    #Q = jnp.diag(diag_vals) * 0.1

    R=jnp.diag(1/jnp.maximum(1e-4, jnp.linalg.norm(C, axis=1))) * 0.1
    #R = jnp.diag(jr.normal(keys[4], (N,)) + 0.1)
    Q=jr.normal(keys[3], (D, D))
    Q = Q.T@Q + jnp.identity(D)
    Q=Q/(jnp.max(Q)*1000)
    Q = make_Q_true(keys[2], D, target_cond=8.0, scale=6e-3)


    emissions=ParamsCTDSEmissions(weights=C, cov=R, bias=jnp.zeros(N))
    dynamics=ParamsCTDSDynamics(weights=A, cov=Q, dynamics_mask=dynamics_mask)
    initial=ParamsCTDSInitial(mean=jnp.zeros(D), cov=jnp.eye(D) * 0.1)
    # Create constraints object
    constraints = ParamsCTDSConstraints(
        cell_types=cell_types,
        cell_sign=cell_sign,
        cell_type_dimensions=cell_type_dimensions,
        cell_type_mask=cell_type_mask
    )
    return ParamsCTDS(
        emissions=emissions,
        dynamics=dynamics,
        initial=initial,
        constraints=constraints,
        observations=None,  # No observations yet
    )

def generate_synthetic_data(
    num_samples: int,
    num_timesteps: int,
    state_dim: int,
    emission_dim: int,
    cell_types: int=2,
    key: jax.random.PRNGKey = jr.PRNGKey(42)
) -> Tuple[Float[Array, "num_samples num_timesteps state_dim"],
           Float[Array, "num_samples num_timesteps emission_dim"],
           CTDS,
           ParamsCTDS]:
    """
    Generates synthetic state and observation data using a CTDS model.
    Args:
        num_samples (int): Number of samples to generate.
        num_timesteps (int): Number of timesteps per sample.
        state_dim (int): Dimensionality of the latent state space.
        emission_dim (int): Dimensionality of the emission/observation space.
        cell_types (int, optional): Number of cell types. Defaults to 2.
        key (jax.random.PRNGKey, optional): JAX random key for reproducibility. Defaults to jr.PRNGKey(42).
    Returns:
        Tuple[
            Float[Array, "num_samples num_timesteps state_dim"],
            Float[Array, "num_samples num_timesteps emission_dim"],
            CTDS,
            ParamsCTDS
        ]: 
            - states: Synthetic latent states.
            - observations: Synthetic observations/emissions.
            - ctds: The CTDS model instance used for generation.
            - ctds_params: The parameters used for the CTDS model.
    """
    
    ctds_params = generate_CTDS_Params(
        N=emission_dim,
        T=num_timesteps,
        D=state_dim,
        K=cell_types,
    )
    ctds=CTDS(
        emission_dim=emission_dim,
        cell_types=ctds_params.constraints.cell_types,
        cell_sign=ctds_params.constraints.cell_sign,
        cell_type_dimensions=ctds_params.constraints.cell_type_dimensions,
        cell_type_mask=ctds_params.constraints.cell_type_mask,
        region_identity=None,
        inputs_dim=None,
        state_dim= state_dim,
    )

    states, observations = ctds.sample(ctds_params, key, num_timesteps)

    return states, observations, ctds, ctds_params


def make_A_true(key, cell_type_dimensions, cell_sign, target_cond=10.0, spectral_radius=0.95):
    """
    Build a Dale-compliant A with controlled condition number.
    
    Column j of A has sign = cell_sign of the cell type that owns dim j:
      - Excitatory columns: off-diagonal entries >= 0
      - Inhibitory columns: off-diagonal entries <= 0
      - Diagonal: unconstrained (controls stability)
    """
    D = int(jnp.sum(cell_type_dimensions))
    col_sign = jnp.repeat(cell_sign, cell_type_dimensions)  # (D,)
    
    k1, k2 = jr.split(key)
    
    # Step 1: Generate non-negative off-diagonal entries
    # Draw from Uniform, then apply column signs for Dale's law
    off_diag = jr.uniform(k1, (D, D), minval=1e-2, maxval=1.0)
    off_diag = off_diag * col_sign[None, :]   # apply sign per column
    off_diag = off_diag.at[jnp.diag_indices(D)].set(0.0)  # clear diagonal
    
    # Step 2: Set diagonal to control eigenvalues
    # Start with negative diagonal (stable) scaled to balance off-diagonal
    row_abs_sum = jnp.sum(jnp.abs(off_diag), axis=1)
    diag_vals = 0.8 * row_abs_sum  # Gershgorin: keeps eigenvalues left of 0.8*row_sum
    A = off_diag + jnp.diag(diag_vals)
    
    # Step 3: Control condition number via SVD projection
    U, s, Vt = jnp.linalg.svd(A)
    # Compress singular values to target range
    s_new = jnp.linspace(s[0], s[0] / target_cond, D)
    A_proj = U @ jnp.diag(s_new) @ Vt
    
    # Step 4: Re-enforce Dale signs (SVD projection may violate them slightly)
    # Off-diagonal: project back to correct sign
    diag_A = jnp.diag(A_proj)
    off_diag_proj = A_proj - jnp.diag(diag_A)
    # E columns (col_sign=1): clamp off-diag to >= 0
    # I columns (col_sign=-1): clamp off-diag to <= 0
    off_diag_proj = jnp.where(col_sign[None, :] == 1,
                               jnp.maximum(off_diag_proj, 0.0),
                               jnp.minimum(off_diag_proj, 0.0))
    A_dale = off_diag_proj + jnp.diag(diag_A)
    
    # Step 5: Scale spectral radius
    sr = jnp.max(jnp.abs(jnp.linalg.eigvals(A_dale)))
    A_dale = A_dale * (spectral_radius / jnp.maximum(sr, 1e-3))
  
    
    return A_dale


def make_Q_true(key, D, target_cond=5.0, scale=1e-3):
    """Q with controlled condition number. Scale sets the magnitude."""
    Q_orth, _ = jnp.linalg.qr(jr.normal(key, (D, D)))
    eig_max = scale
    eig_min = scale / target_cond
    eigs = jnp.linspace(eig_max, eig_min, D)
    Q = Q_orth @ jnp.diag(eigs) @ Q_orth.T
    return (Q + Q.T) / 2  # enforce exact symmetry




def pearsonr_jax(x, y):
    x = x - jnp.mean(x)
    y = y - jnp.mean(y)
    return jnp.sum(x * y) / (jnp.sqrt(jnp.sum(x ** 2)) * jnp.sqrt(jnp.sum(y ** 2)))

def transform_true_rec(C_true, C_rec, A_rec, Q_rec, list_of_dimensions, region_identity=None):
    """JAX version: transform the recovered parameters to match the true parameters, as there are non-identifiabilities."""
    if region_identity is None:
        region_identity = jnp.zeros(C_true.shape[0])

    permuted_indices = jnp.zeros(C_true.shape[1], dtype=int)
    num_cell_type = list_of_dimensions.shape[1]
    num_regions = 1
    for region in range(num_regions):
        d = int(jnp.sum(list_of_dimensions[region]))
        dims_prev_regions = int(jnp.sum(list_of_dimensions[:region])) if region > 0 else 0
        neurons_this_region = jnp.where(region_identity == region)[0]
        C_this_region = C_true[neurons_this_region][:, dims_prev_regions:dims_prev_regions+d]
        C_rec_this_region = C_rec[neurons_this_region][:, dims_prev_regions:dims_prev_regions+d]

        for i in range(num_cell_type):  # cell types
            d_e = int(list_of_dimensions[region, i])
            if d_e == 0:
                continue
            dims_prev_cell_types = int(jnp.sum(list_of_dimensions[region, :i])) if i > 0 else 0
            C_this_type = C_this_region[:, dims_prev_cell_types:dims_prev_cell_types+d_e]
            C_rec_this_type = C_rec_this_region[:, dims_prev_cell_types:dims_prev_cell_types+d_e]
            # Find permutation maximizing correlation
            for j in range(d_e):
                corrs = []
                for k in range(d_e):
                    corr = pearsonr_jax(C_this_type[:, j], C_rec_this_type[:, k])
                    corrs.append(corr)
                best_perm = jnp.argmax(jnp.array(corrs))
                permuted_indices = permuted_indices.at[dims_prev_regions + dims_prev_cell_types + j].set(
                    best_perm + dims_prev_regions + dims_prev_cell_types
                )

    # Permute columns/rows
    C_rec = C_rec[:, permuted_indices]
    A_rec = A_rec[permuted_indices][:, permuted_indices]
    Q_rec = Q_rec[permuted_indices][:, permuted_indices]

    # Scaling using least squares
    total_dims = int(jnp.sum(list_of_dimensions))
    scaling_vec = jnp.zeros(total_dims)
    for i in range(total_dims):
        # Least squares fit: minimize ||a * C_rec[:,i] - C_true[:,i]||^2
        a = jnp.sum(C_rec[:, i] * C_true[:, i]) / jnp.sum(C_rec[:, i] ** 2)
        scaling_vec = scaling_vec.at[i].set(a)

    D_scale = jnp.diag(scaling_vec)
    D_inv = jnp.diag(1.0 / scaling_vec)
    C_rec = (C_rec @ D_scale)
    A_rec = (D_inv @ A_rec @ D_scale)
    Q_rec = (D_inv @ Q_rec @ D_inv)

    return C_rec, A_rec, Q_rec


"""
Simulation Utilis from Aditi Jha
"""
import numpy as np
from scipy.stats.stats import pearsonr  
from sklearn.linear_model import LinearRegression


def generate_low_rank_J(seed, N, N_e, N_i, r, diag = False):
    # let's fix the dynamics matrix, assume some low-d structure and set it's effective dimensionality
    # say J = UV
    # let's ensure U has all positive elements
    U = np.random.rand(N,r)
    assert np.linalg.matrix_rank(U)==r, "U doesn't have the appropriate rank"
    # now for J to have N_e positive columns and N_i negative columns, let's try the following
    V_e = np.random.rand(r, N_e)
    # to ensure that V_e and V_i don't have the same elements
    np.random.seed(seed+1)
    V_i = -np.random.rand(r, N_i)
    V = np.hstack((V_e, V_i))
    assert np.linalg.matrix_rank(V)==r, "V doesn't have the appropriate rank"
    # now get J 
    J = U@V 

    if diag:
        # put zeros on the diagonals, however then J is not low-rank anymore
        np.fill_diagonal(J, 0)
     
    # scale J to ensure all eigen values lie within unit circle
    eig_values, _ = np.linalg.eig(J)
    spectral_radius = np.max(np.abs(eig_values))
    J = J/(spectral_radius+0.5)
    return J


def create_dynamics_matrix(list_of_dimensions, D):
    """ 
    Creates a multi-region dynamics matrix compliant with Dale's law, and only excitatory cross-region connections.
    
    Parameters:
    list_of_dimensions (numpy array): of size num_regions x 2, where the first column is 
                                       the number of excitatory latents for the region 
                                       and the second column is the number of inhibitory latents 
                                       for the region.
    
    Returns:
    numpy.ndarray: The dynamics matrix for the network.
    """
    
    num_regions = 1
    assert num_regions >= 1, "At least 1 region is required"
    
    # Initialize the size of the dynamics matrix
    total_latents = D
    A = np.zeros((total_latents, total_latents))
    
    current_index = 0
    
    # Create A_ii blocks (within-region dynamics)
    for i in range(num_regions):
        excitatory_latents, inhibitory_latents = list_of_dimensions[0],list_of_dimensions[1]
        num_latent_per_region = excitatory_latents + inhibitory_latents

        # Create the within-region dynamics matrix (A_ii)
        A_ii = np.zeros((num_latent_per_region, num_latent_per_region))
        
        # Fill excitatory connections
        A_ii[:, :excitatory_latents] = np.random.rand(num_latent_per_region, excitatory_latents)
        # Fill inhibitory connections
        A_ii[:, excitatory_latents:num_latent_per_region] = -np.random.rand(num_latent_per_region, inhibitory_latents)
        
        # Add positive biases for stabilit
        A_ii = 0.5 * np.identity(num_latent_per_region) + 0.5 * A_ii
        
        # Normalize A_ii and check for NaNs or Infs
        max_eigval = np.max(np.abs(np.linalg.eigvals(A_ii)))
        if max_eigval != 0:
            A_ii /= (max_eigval+0.1)  # Normalize for stability

        # Place A_ii in the correct block location
        A[current_index:current_index + num_latent_per_region,
          current_index:current_index + num_latent_per_region] = A_ii
        
        current_index += num_latent_per_region


    current_index_i = 0
    # Create A_ij blocks (between-region dynamics)
    for i in range(num_regions):
        excitatory_latents_i = list_of_dimensions[0]
        num_latent_per_region_i = np.sum(list_of_dimensions[i])
        current_index_j = 0
        for j in range(num_regions):
            excitatory_latents_j = list_of_dimensions[0]
            num_latent_per_region_j = np.sum(list_of_dimensions[0])
            if i != j:
                # Initialize the between-region dynamics with zeros
                A_ij = np.zeros((num_latent_per_region_i, num_latent_per_region_j))
                
                # Fill only the excitatory to excitatory connections
                A_ij[:, :excitatory_latents_j] = np.random.rand(num_latent_per_region_i, excitatory_latents_j)

                # The connections along the inhibitory dimensions will remain zero, which is already the case
                
                # Normalize to reduce connectivity strength
                max_val = np.max(A_ij)
                if max_val != 0:
                    A_ij /= (10 * max_val)  # Scale down excitatory connections
                
                # Place A_ij in the correct location
                A[current_index_i :current_index_i + num_latent_per_region_i,
                  current_index_j :current_index_j + num_latent_per_region_j] = A_ij
            current_index_j += num_latent_per_region_j
        
        current_index_i += num_latent_per_region_i

    # Normalize the entire dynamics matrix
    max_eigval = np.max(np.abs(np.linalg.eigvals(A)))
    if max_eigval != 0:
        A /= (max_eigval+0.1)  # Normalize to keep it stable

    return A

def transform_true_rec_Numpy(C_true, C_rec, A_rec, Q_rec, list_of_dimensions, region_identity=None):
    """ transform the recovered parameters to match the true parameters, as there are non-identifiabilities """
    # first we might want to permute the E and I latents separately for each region
    # for E and I latents corresponding to each region, we want to find the permutation that 
    # maximizes the correlation between the true and recovered latents, 
    # let's do ths using just the C matrices

    if region_identity is None:
        region_identity = np.zeros(C_true.shape[0])

    permuted_indices = np.zeros(C_true.shape[1])
    num_cell_type = list_of_dimensions.shape[1]
    num_regions = list_of_dimensions.shape[0]
    for region in range(num_regions):
        d = np.sum(list_of_dimensions[region])
        dims_prev_regions = np.sum(list_of_dimensions[:region]) if region>0 else 0
        neurons_this_region = np.where(region_identity == region)[0]
        C_this_region = C_true[neurons_this_region, dims_prev_regions:dims_prev_regions+d]
        C_rec_this_region = C_rec[neurons_this_region, dims_prev_regions:dims_prev_regions+d]

        for i in range(num_cell_type): # cell types
            d_e = list_of_dimensions[region, i]
            if d_e == 0:
                continue
            else:
                dims_prev_cell_types = np.sum(list_of_dimensions[region, :i]) if i>0 else 0
                C_this_type = C_this_region[:, dims_prev_cell_types:dims_prev_cell_types+d_e]
                C_rec_this_type = C_rec_this_region[:, dims_prev_cell_types:dims_prev_cell_types+d_e]
                # now for each column of C_this_type, we want to find the column of C_rec_this_type that is most correlated with it
                for j in range(d_e):
                    corrs = []
                    for k in range(d_e):
                        corr = pearsonr(C_this_type[:, j], C_rec_this_type[:, k])[0]
                        corrs.append(corr)
                    best_perm = np.argmax(corrs)
                    permuted_indices[dims_prev_regions + dims_prev_cell_types + j] = best_perm+dims_prev_regions + dims_prev_cell_types

    # now permute the columns of C_rec
    C_rec = C_rec[:, permuted_indices.astype(int)].copy()
    A_rec = A_rec[permuted_indices.astype(int)][:, permuted_indices.astype(int)].copy()
    Q_rec = Q_rec[permuted_indices.astype(int)][:, permuted_indices.astype(int)].copy()


    # next there might be scaling issues, so lets scale the recovered C matrix to match the true C matrix
    scaling_vec = np.zeros(int(np.sum(list_of_dimensions)))
    for i in range(int(np.sum(list_of_dimensions))):
        reg = LinearRegression().fit(C_rec[:,i].reshape(-1,1), C_true[:,i].reshape(-1,1))
        scaling_vec[i] = reg.coef_[0][0]

    D_scale = np.diag(scaling_vec)
    D_inv = np.linalg.inv(D_scale)
    C_rec = (C_rec@D_scale).copy()
    A_rec = (D_inv@A_rec@D_scale).copy()
    Q_rec = (D_inv@Q_rec@D_inv).copy()
    
    return C_rec, A_rec, Q_rec