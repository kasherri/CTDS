import jax
import jax.numpy as jnp
import jax.random as jr
from typing import Optional, List, Tuple, Dict, Sequence, Any
from jaxtyping import Float, Array
from ctds.params import ParamsCTDSConstraints,ParamsCTDS,ParamsCTDSEmissions,ParamsCTDSDynamics,ParamsCTDSInitial
from ctds.models import CTDS
import numpy as np

def generate_observations_from_states(
    states: jnp.ndarray,
    C: jnp.ndarray,
    R: jnp.ndarray,
    key: jax.random.PRNGKey = jr.PRNGKey(42)
) -> jnp.ndarray:
    """
    Generate observations from states using the emission model: y_t = C @ x_t + v_t
    where v_t ~ N(0, R) is emission noise.
    
    Args:
        states: (D, T) array of latent states over time
        C: (N, D) emission matrix mapping states to observations
        R: (N, N) emission covariance matrix or (N,) diagonal covariance
        key: Random key for noise generation
        
    Returns:
        observations: (N, T) array of observations
    """
    
    D, T = states.shape
    N = C.shape[0]
    
    # Validate dimensions
    #assert C.shape == (N, D), f"C shape {C.shape} incompatible with states shape {states.shape}"
    
    # Handle both full covariance and diagonal covariance
    if R.ndim == 1:
        # Diagonal covariance - R is (N,) vector of variances
        assert R.shape == (N,), f"Diagonal R shape {R.shape} incompatible with N={N}"
        noise_std = jnp.sqrt(R)
        
        # Generate independent noise for each dimension
        keys = jr.split(key, T)
        noise = jnp.array([jr.normal(keys[t], (N,)) * noise_std for t in range(T)]).T
        
    elif R.ndim == 2:
        # Full covariance matrix - R is (N, N)
        assert R.shape == (N, N), f"Full R shape {R.shape} incompatible with N={N}"
        
        # Generate multivariate normal noise
        keys = jr.split(key, T)
        noise = jnp.array([jr.multivariate_normal(keys[t], jnp.zeros(N), R) for t in range(T)]).T
        
    else:
        raise ValueError(f"R must be 1D (diagonal) or 2D (full covariance), got shape {R.shape}")
    
    # Generate observations: y_t = C @ x_t + v_t
    linear_observations = C @ states  # (N, T)
    observations = linear_observations + noise  # (N, T)
    
    return observations.T
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
    """
    # Create stable dynamics matrix
    #A_raw = jr.normal(keys[1], (D, D)) * 0.3
    A_raw = generate_nonneg_matrix(keys[1], D, D)
    eigenvals, eigenvecs = jnp.linalg.eig(A_raw)
    eigenvals_scaled = eigenvals * (dynamics_strength / jnp.max(jnp.abs(eigenvals)))
    A = jnp.real(eigenvecs @ jnp.diag(eigenvals_scaled) @ jnp.linalg.inv(eigenvecs))
    A=jnp.abs(A)  # Ensure non-negative dynamics
    # Apply Dale's law to dynamics matrix
    for i in range(D):
        latent_type = dynamics_mask[i]
        for j in range(D):
            if i != j:  # Skip self-connections
                A = A.at[j,i].set(A[j,i] * latent_type)
    """
    #A=generate_nonneg_matrix(keys[1], D, D)  # Ensure non-negative dynamics
    A=jnp.zeros((D,D))
    A_keys = jr.split(keys[1], D)
    for i in range(D):
        latent_type = dynamics_mask[i]
        key = A_keys[i]
        #fill columns of A
        if latent_type == -1:
            A = A.at[:,i].set(jr.uniform(key, (D,), minval=-1.0, maxval=0.001))
            #setting diagonal to be nonnegative
            A = A.at[i, i].set(jnp.abs(A[i, i]))
        else:
            A = A.at[:,i].set(jr.uniform(key, (D,), minval=0.001, maxval=1.0))

    """
    A=jr.uniform(keys[1], (D, D), minval=0.001, maxval=1.0) 
    # Apply Dale's law to dynamics matrix
    for i in range(D):
        latent_type = dynamics_mask[i]
        for j in range(D):
            if i != j:  # Skip self-connections
                A = A.at[j,i].set(A[j,i] * latent_type)
    """
    #A=0.5 * jnp.eye(D) + 0.5 * A  # Add positive bias for stability
    max_eigval = jnp.max(jnp.abs(jnp.linalg.eigvals(A)))
    if max_eigval != 0:
        A = A / max_eigval  

    # Create emission matrix with Dale's law structure
    keysC = jr.split(keys[2], len(cell_types))
    C_blocks = []
    col_start = 0
    emission_dims = []
    left_padding_dims = []
    right_padding_dims = []
    emiss_start=0
    for i in range(len(cell_types)):
        #get the number of neurons for this cell type
        N_type = jnp.sum(jnp.where(cell_type_mask == cell_types[i], 1, 0))
        D_type = cell_type_dimensions[i]
        C_type=jr.uniform(keysC[i], (N_type, D_type), minval=0.0001, maxval=1.0)
        #C_type = generate_nonneg_matrix(keysC[i], N_type, D_type, noise=0.05, col_scale=1.0) *cell_sign[i]
        
        # create padded block: [zeros_left, U, zeros_right]
        left_pad = jnp.zeros((N_type, col_start))
        right_pad = jnp.zeros((N_type, D - col_start - D_type))
        left_padding_dims.append(left_pad.shape)
        right_padding_dims.append(right_pad.shape)
        
        emission_dims.append((emiss_start, emiss_start + N_type))
        
        emiss_start += N_type
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


    emissions=ParamsCTDSEmissions(weights=C, cov=R, emission_dims=jnp.array(emission_dims), left_padding_dims=jnp.array(left_padding_dims), right_padding_dims=jnp.array(right_padding_dims))
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





"""

def pearsonr_jax(x, y):
    x = x - jnp.mean(x)
    y = y - jnp.mean(y)
    return jnp.sum(x * y) / (jnp.sqrt(jnp.sum(x ** 2)) * jnp.sqrt(jnp.sum(y ** 2)))

def transform_true_rec(C_true, C_rec, A_rec, Q_rec, list_of_dimensions, region_identity=None):
   #JAX version: transform the recovered parameters to match the true parameters, as there are non-identifiabilities.
    if region_identity is None:
        region_identity = jnp.zeros(C_true.shape[0])

    permuted_indices = jnp.zeros(C_true.shape[1], dtype=int)
    num_cell_type = list_of_dimensions.shape[1]
    num_regions = list_of_dimensions.shape[0]
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


# JAX single-region post-hoc alignment: permutation (Hungarian) + diagonal scaling
# Preserves cell-type blocks; transforms (A, C, Q) by the induced similarity map.
#
# Inputs
#   C_true:  (n_neurons, d)  jnp.ndarray
#   C_rec:   (n_neurons, d)  jnp.ndarray
#   A_rec:   (d, d)          jnp.ndarray
#   Q_rec:   (d, d)          jnp.ndarray
#   block_sizes: 1D sequence of ints summing to d, e.g. [d_E, d_PV, d_SOM]
#   ridge: small float for ridge in per-axis scaling
#   use_abs_corr: match by |corr| so signs can be absorbed by scaling
#
# Returns
#   A_aln, C_aln, Q_aln, info
#
# Notes
#   - The Hungarian step uses scipy.optimize.linear_sum_assignment (host side, non-JIT).
#   - Everything else is JAX ndarrays. You can wrap the function outside JIT regions.
#   - If you cannot use SciPy, swap `hungarian_perm_block` with a greedy matcher.



try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

def _corr_matrix(U: jnp.ndarray, V: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """
    Column-wise correlation matrix between U and V.
    U: (n, p), V: (n, q)  ->  Corr: (p, q)
    """
    Uc = U - jnp.mean(U, axis=0, keepdims=True)
    Vc = V - jnp.mean(V, axis=0, keepdims=True)
    num = Uc.T @ Vc                         # (p, q)
    den = jnp.sqrt((jnp.sum(Uc**2, axis=0) + eps)[:, None] *
                   (jnp.sum(Vc**2, axis=0) + eps)[None, :])
    return num / den

def hungarian_perm_block(C_true_blk: jnp.ndarray, C_rec_blk: jnp.ndarray, use_abs_corr: bool = True) -> np.ndarray:
    """
    Build a permutation (as index array) mapping recovered columns to true columns for one block.
    Returns perm indices `perm` such that C_rec_blk[:, perm] aligns to C_true_blk column order.
    """
    p = C_true_blk.shape[1]
    assert C_rec_blk.shape[1] == p, "Block sizes for C_true and C_rec must match."
    # host numpy for Hungarian
    Corr = _corr_matrix(C_true_blk, C_rec_blk)
    Corr = jnp.abs(Corr) if use_abs_corr else Corr
    Corr_np = np.asarray(Corr)
    if not _HAS_SCIPY:
        # Fallback: greedy argmax (not globally optimal). Prefer SciPy whenever possible.
        used = set()
        perm = np.empty(p, dtype=int)
        for i in range(p):
            j = int(np.argmax(Corr_np[i]))
            while j in used:
                Corr_row = Corr_np[i].copy()
                Corr_row[list(used)] = -np.inf
                j = int(np.argmax(Corr_row))
            perm[i] = j
            used.add(j)
        return perm
    # Hungarian on cost = -Corr (maximize Corr)
    row_idx, col_idx = linear_sum_assignment(-Corr_np)
    # row_idx is [0..p-1] in order, so col_idx gives the assigned recovered column per true column
    perm = np.empty(p, dtype=int)
    perm[row_idx] = col_idx
    return perm

def align_single_region_ctds(
    C_true: jnp.ndarray,
    C_rec: jnp.ndarray,
    A_rec: jnp.ndarray,
    Q_rec: jnp.ndarray,
    block_sizes: Sequence[int],
    ridge: float = 1e-6,
    use_abs_corr: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
    """
    Single-region alignment that preserves cell-type blocks:
      1) blockwise permutation via Hungarian assignment on column correlations
      2) per-axis diagonal scaling via ridge-stabilized OLS
      3) apply the induced similarity transform to (A, Q)
    """
    n, d = C_true.shape
    assert C_rec.shape == (n, d)
    assert A_rec.shape == (d, d)
    assert Q_rec.shape == (d, d)
    assert sum(block_sizes) == d

    # 1) Build global permutation across blocks
    start = 0
    perm_global = np.empty(0, dtype=int)
    block_corrs = []
    for bsz in block_sizes:
        sl = slice(start, start + bsz)
        perm_blk = hungarian_perm_block(C_true[:, sl], C_rec[:, sl], use_abs_corr=use_abs_corr)
        # shift to global column indices
        perm_global = np.concatenate([perm_global, perm_blk + start])
        # collect aligned correlations for diagnostics
        Corr_blk = _corr_matrix(C_true[:, sl], C_rec[:, sl][:, perm_blk])
        block_corrs.append(np.asarray(jnp.diag(Corr_blk)))  # correlation of matched pairs
        start += bsz
    # permutation matrix application via indexing
    C_perm = C_rec[:, perm_global]
    A_perm = A_rec[perm_global][:, perm_global]
    Q_perm = Q_rec[perm_global][:, perm_global]

    # 2) Per-axis ridge OLS scaling: s_j = <u, v> / (||u||^2 + ridge)
    u = C_perm  # recovered columns after permutation
    v = C_true
    denom = jnp.sum(u * u, axis=0) + ridge
    numer = jnp.sum(u * v, axis=0)
    s = numer / denom
    D = jnp.diag(s)
    Dinv = jnp.diag(1.0 / (s + 1e-12))

    # 3) Apply similarity transform for scaling
    C_aln = u @ D
    A_aln = Dinv @ A_perm @ D
    Q_aln = Dinv @ Q_perm @ Dinv
    Q_aln = 0.5 * (Q_aln + Q_aln.T)  # symmetrize

    # Diagnostics
    #   per-block matched correlations, columnwise after alignment, and simple condition numbers
    def condnum(mat: jnp.ndarray, eps: float = 1e-12) -> float:
        svals = jnp.linalg.svd(mat, compute_uv=False)
        return float((svals.max() / jnp.maximum(svals.min(), eps)))

    info = {
        "perm_indices": perm_global,                          # numpy array, length d
        "scale_factors": np.asarray(s),                       # per latent
        "matched_corrs_by_block": block_corrs,                # list of np arrays
        "mean_abs_matched_corr": float(jnp.mean(jnp.abs(jnp.concatenate([jnp.array(b) for b in block_corrs])))),
        "cond_C_rec_perm": condnum(C_perm),
        "cond_D": float(np.max(np.abs(s)) / max(np.min(np.abs(s)), 1e-12)),
    }
    return A_aln, C_aln, Q_aln, info
