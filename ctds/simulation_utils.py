import jax
import jax.numpy as jnp
import jax.random as jr
from typing import Optional, List, Tuple
from jaxtyping import Float, Array
from .params import ParamsCTDSConstraints,ParamsCTDS,ParamsCTDSEmissions,ParamsCTDSDynamics,ParamsCTDSInitial
from .models import CTDS
import os
import scipy.linalg

#TODO: Add update bias term in generate synethic data and in tolggsm()

#Plotting Utilis
def save_figure(exp_group_number, fig, name, section):
    """Save figure (PNG only) into Experiment Group #/Section <section>/."""
    folder = os.path.join(f"Exp group {exp_group_number}", f"Section {section}")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{name}.png")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {path}")

# ============================================================================
# GENERATE CTDS PARAMS HELPERS
# ============================================================================

def stationary_latent_cov(A, Q):
    """
    Solve Sigma = A Sigma A.T + Q.
    Requires spectral radius(A) < 1.
    """
    return jnp.array(scipy.linalg.solve_discrete_lyapunov(A, Q))

# ======================Helpers for Observation SNR======================================================
def observation_snr(A, C, Q, R):
    """
    SNR_obs = tr(C Sigma_x C.T) / tr(R),
    where Sigma_x solves Sigma_x = A Sigma_x A.T + Q.
    """
    Sigma_x = stationary_latent_cov(A, Q)
    signal_power = jnp.trace(C @ Sigma_x @ C.T)
    noise_power = jnp.trace(R)
    return signal_power / noise_power


def build_R_targetSNR(A, C, Q, target_snr, R_base=None):
    """
    Make R = r I_N such that observation SNR equals target_snr.
    """
    Sigma_x = stationary_latent_cov(A, Q)
    signal_power = jnp.trace(C @ Sigma_x @ C.T)
    N = C.shape[0]
    if R_base is None:
        r = signal_power / (N * target_snr)
        R = r * jnp.eye(N)
    else: 
        base_noise_power = jnp.trace(R_base)
        r= signal_power / (target_snr * base_noise_power)
        R = r * R_base
    return R
# ======================Helpers for Proccess SNR======================================================
def symmetrize(M):
    return 0.5 * (M + M.T)

def psd_project(M, floor=1e-8):
    """
    Symmetrize and floor eigenvalues to make a valid PSD covariance.
    """
    M = symmetrize(M)
    eigvals, eigvecs = jnp.linalg.eigh(M)
    eigvals = jnp.maximum(eigvals, floor)
    return (eigvecs * eigvals) @ eigvecs.T


def process_snr_ref(A, Q, Sigma_ref=None):
    """
    Process SNR relative to fixed reference covariance:

        tr(A Sigma_ref A.T) / tr(Q)

    If Sigma_ref is None, uses identity.
    """
    D = A.shape[0]
    if Sigma_ref is None:
        Sigma_ref = jnp.eye(D)

    signal_power = jnp.trace(A @ Sigma_ref @ A.T)
    noise_power = jnp.trace(Q)

    return signal_power / noise_power


def build_Q_targetSNR(A, Q_base, target_snr, Sigma_ref=None):
    """
    Scale Q_base so that:

        tr(A Sigma_ref A.T) / tr(Q_scaled) = target_snr.

    This preserves the covariance shape of Q_base.
    """
    D = A.shape[0]
    if Sigma_ref is None:
        Sigma_ref = stationary_latent_cov(A, Q_base)

    signal_power = jnp.trace(A @ Sigma_ref @ A.T)
    base_noise_power = jnp.trace(Q_base)

    scale = signal_power / (target_snr * base_noise_power)
    Q_scaled = scale * Q_base

    return Q_scaled

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


def generate_CTDS_Params(
    N: int = 50,
    T: int = 500,
    D: int = 6,
    K: int = 2,
    excitatory_fraction: float = 0.5,
    # ----------------------------------------------------------------
    # A controls
    # ----------------------------------------------------------------
    # Maximum eigenvalue magnitude of A. Must be < 1 for stability.
    spectral_radius: float = 0.95,
    # Controls the timescale of the dynamics: values close to 1 produce
    # slow, long-memory dynamics; small values produce fast decorrelation.
    # Wired directly into make_A_true(), replacing the hardcoded 0.95.
    target_cond_A: float = 15.0,
    # Condition number of A (ratio of largest to smallest singular value).
    # Passed to make_A_true(). Higher values mean A is more ill-conditioned
    # and some latent directions are much more strongly driven than others.
    # ----------------------------------------------------------------
    # Q controls
    # ----------------------------------------------------------------
    process_snr: float =30.0,
    Q_scale: float = 1e-3,
    # Overall magnitude of the process noise covariance Q.
    # Sets the largest eigenvalue of Q via make_Q_true(scale=Q_scale).
    # Increasing Q_scale raises the stationary latent variance Sigma_inf
    # and therefore the observation SNR (since R is fixed).
    # Decreasing it collapses the latent signal toward zero.
    target_cond_Q: float = 5.0,
    # Condition number of Q. Passed to make_Q_true().
    # ----------------------------------------------------------------
    # R control 
    # ----------------------------------------------------------------
    obs_snr: float = 100.0,
    # If provided, R is rescaled after construction so that the actual
    # observation SNR  ||C Sigma_inf C^T||_F / ||R||_F  matches this
    # target value. Set to None to use the R constructed from noise_level
    
    # If target_obs_snr is set AND fix_Sigma_inf=True, R is the only
    # thing that changes; Sigma_inf is held constant.
    noise_level: float = 0.1,
    # Baseline observation noise scale. Each diagonal entry of R is drawn
    # from Uniform(noise_level/2, noise_level*2). Used only when
    # target_obs_snr is None.
    fix_Sigma_inf: Optional[jnp.ndarray] = None,
    # If provided (a D x D PSD matrix), Q is set to
    #   Q = fix_Sigma_inf - A fix_Sigma_inf A^T
    # so that the stationary covariance is exactly fix_Sigma_inf
    # regardless of spectral_radius. Use this when sweeping spectral_radius
    # while keeping observation SNR constant.
    # fix_Sigma_inf overrides Q_scale and target_cond_Q entirely.
    # ----------------------------------------------------------------
    seed: jax.random.PRNGKey = jr.PRNGKey(42),
    verbose: bool = False,
    config: Optional[dict] = None,
) -> ParamsCTDS:
    """
    Generate a ParamsCTDS with precise control over spectral radius,
    process noise magnitude, and observation SNR.

 
    Returns
    -------
    ParamsCTDS
        Fully constructed parameter object. The `observations` field is
        None; call ctds.sample() separately to generate data.
 
    Diagnostic fields printed (set verbose=False to suppress):
        spectral_radius(A), Sigma_inf eigenvalue range, obs SNR, cond(Q)
    """
    import scipy.linalg as _sla   # local import to avoid top-level dependency
 
    keys = jr.split(seed, 12)
    if fix_Sigma_inf==None:
        fix_Sigma_inf=jnp.eye(D)
 
    # ------------------------------------------------------------------
    # 1. Cell-type partition (unchanged from original)
    # ------------------------------------------------------------------
    cell_types = jnp.arange(K)
    if config is not None:
        #check that config has the right keys
        assert 'cell_type_dimensions' in config, "config must have key 'cell_type_dimensions' (which is a list of length K with the number of latent dimensions for each cell type.)"
        assert 'cell_sign' in config, "config must have key 'cell_sign' (which is a list of length K with the sign of each cell type: +1 for excitatory, -1 for inhibitory.)"
        assert 'cell_type_mask' in config, "config must have key 'cell_type_mask' (which is a length N array mapping each neuron  to a cell type sign.)"
        assert 'cell_type_neuron_count' in config, "config must have key 'cell_type_neuron_count' (which is a list of length K with the number of neurons for each cell type.)"
        cell_sign            = jnp.array(config['cell_sign'])
        cell_type_dimensions = jnp.array(config['cell_type_dimensions'])
        cell_type_mask       = jnp.array(config['cell_type_mask'])
        cell_type_neuron_count = jnp.array(config['cell_type_neuron_count']) 
    else:
        n_excitatory = int(N * excitatory_fraction)
        n_inhibitory = N - n_excitatory
        if K == 2:
            cell_sign            = jnp.array([1, -1])
            cell_type_dimensions = jnp.array([D // 2, D - D // 2])
            cell_type_mask       = jnp.concatenate([
                jnp.zeros(n_excitatory, dtype=int),
                jnp.ones(n_inhibitory,  dtype=int),
            ])
            cell_type_neuron_count = jnp.array([n_excitatory, n_inhibitory])
        else:
            cell_sign      = jnp.concatenate([jnp.ones(K - K // 2), -jnp.ones(K // 2)])
            dims_per_type  = D // K
            cell_type_dimensions = jnp.full(K, dims_per_type)
            remaining      = D - dims_per_type * K
            cell_type_dimensions = cell_type_dimensions.at[:remaining].add(1)
            neurons_per_type     = N // K
            cell_type_neuron_count = jnp.full(K, neurons_per_type)
            cell_type_mask = jnp.repeat(
                jnp.arange(K, dtype=int), neurons_per_type
            )
            remaining_neurons = N % K
            if remaining_neurons > 0:
                cell_type_mask = jnp.concatenate([
                    cell_type_mask,
                    jnp.full(remaining_neurons, K - 1, dtype=int),
                ])
                cell_type_neuron_count = cell_type_neuron_count.at[-1].set(
                    cell_type_neuron_count[-1] + remaining_neurons
                )
 
    dynamics_mask = jnp.concatenate([
        jnp.full(int(cell_type_dimensions[i]), int(cell_sign[i]))
        for i in range(K)
    ])
 
    # ------------------------------------------------------------------
    # 2. A matrix: spectral_radius and target_cond_A are now wired through
    # ------------------------------------------------------------------
    A = make_A_true(
        keys[1],
        cell_type_dimensions,
        cell_sign,
        target_cond    = target_cond_A,
        spectral_radius = spectral_radius,
    )
    # Verify spectral radius was applied correctly (defensive check)
    sr_actual = float(jnp.max(jnp.abs(jnp.linalg.eigvals(A))))
    assert abs(sr_actual - spectral_radius) < 1e-3, (
        f"make_A_true returned rho={sr_actual:.4f}, expected {spectral_radius}"
    )
 
    # ------------------------------------------------------------------
    # 3. C matrix (block-diagonal, non-negative) — unchanged from original
    # ------------------------------------------------------------------
    keysC    = jr.split(keys[2], K)
    C_blocks = []
    col_start = 0
    for i in range(K):
        N_type = int(cell_type_neuron_count[i])
        D_type = int(cell_type_dimensions[i])
        C_type = jr.uniform(keysC[i], (N_type, D_type), minval=0.2, maxval=1.0)
        left_pad  = jnp.zeros((N_type, col_start))
        right_pad = jnp.zeros((N_type, D - col_start - D_type))
        C_blocks.append(jnp.concatenate([left_pad, C_type, right_pad], axis=1))
        col_start += D_type
    C = jnp.concatenate(C_blocks, axis=0)  # (N, D)
 
    # ------------------------------------------------------------------
    # 4. Q matrix
    # ------------------------------------------------------------------
    A_np = jnp.array(A)
    """
    V_target = jnp.array(fix_Sigma_inf)
    Q_base = V_target - A_np @ V_target @ A_np.T
    Q_base = psd_project(Q_base, floor=1e-8)
    """
    Q_base=jax.random.normal(keys[3], (D, D))
    Q_base = Q_base.T@Q_base + jnp.identity(D)
    Q_base=Q_base/(jnp.max(Q_base)*1000)
    Q=build_Q_targetSNR(A, Q_base, process_snr, Sigma_ref=fix_Sigma_inf)
    fix_Sigma_inf = stationary_latent_cov(A, Q)
    # ------------------------------------------------------------------
    # 5. R matrix
    # Base diagonal R from noise_level
    r_diag = jr.uniform(keys[4], (N,),
                        minval=noise_level / 2.0,
                        maxval=noise_level * 2.0)
    R_base = jnp.diag(r_diag)
    R_base = jnp.diag(jnp.array(jnp.diag(jr.uniform(keys[4], (N,N), minval=0.0, maxval=1.0) + 0.1)/1000))
    R=build_R_targetSNR(A,C,Q,obs_snr, R_base)
 
    # ------------------------------------------------------------------
    # 6. Initial distribution
    # Set Sigma_0 = Sigma_inf so the Kalman filter starts in its
    # stationary regime (no transient). This is the Lyapunov
    # initialisation from Section 4.7 of the thesis.
    # ------------------------------------------------------------------
    initial_mean = jnp.zeros(D)
    initial_cov  = fix_Sigma_inf   # stationary covariance, not identity
 
    # ------------------------------------------------------------------
    # 7. Diagnostics
    # ------------------------------------------------------------------
    actual_snr = float(observation_snr(A, C, Q, R))
    eigs_Sigma = jnp.linalg.eigvalsh(fix_Sigma_inf)
    eigs_Q     = jnp.linalg.eigvalsh(Q)
    if verbose==True:
        print(
                f"generate_CTDS_Params diagnostics:\n"
                f"  rho(A)           = {sr_actual:.4f}  (target {spectral_radius})\n"
                f"  cond(A)          = {float(jnp.linalg.cond(A)):.2f}  (target {target_cond_A})\n"
                f"  Q eigenvalues    = [{float(eigs_Q.min()):.2e}, {float(eigs_Q.max()):.2e}]"
                f"  cond(Q) = {float(eigs_Q.max() / eigs_Q.min()):.1f}\n"
                f"  Sigma_inf eigvals= [{float(eigs_Sigma.min()):.2e}, {float(eigs_Sigma.max()):.2e}]\n"
                f"  Obs SNR          = {actual_snr:.3f}"
                + (f"  (target {obs_snr})" if obs_snr is not None else "")
                + ("\n  [Lyapunov-compensated Q: Sigma_inf is fixed]"
                if fix_Sigma_inf is not None else "")
    )
 
    # ------------------------------------------------------------------
    # 8. Assemble ParamsCTDS
    # ------------------------------------------------------------------
    constraints = ParamsCTDSConstraints(
        cell_types           = cell_types,
        cell_sign            = cell_sign,
        cell_type_dimensions = cell_type_dimensions,
        cell_type_mask       = cell_type_mask)
    return ParamsCTDS(
        emissions  = ParamsCTDSEmissions(weights=C, cov=R, bias=jnp.zeros(N)),
        dynamics   = ParamsCTDSDynamics(weights=A, cov=Q, dynamics_mask=dynamics_mask),
        initial    = ParamsCTDSInitial(mean=initial_mean, cov=initial_cov),
        constraints= constraints,
        observations=None)

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







def generate_synthetic_data(
    num_samples: int,
    num_timesteps: int,
    state_dim: int,
    emission_dim: int,
    cell_types: int=2,
    key: jax.random.PRNGKey = jr.PRNGKey(42),
    excitatory_fraction: float = 0.5,  # Fraction of excitatory neurons
    noise_level: float = 0.1,  # Observation noise level
    dynamics_strength: float = 0.8,  # Dynamics eigenvalue magnitude
    process_snr: float = 900.0,  # Process SNR for Q scaling
    obs_snr: float = 10.0,  # Observation SNR for R scaling
    config: Optional[dict] = None,  # Additional config options if needed
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
        excitatory_fraction=excitatory_fraction,
        noise_level=noise_level,
        #dynamics_strength=dynamics_strength,
        process_snr=process_snr,
        obs_snr=obs_snr,
        seed=key,
        config=config)
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
    keys = jr.split(key, num_samples)
    states, observations = jax.vmap(lambda k: ctds.sample(ctds_params, k, num_timesteps))(keys)
    ctds_params=ParamsCTDS(initial=ctds_params.initial, emissions=ctds_params.emissions, dynamics=ctds_params.dynamics, constraints=ctds_params.constraints, observations=observations)

    return states, observations, ctds, ctds_params



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

            #Debug1: dead columns in recovery matrix
            col_norms = np.array([float(jnp.linalg.norm(C_rec_this_type[:, k])) for k in range(d_e)])
            dead = col_norms < 1e-6
            if dead.any():
                print(f"  [align DEBUG] cell_type={i}: {dead.sum()}/{d_e} recovered columns are dead (norm<1e-6): norms={col_norms.round(4)}")
            #Debug 2 fill correlation matrix
            corr_matrix = np.zeros((d_e, d_e))
            for j in range(d_e):
                for k in range(d_e):
                    corr = pearsonr_jax(C_this_type[:, j], C_rec_this_type[:, k])
                    corr_matrix[j, k] = float(corr)
            has_nan = np.isnan(corr_matrix).any()
            if has_nan or True:   # remove "or True" to only print on failure
                print(f"  [align DEBUG] cell_type={i} corr_matrix (true_j x rec_k):\n{corr_matrix.round(3)}")
                if has_nan:
                    print(f"  [align DEBUG] NaN entries at: {list(zip(*np.where(np.isnan(corr_matrix))))}")
            
            
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

    
    print(f"  [align DEBUG] permuted_indices: {np.array(permuted_indices)}")
    # --- DEBUG 3: duplicate assignments (non-bijective) ---
    if len(set(np.array(permuted_indices))) < len(permuted_indices):
        print(f"  [align DEBUG] WARNING: duplicate assignments! permuted_indices={np.array(permuted_indices)}")
    
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
        denom = float(jnp.sum(C_rec[:, i] ** 2))

        # --- DEBUG 4: scaling failures ---
        if denom <= 1e-8 or abs(a) < 1e-6 or abs(a) > 1e4:
            print(f"  [align DEBUG] scaling[{i}]: denom={denom:.2e}, a={a:.4f}  ← suspicious")
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




from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr


def _safe_corr(x, y, eps=1e-12):
    """
    Pearson correlation that returns NaN if either vector is effectively zero.
    """
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)

    nx = np.linalg.norm(x)
    ny = np.linalg.norm(y)
    if nx < eps or ny < eps:
        return np.nan

    c = pearsonr(x, y)[0]
    return c


def transform_true_rec_hungarian(
    C_true,
    C_rec,
    A_rec,
    Q_rec,
    list_of_dimensions,
    region_identity=None,
    dead_thresh=1e-6,
    use_abs_corr=True,
    verbose=False,
):
    """
    Align recovered parameters to true parameters for CTDS using:
      1. blockwise matching by region and cell type
      2. one-to-one Hungarian assignment within each block
      3. per-column scaling after permutation

    Parameters
    ----------
    C_true : np.ndarray, shape (N, D)
        True emission matrix.
    C_rec : np.ndarray, shape (N, D)
        Recovered emission matrix.
    A_rec : np.ndarray, shape (D, D)
        Recovered dynamics matrix.
    Q_rec : np.ndarray, shape (D, D)
        Recovered process covariance.
    list_of_dimensions : np.ndarray, shape (num_regions, num_cell_types) or (num_cell_types,)
        Number of latent dimensions per region x cell type.
        If 1D, interpreted as a single region.
    region_identity : np.ndarray, shape (N,), optional
        Region label for each neuron. If None, assumes one region.
    dead_thresh : float
        Threshold below which a recovered column is considered dead.
    use_abs_corr : bool
        If True, use absolute correlation for matching.
    verbose : bool
        Print diagnostic messages.

    Returns
    -------
    result : dict
        Keys:
            collapsed : bool
            reason : str or None
            permuted_indices : np.ndarray or None
            scaling_vec : np.ndarray or None
            C_aligned : np.ndarray or None
            A_aligned : np.ndarray or None
            Q_aligned : np.ndarray or None
    """
    C_true = np.asarray(C_true, dtype=float)
    C_rec = np.asarray(C_rec, dtype=float)
    A_rec = np.asarray(A_rec, dtype=float)
    Q_rec = np.asarray(Q_rec, dtype=float)

    list_of_dimensions = np.asarray(list_of_dimensions)

    # Allow user to pass shape (num_cell_types,) for one region
    if list_of_dimensions.ndim == 1:
        list_of_dimensions = list_of_dimensions[None, :]

    num_regions, num_cell_types = list_of_dimensions.shape
    N, D = C_true.shape

    if region_identity is None:
        region_identity = np.zeros(N, dtype=int)
    else:
        region_identity = np.asarray(region_identity, dtype=int)

    if C_rec.shape != C_true.shape:
        return {
            "collapsed": True,
            "reason": f"shape mismatch: C_true {C_true.shape}, C_rec {C_rec.shape}",
            "permuted_indices": None,
            "scaling_vec": None,
            "C_aligned": None,
            "A_aligned": None,
            "Q_aligned": None,
        }

    permuted_indices = np.full(D, -1, dtype=int)

    # Build blockwise permutation
    for region in range(num_regions):
        dims_prev_regions = int(np.sum(list_of_dimensions[:region])) if region > 0 else 0
        d_region = int(np.sum(list_of_dimensions[region]))
        neurons_this_region = np.where(region_identity == region)[0]

        C_true_region = C_true[neurons_this_region, dims_prev_regions:dims_prev_regions + d_region]
        C_rec_region = C_rec[neurons_this_region, dims_prev_regions:dims_prev_regions + d_region]

        if verbose:
            print(f"[align] region={region}, neurons={len(neurons_this_region)}, latent_dims={d_region}")

        for cell_type in range(num_cell_types):
            d_block = int(list_of_dimensions[region, cell_type])
            if d_block == 0:
                continue

            dims_prev_cell_types = int(np.sum(list_of_dimensions[region, :cell_type])) if cell_type > 0 else 0
            start = dims_prev_regions + dims_prev_cell_types
            end = start + d_block

            C_true_block = C_true_region[:, dims_prev_cell_types:dims_prev_cell_types + d_block]
            C_rec_block = C_rec_region[:, dims_prev_cell_types:dims_prev_cell_types + d_block]

            # Detect dead recovered columns in this block
            rec_norms = np.linalg.norm(C_rec_block, axis=0)
            dead_mask = rec_norms < dead_thresh
            if np.any(dead_mask):
                reason = (
                    f"dead recovered columns in region={region}, cell_type={cell_type}; "
                    f"norms={np.round(rec_norms, 6)}"
                )
                if verbose:
                    print(f"[align] COLLAPSE: {reason}")
                return {
                    "collapsed": True,
                    "reason": reason,
                    "permuted_indices": None,
                    "scaling_vec": None,
                    "C_aligned": None,
                    "A_aligned": None,
                    "Q_aligned": None,
                }

            # Build correlation matrix
            corr = np.zeros((d_block, d_block), dtype=float)
            for i in range(d_block):
                for j in range(d_block):
                    cij = _safe_corr(C_true_block[:, i], C_rec_block[:, j])
                    corr[i, j] = cij

            if verbose:
                print(f"[align] region={region}, cell_type={cell_type}, corr=\n{np.round(corr, 3)}")

            if np.isnan(corr).any():
                reason = f"NaN correlation matrix in region={region}, cell_type={cell_type}"
                if verbose:
                    print(f"[align] COLLAPSE: {reason}")
                return {
                    "collapsed": True,
                    "reason": reason,
                    "permuted_indices": None,
                    "scaling_vec": None,
                    "C_aligned": None,
                    "A_aligned": None,
                    "Q_aligned": None,
                }

            score = np.abs(corr) if use_abs_corr else corr

            # Hungarian assignment: maximize score => minimize negative score
            row_ind, col_ind = linear_sum_assignment(-score)

            # Put recovered columns in the order of true columns
            local_perm = col_ind[np.argsort(row_ind)]
            permuted_indices[start:end] = np.arange(start, end)[0] * 0 + (start + local_perm)

            if verbose:
                print(
                    f"[align] region={region}, cell_type={cell_type}, "
                    f"assignment true->rec = {list(zip(range(d_block), local_perm))}"
                )

    # Check permutation validity
    if np.any(permuted_indices < 0):
        reason = f"incomplete permutation assignment: {permuted_indices}"
        if verbose:
            print(f"[align] COLLAPSE: {reason}")
        return {
            "collapsed": True,
            "reason": reason,
            "permuted_indices": permuted_indices,
            "scaling_vec": None,
            "C_aligned": None,
            "A_aligned": None,
            "Q_aligned": None,
        }

    if len(np.unique(permuted_indices)) != len(permuted_indices):
        reason = f"non-bijective permutation: {permuted_indices}"
        if verbose:
            print(f"[align] COLLAPSE: {reason}")
        return {
            "collapsed": True,
            "reason": reason,
            "permuted_indices": permuted_indices,
            "scaling_vec": None,
            "C_aligned": None,
            "A_aligned": None,
            "Q_aligned": None,
        }

    # Apply permutation
    C_perm = C_rec[:, permuted_indices].copy()
    A_perm = A_rec[permuted_indices][:, permuted_indices].copy()
    Q_perm = Q_rec[permuted_indices][:, permuted_indices].copy()

    # Per-column scaling
    scaling_vec = np.ones(D, dtype=float)
    for i in range(D):
        denom = np.sum(C_perm[:, i] ** 2)
        if denom < dead_thresh:
            reason = f"dead column after permutation at i={i}"
            if verbose:
                print(f"[align] COLLAPSE: {reason}")
            return {
                "collapsed": True,
                "reason": reason,
                "permuted_indices": permuted_indices,
                "scaling_vec": None,
                "C_aligned": None,
                "A_aligned": None,
                "Q_aligned": None,
            }

        a = np.sum(C_perm[:, i] * C_true[:, i]) / denom
        if not np.isfinite(a):
            reason = f"non-finite scaling at i={i}, denom={denom}, a={a}"
            if verbose:
                print(f"[align] COLLAPSE: {reason}")
            return {
                "collapsed": True,
                "reason": reason,
                "permuted_indices": permuted_indices,
                "scaling_vec": None,
                "C_aligned": None,
                "A_aligned": None,
                "Q_aligned": None,
            }

        scaling_vec[i] = a

    # Similarity transform
    D_scale = np.diag(scaling_vec)
    D_inv = np.diag(1.0 / scaling_vec)

    C_aligned = C_perm @ D_scale
    A_aligned = D_inv @ A_perm @ D_scale
    Q_aligned = D_inv @ Q_perm @ D_inv

    return {
        "collapsed": False,
        "reason": None,
        "permuted_indices": permuted_indices,
        "scaling_vec": scaling_vec,
        "C_aligned": C_aligned,
        "A_aligned": A_aligned,
        "Q_aligned": Q_aligned,
    }






def generate_random_CTDS_params(
    N: int = 50,
    T: int = 500,
    D: int = 6,
    K: int = 2,
    excitatory_fraction: float = 0.5,
    seed: jax.random.PRNGKey = jr.PRNGKey(42),
    verbose: bool = True,
    config: Optional[dict] = None,
    target_cond_A: float = 10.0,
    obs_snr: float = 10.0,
) -> ParamsCTDS:
    """
    Generate a ParamsCTDS with precise control over spectral radius,
    process noise magnitude, and observation SNR.

 
    Returns
    -------
    ParamsCTDS
        Fully constructed parameter object. The `observations` field is
        None; call ctds.sample() separately to generate data.
 
    Diagnostic fields printed (set verbose=False to suppress):
        spectral_radius(A), Sigma_inf eigenvalue range, obs SNR, cond(Q)
    """
 
    keys = jr.split(seed, 12)
  
 
    # ------------------------------------------------------------------
    # 1. Cell-type partition (unchanged from original)
    # ------------------------------------------------------------------
    cell_types = jnp.arange(K)
    if config is not None:
        #check that config has the right keys
        assert 'cell_type_dimensions' in config, "config must have key 'cell_type_dimensions' (which is a list of length K with the number of latent dimensions for each cell type.)"
        assert 'cell_sign' in config, "config must have key 'cell_sign' (which is a list of length K with the sign of each cell type: +1 for excitatory, -1 for inhibitory.)"
        assert 'cell_type_mask' in config, "config must have key 'cell_type_mask' (which is a length N array mapping each neuron  to a cell type sign.)"
        assert 'cell_type_neuron_count' in config, "config must have key 'cell_type_neuron_count' (which is a list of length K with the number of neurons for each cell type.)"
        cell_sign            = jnp.array(config['cell_sign'])
        cell_type_dimensions = jnp.array(config['cell_type_dimensions'])
        cell_type_mask       = jnp.array(config['cell_type_mask'])
        cell_type_neuron_count = jnp.array(config['cell_type_neuron_count']) 
    else:
        n_excitatory = int(N * excitatory_fraction)
        n_inhibitory = N - n_excitatory
        if K == 2:
            cell_sign            = jnp.array([1, -1])
            cell_type_dimensions = jnp.array([D // 2, D - D // 2])
            cell_type_mask       = jnp.concatenate([
                jnp.zeros(n_excitatory, dtype=int),
                jnp.ones(n_inhibitory,  dtype=int),
            ])
            cell_type_neuron_count = jnp.array([n_excitatory, n_inhibitory])
        else:
            cell_sign      = jnp.concatenate([jnp.ones(K - K // 2), -jnp.ones(K // 2)])
            dims_per_type  = D // K
            cell_type_dimensions = jnp.full(K, dims_per_type)
            remaining      = D - dims_per_type * K
            cell_type_dimensions = cell_type_dimensions.at[:remaining].add(1)
            neurons_per_type     = N // K
            cell_type_neuron_count = jnp.full(K, neurons_per_type)
            cell_type_mask = jnp.repeat(
                jnp.arange(K, dtype=int), neurons_per_type
            )
            remaining_neurons = N % K
            if remaining_neurons > 0:
                cell_type_mask = jnp.concatenate([
                    cell_type_mask,
                    jnp.full(remaining_neurons, K - 1, dtype=int),
                ])
                cell_type_neuron_count = cell_type_neuron_count.at[-1].set(
                    cell_type_neuron_count[-1] + remaining_neurons
                )
 
    dynamics_mask = jnp.concatenate([
        jnp.full(int(cell_type_dimensions[i]), int(cell_sign[i]))
        for i in range(K)
    ])
 
    # ------------------------------------------------------------------
    # 2. A matrix: spectral_radius and target_cond_A are now wired through
    # ------------------------------------------------------------------
    #A_np = create_dynamics_matrix(np.array(cell_type_dimensions), D)
    #A=jnp.array(A_np)
    A = make_A_true(
        keys[1],
        cell_type_dimensions,
        cell_sign,
        target_cond    = target_cond_A,
        spectral_radius = 0.95, ) # slightly less than 1 for stability)
    # Verify spectral radius was applied correctly (defensive check)
    sr_actual = float(jnp.max(jnp.abs(jnp.linalg.eigvals(A))))
 
    # ------------------------------------------------------------------
    # 3. C matrix (block-diagonal, non-negative) — unchanged from original
    # ------------------------------------------------------------------
    keysC    = jr.split(keys[2], K)
    C_blocks = []
    col_start = 0
    for i in range(K):
        N_type = int(cell_type_neuron_count[i])
        D_type = int(cell_type_dimensions[i])
        C_type = jr.uniform(keysC[i], (N_type, D_type), minval=0.2, maxval=1.0)
        left_pad  = jnp.zeros((N_type, col_start))
        right_pad = jnp.zeros((N_type, D - col_start - D_type))
        C_blocks.append(jnp.concatenate([left_pad, C_type, right_pad], axis=1))
        col_start += D_type
    C = jnp.concatenate(C_blocks, axis=0)  # (N, D)
 
    # ------------------------------------------------------------------
    # 4. Q matrix
    # ------------------------------------------------------------------
    A_np = jnp.array(A)
    """
    V_target = jnp.array(fix_Sigma_inf)
    Q_base = V_target - A_np @ V_target @ A_np.T
    Q_base = psd_project(Q_base, floor=1e-8)
    """
    Q=jax.random.normal(keys[3], (D, D))
    # ------------------------------------------------------------------
    # 5. R matrix
    R = jnp.diag(jnp.array(jnp.diag(jr.uniform(keys[4], (N,N), minval=0.0, maxval=1.0))))
 
    # ------------------------------------------------------------------
    # 6. Initial distribution
    # Set Sigma_0 = Sigma_inf so the Kalman filter starts in its
    # stationary regime (no transient). This is the Lyapunov
    # initialisation from Section 4.7 of the thesis.
    # ------------------------------------------------------------------
    initial_mean = jnp.zeros(D)
    initial_cov  = jax.random.normal(keys[5], (D, D))
 
    # ------------------------------------------------------------------
    # 7. Diagnostics
    # ------------------------------------------------------------------
    actual_snr = float(observation_snr(A, C, Q, R))
    fix_Sigma_inf = scipy.linalg.solve_discrete_lyapunov(A, Q)
    eigs_Sigma = jnp.linalg.eigvalsh(fix_Sigma_inf)
    eigs_Q     = jnp.linalg.eigvalsh(Q)
    spectral_radius = float(jnp.max(jnp.abs(jnp.linalg.eigvals(A))))
    if verbose==True:
        print(
                f"generate_CTDS_Params diagnostics:\n"
                f"  rho(A)           = {sr_actual:.4f}  (target {spectral_radius})\n"
                f"  cond(A)          = {float(jnp.linalg.cond(A)):.2f}  (target {target_cond_A})\n"
                f"  Q eigenvalues    = [{float(eigs_Q.min()):.2e}, {float(eigs_Q.max()):.2e}]"
                f"  cond(Q) = {float(eigs_Q.max() / eigs_Q.min()):.1f}\n"
                f"  Sigma_inf eigvals= [{float(eigs_Sigma.min()):.2e}, {float(eigs_Sigma.max()):.2e}]\n"
                f"  Obs SNR          = {actual_snr:.3f}"
                + (f"  (target {obs_snr})" if obs_snr is not None else "")
                + ("\n  [Lyapunov-compensated Q: Sigma_inf is fixed]"
                if fix_Sigma_inf is not None else "")
    )
 
    # ------------------------------------------------------------------
    # 8. Assemble ParamsCTDS
    # ------------------------------------------------------------------
    constraints = ParamsCTDSConstraints(
        cell_types           = cell_types,
        cell_sign            = cell_sign,
        cell_type_dimensions = cell_type_dimensions,
        cell_type_mask       = cell_type_mask)
    return ParamsCTDS(
        emissions  = ParamsCTDSEmissions(weights=C, cov=R, bias=jnp.zeros(N)),
        dynamics   = ParamsCTDSDynamics(weights=A, cov=Q, dynamics_mask=dynamics_mask),
        initial    = ParamsCTDSInitial(mean=initial_mean, cov=initial_cov),
        constraints= constraints,
        observations=None)