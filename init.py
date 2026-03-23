"""
init.py — FA-based and PCA-based initialization pipelines for CTDS.

Produces valid CTDS parameter dicts:
    X0  : (B, T, D)       latent trajectories
    C0  : (N, D)          emission matrix  [block-diag, nonneg per type]
    d0  : (N,)            observation bias
    R0  : (N,)            diagonal obs noise
    A0  : (D, D)          dynamics matrix  [CTDS sign constraints]
    Q0  : (D, D)          process noise

Constraints enforced here match models.py / params.py conventions:
  - Emission C is block-diagonal: neuron of type k only connects to
    latent dims [sum(D_:k), sum(D_:k+1)).  Within that block, C >= 0.
  - Dynamics A: off-diagonal column j entries have sign == dynamics_mask[j].
    Diagonal entries are unconstrained.
  - Q is PSD (symmetric, eigenvalues >= eps).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
from functools import partial
from typing import NamedTuple, Tuple, Dict, Optional

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Block-diagonal emission bounds (mirrors models.py create_emission_bounds)
# ---------------------------------------------------------------------------

def _build_emission_mask(
    cell_type_mask: jnp.ndarray,        # (N,)  cell-type index per neuron
    cell_types: jnp.ndarray,            # (K,)  ordered cell-type indices 0..K-1
    cell_type_dimensions: jnp.ndarray,  # (K,)  latent dims per type
) -> jnp.ndarray:
    """
    Return boolean (N, D) mask: True where neuron n is allowed to connect
    to latent dim d (i.e., same cell type block).

    Args:
        cell_type_mask      : (N,)
        cell_types          : (K,)
        cell_type_dimensions: (K,)

    Returns: (N, D) bool array.
    """
    D = int(jnp.sum(cell_type_dimensions))
    N = cell_type_mask.shape[0]
    K = cell_types.shape[0]

    # latent_to_type[d] = cell type owning latent dim d
    latent_to_type = jnp.repeat(cell_types, cell_type_dimensions)  # (D,)

    neuron_types = cell_type_mask[:, None]   # (N, 1)
    latent_types = latent_to_type[None, :]   # (1, D)
    return neuron_types == latent_types       # (N, D)


def _nnls_row_pgd(
    x_init: jnp.ndarray,   # (D,)  initial solution
    XtX: jnp.ndarray,      # (D, D)  X.T @ X  (precomputed)
    XtY: jnp.ndarray,      # (D,)   X.T @ y_n  (for neuron n)
    mask: jnp.ndarray,     # (D,)  bool — which entries may be nonzero
    n_iter: int = 300,
    lr: float = 1e-2,
) -> jnp.ndarray:
    """
    Solve one row of  min_{c>=0, block-zero} ||y - X c||^2  via projected GD.

    c must be nonneg where mask==True, and ==0 elsewhere (block-diagonal).

    grad_c = 2 XtX c - 2 XtY
    project = clamp to [0, inf] then zero out forbidden dims.

    Args:
        x_init : (D,)
        XtX    : (D, D)
        XtY    : (D,)  — note sign: gradient = XtX c - XtY  ∝ - X.T(y - Xc)
        mask   : (D,) bool
        n_iter : gradient descent steps
        lr     : learning rate

    Returns: (D,) solution for one neuron.
    """
    # Use adaptive step: 1 / max_eig(XtX) from power-method estimate
    # (but we pass lr as fallback for speed)
    def _step(c, _):
        grad = XtX @ c - XtY          # (D,)
        c_new = c - lr * grad          # gradient step
        c_new = jnp.where(mask, jnp.maximum(c_new, 0.0), 0.0)  # project
        return c_new, None

    c_final, _ = jax.lax.scan(_step, x_init, None, length=n_iter)
    return c_final


def _fit_C_constrained(
    X_flat: jnp.ndarray,              # (BT, D)  latent trajectories flattened
    Y_flat: jnp.ndarray,              # (BT, N)  centered observations flattened
    cell_type_mask: jnp.ndarray,      # (N,)
    cell_types: jnp.ndarray,          # (K,)
    cell_type_dimensions: jnp.ndarray,# (K,)
    n_iter: int = 400,
    ridge: float = 1e-4,
) -> jnp.ndarray:
    """
    Solve for emission matrix C: (N, D) with block-diagonal nonneg constraint.

    Problem (per neuron n):
        min_{c_n >= 0, forbidden=0} ||Y[:, n] - X c_n||^2

    Solved via vmapped projected gradient descent.

    Args:
        X_flat              : (BT, D)
        Y_flat              : (BT, N)
        cell_type_mask      : (N,)
        cell_types          : (K,)
        cell_type_dimensions: (K,)
        n_iter              : PGD iterations
        ridge               : L2 regularisation on XtX

    Returns: C (N, D).
    """
    D = X_flat.shape[1]
    N = Y_flat.shape[1]

    XtX = X_flat.T @ X_flat + ridge * jnp.eye(D)  # (D, D)
    XtY = X_flat.T @ Y_flat                        # (D, N)

    # Block-diagonal mask
    block_mask = _build_emission_mask(
        cell_type_mask, cell_types, cell_type_dimensions
    )  # (N, D)  bool

    # Adaptive step size: 1 / largest eigenvalue of XtX
    max_eig = jnp.linalg.eigvalsh(XtX)[-1]
    lr = 0.9 / (max_eig + 1e-8)

    # Initialise: least-squares then project
    C_ls = jnp.linalg.solve(XtX, XtY).T       # (N, D)
    C_init = jnp.where(block_mask, jnp.maximum(C_ls, 0.0), 0.0)

    # vmap over neurons (axis 0 of XtY and block_mask)
    _solve_one = partial(_nnls_row_pgd, n_iter=n_iter, lr=lr)
    C = jax.vmap(_solve_one, in_axes=(0, None,1,0))(
        C_init,XtX, XtY, block_mask
    )  # (N, D)
    return C


# ---------------------------------------------------------------------------
# Dynamics: A and Q from latent trajectories
# ---------------------------------------------------------------------------

def _fit_A_dale(
    X0_flat: jnp.ndarray,             # (BT, D)  all time steps (incl. t=T)
    dynamics_mask: jnp.ndarray,       # (D,)
    ridge: float = 1e-5,
    n_iter_pgd: int = 500,
) -> jnp.ndarray:
    """
    Fit dynamics A: (D, D) from X_{t+1} ~ A X_t.

    Least squares ignoring constraints → project onto Dale cone.
    Projection is exact for independent columns, and cheap.

    Steps:
      1. Stack  X_prev = X[:-1, :]  (BT-B, D)
                X_next = X[1:,  :]  (BT-B, D)
         across batches (each batch contributes T-1 pairs).
      2. LS:  A_ls = (X_next.T @ X_prev) (X_prev.T @ X_prev + ridge I)^{-1}
      3. Project A_ls to Dale cone.
      4. Scale spectral radius to < 1.

    Args:
        X0_flat : (BT, D) — already time-ordered within each batch block
        dynamics_mask: (D,)
        ridge   : ridge for XtX
        n_iter_pgd : unused (projection is exact here; kept for API compat)

    Returns: (D, D)
    """
    X_prev = X0_flat[:-1, :]          # (BT-B, D)  — ignores last step per batch
    X_next = X0_flat[1:, :]           # (BT-B, D)

    XtX = X_prev.T @ X_prev + ridge * jnp.eye(X_prev.shape[1])   # (D, D)
    XtY = X_prev.T @ X_next           # (D, D)   —  X_prev.T X_next
    # A.T = XtX^{-1} XtY  →  A = XtY.T XtX^{-T}
    A_ls = jnp.linalg.solve(XtX, XtY).T    # (D, D)

    # Project to Dale
    A = _project_A_dale(A_ls, dynamics_mask)

    # Rescale spectral radius to 0.95
    eigvals = jnp.linalg.eigvals(A)
    sr = jnp.max(jnp.abs(eigvals))
    A = jnp.where(sr > 1e-8, A * (0.95 / (sr + 1e-8)), A)
    return A


def _fit_Q(
    X0_flat: jnp.ndarray,             # (BT, D)
    A: jnp.ndarray,                   # (D, D)
    ridge: float = 1e-5,
) -> jnp.ndarray:
    """
    Compute process-noise covariance Q from residuals of  X_next - A X_prev.

    Q = (1/(T-1)) sum_t (x_{t+1} - A x_t)(x_{t+1} - A x_t).T + ridge I

    Returns: (D, D) PSD.
    """
    X_prev = X0_flat[:-1, :]           # (BT-B, D)
    X_next = X0_flat[1:, :]           # (BT-B, D)
    resid = X_next - (A @ X_prev.T).T  # (BT-B, D)
    Q_raw = (resid.T @ resid) / resid.shape[0] + ridge * jnp.eye(A.shape[0])
    return _psd_project(Q_raw, floor=ridge)


# ---------------------------------------------------------------------------
# Latent normalization
# ---------------------------------------------------------------------------

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


# ===========================================================================
# PART 1 — Factor Analysis (FA) initialization
# ===========================================================================

def _fa_em_single_type(
    Y_type: jnp.ndarray,   # (BT, N_type)  data for one cell type
    D_type: int,           # number of factors
    key: jax.random.PRNGKey,
    n_iter: int = 100,
    eps: float = 1e-6,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    EM for Factor Analysis on a single cell-type block.

    Model:  y = Lambda z + e,  z ~ N(0, I_D),  e ~ N(0, diag(Psi))

    E-step (per observation y_n):
        beta = Lambda.T (Lambda Lambda.T + Psi)^{-1}       (D, N)
        Ez   = beta Y.T                                     (D, BT)
        Ezz  = I - beta Lambda + Ez Ez.T / BT              (D, D)  [avg]

    M-step:
        Lambda_new = (sum Ez y.T) / (BT Ezz)               (N, D) → transpose
        Psi_new    = diag( yy.T/BT - Lambda_new Ezz Lambda_new.T )

    Args:
        Y_type : (BT, N_type)
        D_type : int
        key    : PRNG key
        n_iter : EM iterations
        eps    : diagonal floor for Psi

    Returns:
        Lambda : (N_type, D_type)  factor loadings (nonneg)
        Psi    : (N_type,)         diagonal noise
        Ez     : (BT, D_type)     posterior means of latent factors
    """
    BT, N_type = Y_type.shape

    # Initialize Lambda with small positive random values, Psi = var(Y) / D
    Lambda = jnp.abs(jr.normal(key, (N_type, D_type))) * 0.1   # (N_type, D_type)
    Psi = jnp.var(Y_type, axis=0) / D_type + eps               # (N_type,)

    # Precompute YY.T / BT for M-step reuse
    YtY_mean = (Y_type.T @ Y_type) / BT   # (N_type, N_type)

    def em_step(carry, _):
        Lambda, Psi = carry

        # ---------- E-step ----------
        Psi_inv = 1.0 / (Psi + eps)                               # (N_type,)
        # beta = Lambda.T diag(Psi)^{-1} (I + Lambda.T Psi^{-1} Lambda)^{-1}
        # via Woodbury: (Lambda Lambda.T + Psi)^{-1} = Psi^{-1} - Psi^{-1} Lambda M Lambda.T Psi^{-1}
        # where M = (I + Lambda.T Psi^{-1} Lambda)^{-1}  (D x D)
        LtPsiInv = Lambda.T * Psi_inv[None, :]          # (D_type, N_type)  Lambda.T diag(Psi^-1)
        M_mat = jnp.eye(D_type) + LtPsiInv @ Lambda     # (D_type, D_type)
        M_inv = jnp.linalg.inv(M_mat + eps * jnp.eye(D_type))   # (D_type, D_type)
        beta = M_inv @ LtPsiInv                          # (D_type, N_type)

        Ez = Y_type @ beta.T                             # (BT, D_type)  posterior means
        # E[zz.T] = (I - beta Lambda) + Ez.T Ez / BT
        Ezz_avg = M_inv + (Ez.T @ Ez) / BT              # (D_type, D_type)

        # ---------- M-step ----------
        # Lambda = (Y.T Ez) Ezz^{-1}
        YtEz = Y_type.T @ Ez / BT                       # (N_type, D_type)
        Ezz_inv = jnp.linalg.inv(Ezz_avg + eps * jnp.eye(D_type))
        Lambda_new = YtEz @ Ezz_inv                     # (N_type, D_type)
        #Lambda_new = jnp.maximum(Lambda_new, 0.0)       # enforce nonnegativity

        # Psi = diag(YY/BT - Lambda_new Ezz Lambda_new.T)
        Psi_new = jnp.diag(YtY_mean - Lambda_new @ Ezz_avg @ Lambda_new.T)
        Psi_new = jnp.maximum(Psi_new, eps)

        return (Lambda_new, Psi_new), None

    (Lambda, Psi), _ = jax.lax.scan(em_step, (Lambda, Psi), None, length=n_iter)

    # Final E-step to get Ez
    Psi_inv = 1.0 / (Psi + eps)
    LtPsiInv = Lambda.T * Psi_inv[None, :]
    M_mat = jnp.eye(D_type) + LtPsiInv @ Lambda
    M_inv = jnp.linalg.inv(M_mat + eps * jnp.eye(D_type))
    beta = M_inv @ LtPsiInv
    Ez = Y_type @ beta.T       # (BT, D_type)

    return Lambda, Psi, Ez


def fa_initialize_ctds(
    Y: jnp.ndarray,                     # (B, T, N)
    e_mask: jnp.ndarray,                # (N,) bool  excitatory neurons
    i_mask: jnp.ndarray,                # (N,) bool  inhibitory neurons
    D: int,
    cell_types: jnp.ndarray,            # (K,)  e.g. jnp.array([0, 1])
    cell_sign: jnp.ndarray,             # (K,)  e.g. jnp.array([1, -1])
    cell_type_dimensions: jnp.ndarray,  # (K,)  e.g. jnp.array([D_e, D_i])
    cell_type_mask: jnp.ndarray,        # (N,)  cell-type label per neuron
    key: jax.random.PRNGKey = jr.PRNGKey(0),
    fa_iters: int = 100,
    pgd_iters: int = 400,
    pgd_ridge: float = 1e-4,
) -> Dict[str, jnp.ndarray]:
    """
    FA-based CTDS initialization.

    Args:
        Y                   : (B, T, N)
        e_mask              : (N,) bool
        i_mask              : (N,) bool
        D_e                 : int  excitatory latent dims
        D_i                 : int  inhibitory latent dims
        cell_types          : (K,)
        cell_sign           : (K,)   +1 or -1
        cell_type_dimensions: (K,)
        cell_type_mask      : (N,)   0-indexed cell-type per neuron
        key                 : PRNG key
        fa_iters            : EM iterations for FA
        pgd_iters           : PGD iterations for constrained C regression
        pgd_ridge           : ridge for XtX in C regression

    Returns dict with keys:
        X0 (B,T,D), C0 (N,D), d0 (N,), R0 (N,), A0 (D,D), Q0 (D,D)
    """
    B, T, N = Y.shape
    k1, k2, k3 = jr.split(key, 3)

    # ---- Step 1: Preprocess ----
    Y_flat = Y.reshape(B * T, N)          # (BT, N)
    d0 = jnp.mean(Y_flat, axis=0)         # (N,)
    Y_centered = Y_flat - d0[None, :]     # (BT, N)

    # ---- Step 2: FA per cell type ----
    Y_e = Y_centered[:, e_mask]           # (BT, N_e)
    Y_i = Y_centered[:, i_mask]           # (BT, N_i)

    Lambda_e, psi_e, Ez_e = _fa_em_single_type(Y_e, D_e, k1, n_iter=fa_iters)
    # (N_e, D_e), (N_e,), (BT, D_e)

    Lambda_i, psi_i, Ez_i = _fa_em_single_type(Y_i, D_i, k2, n_iter=fa_iters)
    # (N_i, D_i), (N_i,), (BT, D_i)

    # ---- Step 3: Concatenate latent trajectories ----
    X_flat = jnp.concatenate([Ez_e, Ez_i], axis=1)   # (BT, D)
    X0 = X_flat.reshape(B, T, D)                      # (B, T, D)

    # ---- Step 4: R0 from FA noise ----
    # Reconstruct per-neuron R ordering: e neurons first, then i neurons
    # We need to reorder psi back to original neuron order
    R_arr = jnp.zeros(N)
    R_arr = R_arr.at[e_mask].set(psi_e)
    R_arr = R_arr.at[i_mask].set(psi_i)
    R0 = R_arr                                         # (N,)

    # ---- Step 5: Constrained C regression ----
    # Use X_flat from FA posterior means
    C0 = _fit_C_constrained(
        X_flat, Y_centered,
        cell_type_mask, cell_types, cell_type_dimensions,
        n_iter=pgd_iters, ridge=pgd_ridge,
    )  # (N, D)

    # ---- Step 6: Normalize latent space ----
    #X_flat_norm, C0 = _normalize_latents(X_flat, C0)   # both updated
    #X0 = X_flat_norm.reshape(B, T, D)

    # ---- Step 7: Fit A with Dale constraints ----
    # Build dynamics_mask: repeat cell_sign for each dim
    dynamics_mask = jnp.repeat(cell_sign, cell_type_dimensions)   # (D,)
    # Flatten batches for regression (each batch contributes T-1 pairs)
    # We treat each batch independently to avoid cross-batch transitions
    # → collect (B*(T-1), D) pairs
    X_prev_list = X0[:, :-1, :].reshape(B * (T - 1), D)   # (B(T-1), D)
    X_next_list = X0[:, 1:, :].reshape(B * (T - 1), D)    # (B(T-1), D)
    X_all = jnp.concatenate([X_prev_list, X_next_list[-1:]], axis=0)  # crude; use helper
    A0 = _fit_A_dale(
        jnp.concatenate([X_prev_list, X_next_list], axis=0),  # not used directly
        dynamics_mask,
    )
    # Re-fit properly using pairs
    A0 = _fit_A_from_pairs(X_prev_list, X_next_list, dynamics_mask)

    # ---- Step 8: Fit Q ----
    Q0 = _fit_Q_from_pairs(X_prev_list, X_next_list, A0)

    return dict(X0=X0, C0=C0, d0=d0, R0=R0, A0=A0, Q0=Q0)


# ---------------------------------------------------------------------------
# Helper: fit A and Q from explicit (X_prev, X_next) pairs
# ---------------------------------------------------------------------------

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


# ===========================================================================
# PART 2 — PCA-based initialization
# ===========================================================================

def pca_initialize_ctds(
    Y: jnp.ndarray,                     # (B, T, N)
    e_mask: jnp.ndarray,                # (N,) bool
    i_mask: jnp.ndarray,                # (N,) bool
    D_e: int,
    D_i: int,
    cell_types: jnp.ndarray,            # (K,)
    cell_sign: jnp.ndarray,             # (K,)
    cell_type_dimensions: jnp.ndarray,  # (K,)
    cell_type_mask: jnp.ndarray,        # (N,)
    key: jax.random.PRNGKey = jr.PRNGKey(0),
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
    D = D_e + D_i

    # ---- Step 1: Preprocess ----
    Y_flat = Y.reshape(B * T, N)          # (BT, N)
    d0 = jnp.mean(Y_flat, axis=0)         # (N,)
    Y_centered = Y_flat - d0[None, :]     # (BT, N)

    # ---- Step 2: PCA via truncated SVD on full Y_centered ----
    # Y_centered: (BT, N);  U: (BT, min), S: (min,), Vt: (min, N)
    U, S, Vt = jnp.linalg.svd(Y_centered, full_matrices=False)
    # Take top D principal components
    V_top = Vt[:D, :].T      # (N, D)  — columns are PC directions in obs space
    X_flat = Y_centered @ V_top   # (BT, D)  — latent trajectories

    X0 = X_flat.reshape(B, T, D)  # (B, T, D)

    # ---- Step 4: Constrained C regression (same as FA) ----
    C0 = _fit_C_constrained(
        X_flat, Y_centered,
        cell_type_mask, cell_types, cell_type_dimensions,
        n_iter=pgd_iters, ridge=pgd_ridge,
    )  # (N, D)

    
    # ---- Normalize latents ----
    X_flat_norm, C0 = _normalize_latents(X_flat, C0)
    X0 = X_flat_norm.reshape(B, T, D)

    # ---- Step 5: R from residual variance ----
    Y_pred = (C0 @ X_flat_norm.T).T      # (BT, N)
    resid_Y = Y_centered - Y_pred        # (BT, N)
    R0 = jnp.mean(resid_Y ** 2, axis=0) + 1e-6    # (N,)

    # ---- Step 6: A and Q ----
    dynamics_mask = jnp.repeat(cell_sign, cell_type_dimensions)   # (D,)
    X_prev_list = X0[:, :-1, :].reshape(B * (T - 1), D)
    X_next_list = X0[:, 1:, :].reshape(B * (T - 1), D)

    A0 = _fit_A_from_pairs(X_prev_list, X_next_list, dynamics_mask)
    Q0 = _fit_Q_from_pairs(X_prev_list, X_next_list, A0)

    return dict(X0=X0, C0=C0, d0=d0, R0=R0, A0=A0, Q0=Q0)


# ===========================================================================
# PART 3 — Diagnostics
# ===========================================================================

def diagnostics(
    Y: jnp.ndarray,          # (B, T, N)
    init: Dict[str, jnp.ndarray],
) -> Dict[str, jnp.ndarray]:
    """
    Compute diagnostic metrics for a CTDS initialization.

    1. Observation reconstruction:
       - total R²
       - per-neuron R²  (N,)

    2. Dynamics quality:
       - one-step prediction R²
       - spectral radius of A0
       - trace of Q0

    3. Latent conditioning:
       - singular values of X_flat   (D,)
       - condition number of X_flat

    Args:
        Y    : (B, T, N) raw observations (uncentered)
        init : dict from fa_initialize_ctds or pca_initialize_ctds

    Returns dict of scalar / array diagnostics.
    """
    X0 = init["X0"]      # (B, T, D)
    C0 = init["C0"]      # (N, D)
    d0 = init["d0"]      # (N,)
    A0 = init["A0"]      # (D, D)
    Q0 = init["Q0"]      # (D, D)

    B, T, N = Y.shape
    D = X0.shape[2]

    # Flatten
    Y_flat = Y.reshape(B * T, N)           # (BT, N)
    X_flat = X0.reshape(B * T, D)          # (BT, D)

    # ---- 1. Observation reconstruction ----
    Y_hat = (C0 @ X_flat.T).T + d0[None, :]   # (BT, N)
    resid = Y_flat - Y_hat                      # (BT, N)

    ss_res_total = jnp.sum(resid ** 2)
    ss_tot_total = jnp.sum((Y_flat - jnp.mean(Y_flat)) ** 2)
    r2_total = 1.0 - ss_res_total / (ss_tot_total + 1e-12)

    # Per-neuron R²
    ss_res_n = jnp.sum(resid ** 2, axis=0)                         # (N,)
    ss_tot_n = jnp.sum((Y_flat - jnp.mean(Y_flat, axis=0)) ** 2, axis=0)  # (N,)
    r2_per_neuron = 1.0 - ss_res_n / (ss_tot_n + 1e-12)           # (N,)

    # ---- 2. Dynamics quality ----
    X_prev = X0[:, :-1, :].reshape(B * (T - 1), D)
    X_next = X0[:, 1:, :].reshape(B * (T - 1), D)
    X_pred = (A0 @ X_prev.T).T                                     # (B(T-1), D)

    dyn_resid = X_next - X_pred
    ss_dyn_res = jnp.sum(dyn_resid ** 2)
    ss_dyn_tot = jnp.sum((X_next - jnp.mean(X_next, axis=0)) ** 2)
    r2_dynamics = 1.0 - ss_dyn_res / (ss_dyn_tot + 1e-12)

    spectral_radius = jnp.max(jnp.abs(jnp.linalg.eigvals(A0)))
    trace_Q = jnp.trace(Q0)

    # ---- 3. Latent conditioning ----
    _, sv, _ = jnp.linalg.svd(X_flat, full_matrices=False)         # (D,)
    condition_number = sv[0] / (sv[-1] + 1e-12)

    return dict(
        r2_total=r2_total,
        r2_per_neuron=r2_per_neuron,
        r2_dynamics=r2_dynamics,
        spectral_radius=spectral_radius,
        trace_Q=trace_Q,
        singular_values_X=sv,
        condition_number_X=condition_number,
    )


# ===========================================================================
# Minimal usage example
# ===========================================================================
"""
if __name__ == "__main__":
    import jax.random as jr

    # -----------------------------------------------------------------------
    # Synthetic problem setup
    # -----------------------------------------------------------------------
    B, T, N = 4, 200, 30   # 4 batches, 200 timesteps, 30 neurons
    D_e, D_i = 4, 2        # 4 excitatory + 2 inhibitory latent dims
    D = D_e + D_i
    K = 2                   # two cell types: excitatory (type 0), inhibitory (type 1)
    N_e, N_i = 20, 10

    key = jr.PRNGKey(42)
    Y = jr.normal(key, (B, T, N))   # synthetic neural data

    # Boolean neuron-type masks
    e_mask = jnp.array([True]  * N_e + [False] * N_i)   # (N,)
    i_mask = jnp.array([False] * N_e + [True]  * N_i)   # (N,)

    # CTDS constraint arrays
    cell_types           = jnp.array([0, 1])
    cell_sign            = jnp.array([1, -1])
    cell_type_dimensions = jnp.array([D_e, D_i])
    cell_type_mask       = jnp.array([0] * N_e + [1] * N_i)   # (N,)

    # -----------------------------------------------------------------------
    # FA initialization
    # -----------------------------------------------------------------------
    print("Running FA initialization...")
    fa_init = fa_initialize_ctds(
        Y=Y,
        e_mask=e_mask,
        i_mask=i_mask,
        D_e=D_e,
        D_i=D_i,
        cell_types=cell_types,
        cell_sign=cell_sign,
        cell_type_dimensions=cell_type_dimensions,
        cell_type_mask=cell_type_mask,
        key=jr.PRNGKey(0),
        fa_iters=80,
        pgd_iters=300,
    )
    print("FA init shapes:")
    for k, v in fa_init.items():
        print(f"  {k}: {v.shape}")

    fa_diag = diagnostics(Y, fa_init)
    print(f"  FA R² total:      {fa_diag['r2_total']:.4f}")
    print(f"  FA R² dynamics:   {fa_diag['r2_dynamics']:.4f}")
    print(f"  FA spectral rad:  {fa_diag['spectral_radius']:.4f}")
    print(f"  FA cond(X):       {fa_diag['condition_number_X']:.2f}")

    # -----------------------------------------------------------------------
    # PCA initialization
    # -----------------------------------------------------------------------
    print("\nRunning PCA initialization...")
    pca_init = pca_initialize_ctds(
        Y=Y,
        e_mask=e_mask,
        i_mask=i_mask,
        D_e=D_e,
        D_i=D_i,
        cell_types=cell_types,
        cell_sign=cell_sign,
        cell_type_dimensions=cell_type_dimensions,
        cell_type_mask=cell_type_mask,
        key=jr.PRNGKey(1),
        pgd_iters=300,
    )
    print("PCA init shapes:")
    for k, v in pca_init.items():
        print(f"  {k}: {v.shape}")

    pca_diag = diagnostics(Y, pca_init)
    print(f"  PCA R² total:     {pca_diag['r2_total']:.4f}")
    print(f"  PCA R² dynamics:  {pca_diag['r2_dynamics']:.4f}")
    print(f"  PCA spectral rad: {pca_diag['spectral_radius']:.4f}")
    print(f"  PCA cond(X):      {pca_diag['condition_number_X']:.2f}")

    # -----------------------------------------------------------------------
    # Quick constraint checks
    # -----------------------------------------------------------------------
    print("\n--- Constraint checks (FA) ---")
    C0 = fa_init["C0"]
    A0 = fa_init["A0"]
    dynamics_mask = jnp.repeat(cell_sign, cell_type_dimensions)

    # C nonnegativity (within block)
    block_mask = _build_emission_mask(cell_type_mask, cell_types, cell_type_dimensions)
    min_C_in_block = jnp.min(jnp.where(block_mask, C0, jnp.inf))
    print(f"  min(C0) inside block: {min_C_in_block:.6f}  (should be >= 0)")

    # C zero outside block
    max_C_out_block = jnp.max(jnp.where(~block_mask, jnp.abs(C0), 0.0))
    print(f"  max|C0| outside block: {max_C_out_block:.6f}  (should be ~0)")

    # A Dale constraint: off-diagonal column signs
    D_mat = A0.shape[0]
    off_diag = ~jnp.eye(D_mat, dtype=bool)
    for j in range(D_mat):
        col = A0[:, j]
        sign = dynamics_mask[j]
        off_vals = col[jnp.arange(D_mat) != j]
        if sign == 1:
            viol = jnp.sum(off_vals < -1e-3)
            print(f"  A col {j} (exc): {viol} off-diag negative violations")
        else:
            viol = jnp.sum(off_vals > 1e-3)
            print(f"  A col {j} (inh): {viol} off-diag positive violations")

    # Q PSD
    eigs_Q = jnp.linalg.eigvalsh(fa_init["Q0"])
    print(f"  min eigenvalue Q0: {jnp.min(eigs_Q):.2e}  (should be > 0)")

"""

