"""
Helper functions and fixtures for testing CTDS EM algorithm.
"""
import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Optional
from params import ParamsCTDS, ParamsCTDSDynamics, ParamsCTDSEmissions, ParamsCTDSInitial, ParamsCTDSConstraints

jax.config.update("jax_enable_x64", True)

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
                           Q_scale: float = 0.1,
                           R_scale: float = 0.5) -> Tuple[ParamsCTDS, jnp.ndarray, jnp.ndarray]:
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
    A = generate_stable_A(D, dynamics_mask, keys[0])
    Q = jnp.eye(D) * Q_scale
    C = generate_nonnegative_C(N, D, keys[1])
    R = jnp.eye(N) * R_scale
    
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
    observation_noise = jax.random.multivariate_normal(keys[7], jnp.zeros(N), R, (T,))
    observations = (C @ latent_states.T).T + observation_noise
    
    # Create parameter object
    constraints = ParamsCTDSConstraints(
        cell_types=cell_types,
        cell_sign=cell_sign,
        cell_type_dimensions=cell_type_dimensions,
        cell_type_mask=cell_type_mask
    )
    
    initial = ParamsCTDSInitial(mean=initial_mean, cov=initial_cov)
    dynamics = ParamsCTDSDynamics(weights=A, cov=Q, dynamics_mask=dynamics_mask)
    emissions = ParamsCTDSEmissions(weights=C, cov=R)
    
    params = ParamsCTDS(
        initial=initial,
        dynamics=dynamics,
        emissions=emissions,
        constraints=constraints,
        observations=observations 
    )
    
    return params, latent_states, observations


def perturb_params(params: ParamsCTDS, key: jax.random.PRNGKey, 
                  scale: float = 0.3) -> ParamsCTDS:
    """Perturb parameters while maintaining constraints."""
    keys = jax.random.split(key, 4)
    
    D = params.dynamics.weights.shape[0]
    N = params.emissions.weights.shape[0]
    
    # Perturb A while maintaining Dale constraints
    A_noise = jax.random.normal(keys[0], (D, D)) * scale * 0.1
    A_new = params.dynamics.weights + A_noise
    
    # Re-apply Dale constraints
    dynamics_mask = params.dynamics.dynamics_mask
    for j in range(D):
        if dynamics_mask[j] == 1:  # Excitatory
            off_diag_mask = jnp.arange(D) != j
            A_new = A_new.at[off_diag_mask, j].set(jnp.maximum(A_new[off_diag_mask, j], 0.0))
        elif dynamics_mask[j] == -1:  # Inhibitory
            off_diag_mask = jnp.arange(D) != j
            A_new = A_new.at[off_diag_mask, j].set(jnp.minimum(A_new[off_diag_mask, j], 0.0))
    
    # Perturb C while maintaining non-negativity
    C_noise = jax.random.normal(keys[1], (N, D)) * scale * 0.1
    C_new = jnp.maximum(params.emissions.weights + C_noise, 0.0)
    
    # Perturb Q and R
    Q_new = params.dynamics.cov + jnp.eye(D) * jax.random.uniform(keys[2]) * scale * 0.01
    R_new = params.emissions.cov + jnp.eye(N) * jax.random.uniform(keys[3]) * scale * 0.01
    
    # Create new parameter object
    initial_new = ParamsCTDSInitial(
        mean=params.initial.mean,
        cov=params.initial.cov
    )
    dynamics_new = ParamsCTDSDynamics(
        weights=A_new,
        cov=Q_new,
        dynamics_mask=dynamics_mask
    )
    emissions_new = ParamsCTDSEmissions(
        weights=C_new,
        cov=R_new
    )
    
    return ParamsCTDS(
        initial=initial_new,
        dynamics=dynamics_new,
        emissions=emissions_new,
        constraints=params.constraints,
        observations=params.observations
    )


# ============================================================================
# GEOMETRY UTILITIES
# ============================================================================

def principal_angles(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    """
    Compute principal angles between column spaces of A and B.
    
    Returns angles in [0, π/2].
    """
    # Orthonormalize columns
    QA, _ = jnp.linalg.qr(A)
    QB, _ = jnp.linalg.qr(B)
    
    # Compute singular values of Q_A^T Q_B
    _, s, _ = jnp.linalg.svd(QA.T @ QB)
    
    # Clamp to [0, 1] for numerical stability
    s = jnp.clip(s, 0.0, 1.0)
    
    # Angles from singular values
    angles = jnp.arccos(s)
    
    return angles


def subspace_distance(A: jnp.ndarray, B: jnp.ndarray) -> float:
    """Compute max principal angle between column spaces."""
    angles = principal_angles(A, B)
    return float(jnp.max(angles))
