"""
Test C-update (emission matrix) in isolation.
"""
import pytest
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxopt import BoxCDQP
from scipy.optimize import nnls as scipy_nnls
from test_helpers import (
    assert_nonnegative, check_kkt_conditions,
    generate_nonnegative_C
)
from models import create_emission_bounds

jax.config.update("jax_enable_x64", True)

_boxCDQP = BoxCDQP(tol=1e-7, maxiter=10000, verbose=False)


def test_C_update_kkt_conditions():
    """Test that C update satisfies KKT conditions for each row."""
    N = 10
    D = 4
    T = 100
    
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 5)
    
    # Generate synthetic latent trajectory
    latent = jax.random.normal(keys[0], (T, D))
    
    # Generate true C and observations
    C_true = generate_nonnegative_C(N, D, keys[1])
    obs_noise = jax.random.normal(keys[2], (T, N)) * 0.1
    observations = (C_true @ latent.T).T + obs_noise
    
    # Compute sufficient statistics
    Mxx = jnp.sum(latent[:, :, None] * latent[:, None, :], axis=0)  # (D, D)
    Ytil = observations.T @ latent  # (N, D)
    
    # Solve for C row-wise
    # Each row solves: min 0.5 c^T (2*Mxx) c - (2*Ytil[i])^T c  s.t. c >= 0
    
    # Test first row in detail
    i = 0
    P_row = 2.0 * Mxx
    q_row = -2.0 * Ytil[i]
    lb_row = jnp.zeros(D)
    ub_row = jnp.full(D, jnp.inf)
    
    c_init = jnp.ones(D) * 0.1
    result = _boxCDQP.run(c_init, params_obj=(P_row, q_row), params_ineq=(lb_row, ub_row))
    c_sol = result.params
    
    # Check KKT conditions
    kkt_results = check_kkt_conditions(c_sol, P_row, q_row, lb_row, ub_row, tol=1e-4)
    
    print(f"\nKKT Results for C update (row {i}):")
    print(f"  Interior violations: {kkt_results['interior_violations']} / {kkt_results['interior_count']}")
    print(f"  Interior max error: {kkt_results['interior_max_error']:.2e}")
    print(f"  Lower bound violations: {kkt_results['lower_violations']}")
    print(f"  Total violations: {kkt_results['total_violations']}")
    
    assert kkt_results['total_violations'] < 2, "Too many KKT violations"
    assert kkt_results['interior_max_error'] < 1e-3, "Interior stationarity violated"


def test_C_update_vs_scipy_nnls():
    """Compare C update to scipy NNLS for small problem."""
    N = 5
    D = 3
    T = 50
    
    key = jax.random.PRNGKey(123)
    keys = jax.random.split(key, 5)
    
    # Generate data
    latent = jax.random.normal(keys[0], (T, D))
    C_true = generate_nonnegative_C(N, D, keys[1])
    obs = (C_true @ latent.T).T + jax.random.normal(keys[2], (T, N)) * 0.1
    
    # Compute statistics
    Mxx = jnp.sum(latent[:, :, None] * latent[:, None, :], axis=0)
    Ytil = obs.T @ latent
    
    # Solve with JAX BoxCDQP
    P = 2.0 * Mxx
    q_matrix = -2.0 * Ytil
    lb = jnp.zeros((N, D))
    ub = jnp.full((N, D), jnp.inf)
    
    C_init = jnp.ones((N, D)) * 0.1
    vmap_solver = jax.vmap(_boxCDQP.run, in_axes=(0, (None, 1), (0, 0)))
    C_jax = vmap_solver(C_init, (P, q_matrix.T), (lb, ub)).params
    
    # Solve with scipy NNLS
    C_scipy = jnp.zeros((N, D))
    Mxx_np = np.array(Mxx)
    Ytil_np = np.array(Ytil)
    
    for i in range(N):
        # scipy.optimize.nnls solves: min ||Ax - b||^2  s.t. x >= 0
        # Our problem: min 0.5 x^T M x - y^T x
        # Equivalent to: min ||L x - L^{-T} y||^2 where M = L L^T
        L = np.linalg.cholesky(Mxx_np)
        b = np.linalg.solve(L.T, Ytil_np[i])
        c_row, _ = scipy_nnls(L, b)
        C_scipy = C_scipy.at[i].set(c_row)
    
    # Compare
    relative_error = jnp.linalg.norm(C_jax - C_scipy) / jnp.linalg.norm(C_scipy)
    print(f"\nRelative error between JAX and scipy: {relative_error:.2e}")
    print(f"C_jax[0]: {C_jax[0]}")
    print(f"C_scipy[0]: {C_scipy[0]}")
    
    assert relative_error < 0.2, f"Solutions differ: error={relative_error}"


def test_C_update_satisfies_nonnegativity():
    """Test that C update produces non-negative matrix."""
    N = 15
    D = 6
    T = 80
    
    key = jax.random.PRNGKey(456)
    keys = jax.random.split(key, 5)
    
    # Generate data
    latent = jax.random.normal(keys[0], (T, D))
    C_true = generate_nonnegative_C(N, D, keys[1])
    obs = (C_true @ latent.T).T + jax.random.normal(keys[2], (T, N)) * 0.2
    
    # Compute statistics
    Mxx = jnp.sum(latent[:, :, None] * latent[:, None, :], axis=0)
    Ytil = obs.T @ latent
    
    # Solve for C
    P = 2.0 * Mxx
    q_matrix = -2.0 * Ytil
    lb = jnp.zeros((N, D))
    ub = jnp.full((N, D), jnp.inf)
    
    C_init = jnp.ones((N, D)) * 0.1
    vmap_solver = jax.vmap(_boxCDQP.run, in_axes=(0, (None, 1), (0, 0)))
    C = vmap_solver(C_init, (P, q_matrix.T), (lb, ub)).params
    
    # Check non-negativity
    assert_nonnegative(C, "C", tol=-1e-6)


def test_emission_bounds_structure():
    """Test that create_emission_bounds produces correct block structure."""
    from params import ParamsCTDSConstraints
    
    # Setup: 3 cell types with different latent dimensions
    N = 30  # 10 neurons per cell type
    D = 10  # 2, 3, 5 dims per cell type
    
    cell_types = jnp.array([0, 1, 2])
    cell_sign = jnp.array([1, -1, 1])
    cell_type_dimensions = jnp.array([2, 3, 5])  # Total = 10
    cell_type_mask = jnp.concatenate([
        jnp.zeros(10, dtype=jnp.int32),   # First 10 neurons are type 0
        jnp.ones(10, dtype=jnp.int32),    # Next 10 are type 1
        jnp.full(10, 2, dtype=jnp.int32), # Last 10 are type 2
    ])
    
    constraints = ParamsCTDSConstraints(
        cell_types=cell_types,
        cell_sign=cell_sign,
        cell_type_dimensions=cell_type_dimensions,
        cell_type_mask=cell_type_mask
    )
    
    lb, ub = create_emission_bounds(constraints, D, N)
    
    # Check shapes
    assert lb.shape == (N, D), f"lb shape {lb.shape} != ({N}, {D})"
    assert ub.shape == (N, D), f"ub shape {ub.shape} != ({N}, {D})"
    
    # Check all lower bounds are zero
    assert jnp.allclose(lb, 0.0), "All lower bounds should be 0"
    
    # Map latent dimensions to cell types
    # Dims 0-1: type 0, Dims 2-4: type 1, Dims 5-9: type 2
    latent_to_cell_type = jnp.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
    
    # Check upper bounds have correct block structure
    for neuron_idx in range(N):
        neuron_type = cell_type_mask[neuron_idx]
        for latent_idx in range(D):
            latent_type = latent_to_cell_type[latent_idx]
            
            if neuron_type == latent_type:
                # Same cell type: should be inf (unconstrained)
                assert jnp.isinf(ub[neuron_idx, latent_idx]), \
                    f"ub[{neuron_idx}, {latent_idx}] should be inf (neuron type {neuron_type}, latent type {latent_type})"
            else:
                # Different cell types: should be 0 (forced to zero)
                assert ub[neuron_idx, latent_idx] == 0.0, \
                    f"ub[{neuron_idx}, {latent_idx}] should be 0 (neuron type {neuron_type}, latent type {latent_type})"
    
    # Explicitly check the blocks
    # Type 0 neurons (0-9) x Type 0 dims (0-1): should be inf
    assert jnp.all(jnp.isinf(ub[:10, :2])), "Type 0 neurons to type 0 dims should be unconstrained"
    # Type 0 neurons (0-9) x Type 1 dims (2-4): should be 0
    assert jnp.allclose(ub[:10, 2:5], 0.0), "Type 0 neurons to type 1 dims should be zero"
    # Type 0 neurons (0-9) x Type 2 dims (5-9): should be 0
    assert jnp.allclose(ub[:10, 5:], 0.0), "Type 0 neurons to type 2 dims should be zero"
    
    # Type 1 neurons (10-19) x Type 1 dims (2-4): should be inf
    assert jnp.all(jnp.isinf(ub[10:20, 2:5])), "Type 1 neurons to type 1 dims should be unconstrained"
    # Type 1 neurons (10-19) x Type 0 dims (0-1): should be 0
    assert jnp.allclose(ub[10:20, :2], 0.0), "Type 1 neurons to type 0 dims should be zero"
    # Type 1 neurons (10-19) x Type 2 dims (5-9): should be 0
    assert jnp.allclose(ub[10:20, 5:], 0.0), "Type 1 neurons to type 2 dims should be zero"
    
    # Type 2 neurons (20-29) x Type 2 dims (5-9): should be inf
    assert jnp.all(jnp.isinf(ub[20:, 5:])), "Type 2 neurons to type 2 dims should be unconstrained"
    # Type 2 neurons (20-29) x Type 0 dims (0-1): should be 0
    assert jnp.allclose(ub[20:, :2], 0.0), "Type 2 neurons to type 0 dims should be zero"
    # Type 2 neurons (20-29) x Type 1 dims (2-4): should be 0
    assert jnp.allclose(ub[20:, 2:5], 0.0), "Type 2 neurons to type 1 dims should be zero"
    
    print("\n✓ Emission bounds have correct block structure")


def test_C_update_preserves_block_structure():
    """Test that C update maintains block-diagonal structure with multiple cell types."""
    from params import ParamsCTDSConstraints
    
    # Setup: 3 cell types
    N = 30
    D = 10
    T = 200
    
    cell_types = jnp.array([0, 1, 2])
    cell_sign = jnp.array([1, -1, 1])
    cell_type_dimensions = jnp.array([2, 3, 5])
    cell_type_mask = jnp.concatenate([
        jnp.zeros(10, dtype=jnp.int32),
        jnp.ones(10, dtype=jnp.int32),
        jnp.full(10, 2, dtype=jnp.int32),
    ])
    
    constraints = ParamsCTDSConstraints(
        cell_types=cell_types,
        cell_sign=cell_sign,
        cell_type_dimensions=cell_type_dimensions,
        cell_type_mask=cell_type_mask
    )
    
    # Generate data with true block structure
    key = jax.random.PRNGKey(101)
    keys = jax.random.split(key, 6)
    
    latent = jax.random.normal(keys[0], (T, D))
    
    # Generate true C with block structure
    C_true = jnp.zeros((N, D))
    C_true = C_true.at[:10, :2].set(jnp.abs(jax.random.normal(keys[1], (10, 2))))
    C_true = C_true.at[10:20, 2:5].set(jnp.abs(jax.random.normal(keys[2], (10, 3))))
    C_true = C_true.at[20:, 5:].set(jnp.abs(jax.random.normal(keys[3], (10, 5))))
    
    obs = (C_true @ latent.T).T + jax.random.normal(keys[4], (T, N)) * 0.1
    
    # Compute sufficient statistics
    Mxx = jnp.sum(latent[:, :, None] * latent[:, None, :], axis=0)
    Ytil = obs.T @ latent
    
    # Get bounds and solve
    lb, ub = create_emission_bounds(constraints, D, N)
    P = 2.0 * Mxx
    q_matrix = -2.0 * Ytil
    C_init = jnp.ones((N, D)) * 0.1
    
    vmap_solver = jax.vmap(_boxCDQP.run, in_axes=(0, (None, 1), (0, 0)))
    C = vmap_solver(C_init, (P, q_matrix.T), (lb, ub)).params
    
    # Create latent-to-cell-type mapping
    latent_to_cell_type = jnp.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
    
    # Check block structure: verify zeros in off-diagonal blocks
    for neuron_idx in range(N):
        neuron_type = cell_type_mask[neuron_idx]
        for latent_idx in range(D):
            latent_type = latent_to_cell_type[latent_idx]
            
            if neuron_type != latent_type:
                # Different cell types: must be exactly zero
                assert jnp.abs(C[neuron_idx, latent_idx]) < 1e-8, \
                    f"C[{neuron_idx}, {latent_idx}] = {C[neuron_idx, latent_idx]:.6e} should be 0 " \
                    f"(neuron type {neuron_type} ≠ latent type {latent_type})"
    
    # Check explicit blocks
    assert jnp.allclose(C[:10, 2:], 0.0, atol=1e-8), "Type 0 neurons leak to non-type-0 dims"
    assert jnp.allclose(C[10:20, :2], 0.0, atol=1e-8), "Type 1 neurons leak to type 0 dims"
    assert jnp.allclose(C[10:20, 5:], 0.0, atol=1e-8), "Type 1 neurons leak to type 2 dims"
    assert jnp.allclose(C[20:, :5], 0.0, atol=1e-8), "Type 2 neurons leak to non-type-2 dims"
    
    # Check non-negativity in allowed blocks
    assert_nonnegative(C[:10, :2], "Type 0 block", tol=-1e-6)
    assert_nonnegative(C[10:20, 2:5], "Type 1 block", tol=-1e-6)
    assert_nonnegative(C[20:, 5:], "Type 2 block", tol=-1e-6)
    
    # Check that allowed blocks have non-zero entries (sanity check)
    assert jnp.linalg.norm(C[:10, :2]) > 0.1, "Type 0 block should have learned weights"
    assert jnp.linalg.norm(C[10:20, 2:5]) > 0.1, "Type 1 block should have learned weights"
    assert jnp.linalg.norm(C[20:, 5:]) > 0.1, "Type 2 block should have learned weights"
    
    print("\n✓ C update preserves block structure")
    print(f"  Type 0 block norm: {jnp.linalg.norm(C[:10, :2]):.3f}")
    print(f"  Type 1 block norm: {jnp.linalg.norm(C[10:20, 2:5]):.3f}")
    print(f"  Type 2 block norm: {jnp.linalg.norm(C[20:, 5:]):.3f}")


def test_C_update_nonnegativity_with_bounds():
    """Test that C update respects non-negativity even with tight bounds."""
    from params import ParamsCTDSConstraints
    
    N = 15
    D = 6
    T = 80
    
    # 2 cell types
    cell_types = jnp.array([0, 1])
    cell_sign = jnp.array([1, 1])
    cell_type_dimensions = jnp.array([3, 3])
    cell_type_mask = jnp.concatenate([
        jnp.zeros(8, dtype=jnp.int32),
        jnp.ones(7, dtype=jnp.int32)
    ])
    
    constraints = ParamsCTDSConstraints(
        cell_types=cell_types,
        cell_sign=cell_sign,
        cell_type_dimensions=cell_type_dimensions,
        cell_type_mask=cell_type_mask
    )
    
    # Generate data with varying scales
    key = jax.random.PRNGKey(303)
    keys = jax.random.split(key, 5)
    
    latent = jax.random.normal(keys[0], (T, D)) * 2.0  # Larger scale
    
    C_true = jnp.zeros((N, D))
    C_true = C_true.at[:8, :3].set(jnp.abs(jax.random.normal(keys[1], (8, 3))) * 0.5)
    C_true = C_true.at[8:, 3:].set(jnp.abs(jax.random.normal(keys[2], (7, 3))) * 1.5)
    
    obs = (C_true @ latent.T).T + jax.random.normal(keys[3], (T, N)) * 0.2
    
    # Compute statistics
    Mxx = jnp.sum(latent[:, :, None] * latent[:, None, :], axis=0)
    Ytil = obs.T @ latent
    
    lb, ub = create_emission_bounds(constraints, D, N)
    P = 2.0 * Mxx
    q_matrix = -2.0 * Ytil
    C_init = jnp.ones((N, D)) * 0.1
    
    vmap_solver = jax.vmap(_boxCDQP.run, in_axes=(0, (None, 1), (0, 0)))
    C = vmap_solver(C_init, (P, q_matrix.T), (lb, ub)).params
    
    # Verify strict non-negativity
    assert jnp.all(C >= 0.0), f"C has negative entries: min = {jnp.min(C)}"
    
    # Check that allowed blocks have recovered non-zero weights
    type0_block = C[:8, :3]
    type1_block = C[8:, 3:]
    
    assert jnp.all(type0_block >= 0.0), "Type 0 block has negative values"
    assert jnp.all(type1_block >= 0.0), "Type 1 block has negative values"
    
    # Check forbidden blocks are exactly zero
    assert jnp.allclose(C[:8, 3:], 0.0, atol=1e-8), "Type 0 neurons leak to type 1 dims"
    assert jnp.allclose(C[8:, :3], 0.0, atol=1e-8), "Type 1 neurons leak to type 0 dims"
    
    print(f"\n✓ C respects non-negativity: min={jnp.min(C):.6f}, max={jnp.max(C):.6f}")


def test_C_update_bounds_enforcement():
    """Test that upper bounds of 0 strictly enforce zeros, and inf allows positive values."""
    from params import ParamsCTDSConstraints
    
    N = 12
    D = 4
    T = 100
    
    cell_types = jnp.array([0, 1])
    cell_sign = jnp.array([1, 1])
    cell_type_dimensions = jnp.array([2, 2])
    cell_type_mask = jnp.concatenate([
        jnp.zeros(6, dtype=jnp.int32),
        jnp.ones(6, dtype=jnp.int32)
    ])
    
    constraints = ParamsCTDSConstraints(
        cell_types=cell_types,
        cell_sign=cell_sign,
        cell_type_dimensions=cell_type_dimensions,
        cell_type_mask=cell_type_mask
    )
    
    # Generate data where cross-type correlations would be strong without constraints
    key = jax.random.PRNGKey(404)
    keys = jax.random.split(key, 5)
    
    latent = jax.random.normal(keys[0], (T, D))
    
    # Create C that violates block structure (intentionally)
    C_violating = jnp.abs(jax.random.normal(keys[1], (N, D)))  # All entries positive
    
    # But generate observations from correct block structure
    C_true = jnp.zeros((N, D))
    C_true = C_true.at[:6, :2].set(jnp.abs(jax.random.normal(keys[2], (6, 2))))
    C_true = C_true.at[6:, 2:].set(jnp.abs(jax.random.normal(keys[3], (6, 2))))
    
    obs = (C_true @ latent.T).T + jax.random.normal(keys[4], (T, N)) * 0.05
    
    # Compute statistics
    Mxx = jnp.sum(latent[:, :, None] * latent[:, None, :], axis=0)
    Ytil = obs.T @ latent
    
    lb, ub = create_emission_bounds(constraints, D, N)
    
    # Verify bounds structure
    assert jnp.all(jnp.isinf(ub[:6, :2])), "Type 0 to type 0 should be unconstrained"
    assert jnp.allclose(ub[:6, 2:], 0.0), "Type 0 to type 1 should be bounded to zero"
    assert jnp.all(jnp.isinf(ub[6:, 2:])), "Type 1 to type 1 should be unconstrained"
    assert jnp.allclose(ub[6:, :2], 0.0), "Type 1 to type 0 should be bounded to zero"
    
    # Solve
    P = 2.0 * Mxx
    q_matrix = -2.0 * Ytil
    C_init = C_violating  # Start from violating initialization
    
    vmap_solver = jax.vmap(_boxCDQP.run, in_axes=(0, (None, 1), (0, 0)))
    C = vmap_solver(C_init, (P, q_matrix.T), (lb, ub)).params
    
    # Check that bounds were enforced despite violating initialization
    assert jnp.allclose(C[:6, 2:], 0.0, atol=1e-9), "Bounds failed to enforce zeros for type 0 to type 1"
    assert jnp.allclose(C[6:, :2], 0.0, atol=1e-9), "Bounds failed to enforce zeros for type 1 to type 0"
    
    # Check that allowed blocks have positive weights
    assert jnp.all(C[:6, :2] >= 0.0), "Type 0 block should be non-negative"
    assert jnp.all(C[6:, 2:] >= 0.0), "Type 1 block should be non-negative"
    assert jnp.sum(C[:6, :2]) > 0.1, "Type 0 block should have learned weights"
    assert jnp.sum(C[6:, 2:]) > 0.1, "Type 1 block should have learned weights"
    
    print(f"\n✓ Bounds strictly enforced despite violating initialization")
    print(f"  Forbidden entries (should be 0): max = {max(jnp.max(jnp.abs(C[:6, 2:])), jnp.max(jnp.abs(C[6:, :2]))):.2e}")
    print(f"  Allowed entries: min = {min(jnp.min(C[:6, :2]), jnp.min(C[6:, 2:])):.3f}")


def visualize_C_block_structure(C: Array, cell_type_mask: Array, 
                                 cell_type_dimensions: Array) -> str:
    """
    Helper to visualize C matrix block structure for debugging.
    
    Returns a string showing which entries are zero/non-zero with cell type annotations.
    """
    N, D = C.shape
    
    # Create latent-to-cell-type mapping
    num_types = len(cell_type_dimensions)
    latent_to_cell_type = jnp.repeat(
        jnp.arange(num_types),
        cell_type_dimensions
    )
    
    lines = ["\nC Matrix Block Structure:"]
    lines.append(f"Shape: ({N}, {D})")
    lines.append(f"Latent dims by type: {latent_to_cell_type.tolist()}")
    lines.append("")
    
    # Header showing latent dim types
    header = "    " + " ".join(f"{i}" for i in range(D))
    lines.append(header)
    lines.append("    " + "-" * (2 * D - 1))
    
    # Show each neuron
    for i in range(N):
        neuron_type = cell_type_mask[i]
        row_str = f"n{i:2d}|"
        
        for j in range(D):
            latent_type = latent_to_cell_type[j]
            val = C[i, j]
            
            if neuron_type != latent_type:
                # Should be zero
                if jnp.abs(val) < 1e-8:
                    row_str += " ."
                else:
                    row_str += " X"  # ERROR: non-zero where should be zero
            else:
                # Allowed block
                if jnp.abs(val) < 1e-8:
                    row_str += " 0"
                else:
                    row_str += " +"
        
        row_str += f"  (type {neuron_type})"
        lines.append(row_str)
    
    lines.append("")
    lines.append("Legend: . = zero (correct), + = positive (allowed), 0 = zero (allowed), X = ERROR (non-zero forbidden)")
    
    return "\n".join(lines)


def test_visualize_block_structure():
    """Test the block structure visualization helper."""
    from params import ParamsCTDSConstraints
    
    N = 12
    D = 6
    
    cell_types = jnp.array([0, 1])
    cell_sign = jnp.array([1, 1])
    cell_type_dimensions = jnp.array([3, 3])
    cell_type_mask = jnp.concatenate([
        jnp.zeros(6, dtype=jnp.int32),
        jnp.ones(6, dtype=jnp.int32)
    ])
    
    constraints = ParamsCTDSConstraints(
        cell_types=cell_types,
        cell_sign=cell_sign,
        cell_type_dimensions=cell_type_dimensions,
        cell_type_mask=cell_type_mask
    )
    
    # Create a C with correct block structure
    key = jax.random.PRNGKey(505)
    keys = jax.random.split(key, 3)
    
    C_correct = jnp.zeros((N, D))
    C_correct = C_correct.at[:6, :3].set(jnp.abs(jax.random.normal(keys[0], (6, 3))))
    C_correct = C_correct.at[6:, 3:].set(jnp.abs(jax.random.normal(keys[1], (6, 3))))
    
    viz = visualize_C_block_structure(C_correct, cell_type_mask, cell_type_dimensions)
    print(viz)
    
    # Count actual errors (X symbols that aren't in the legend)
    lines = viz.split('\n')
    error_count = sum(1 for line in lines if '|' in line and 'X' in line.split('|')[1])
    
    assert error_count == 0, f"Visualization shows {error_count} errors for correct structure"


def test_C_update_with_block_constraints():
    """Test C update with cell-type block constraints."""
    N = 20
    D = 6
    T = 100
    
    # Setup: 2 cell types, 10 neurons each
    cell_types = jnp.array([0, 1])
    cell_sign = jnp.array([1, -1])
    cell_type_dimensions = jnp.array([3, 3])
    cell_type_mask = jnp.concatenate([
        jnp.zeros(10, dtype=jnp.int32),
        jnp.ones(10, dtype=jnp.int32)
    ])
    
    from params import ParamsCTDSConstraints
    constraints = ParamsCTDSConstraints(
        cell_types=cell_types,
        cell_sign=cell_sign,
        cell_type_dimensions=cell_type_dimensions,
        cell_type_mask=cell_type_mask
    )
    
    # Generate data
    key = jax.random.PRNGKey(789)
    keys = jax.random.split(key, 5)
    
    latent = jax.random.normal(keys[0], (T, D))
    
    # Generate true C with block structure
    C_true = jnp.zeros((N, D))
    # Type 0 neurons only connect to first 3 dims
    C_true = C_true.at[:10, :3].set(jnp.abs(jax.random.normal(keys[1], (10, 3))))
    # Type 1 neurons only connect to last 3 dims
    C_true = C_true.at[10:, 3:].set(jnp.abs(jax.random.normal(keys[2], (10, 3))))
    
    obs = (C_true @ latent.T).T + jax.random.normal(keys[3], (T, N)) * 0.1
    
    # Compute statistics
    Mxx = jnp.sum(latent[:, :, None] * latent[:, None, :], axis=0)
    Ytil = obs.T @ latent
    
    # Get block-constrained bounds
    lb, ub = create_emission_bounds(constraints, D, N)
    
    # Solve for C
    P = 2.0 * Mxx
    q_matrix = -2.0 * Ytil
    C_init = jnp.ones((N, D)) * 0.1
    
    vmap_solver = jax.vmap(_boxCDQP.run, in_axes=(0, (None, 1), (0, 0)))
    C = vmap_solver(C_init, (P, q_matrix.T), (lb, ub)).params
    
    print("\nRecovered C structure:")
    print(f"Type 0 neurons (0-9) to type 0 dims (0-2): {jnp.linalg.norm(C[:10, :3]):.3f}")
    print(f"Type 0 neurons (0-9) to type 1 dims (3-5): {jnp.linalg.norm(C[:10, 3:]):.3f}")
    print(f"Type 1 neurons (10-19) to type 0 dims (0-2): {jnp.linalg.norm(C[10:, :3]):.3f}")
    print(f"Type 1 neurons (10-19) to type 1 dims (3-5): {jnp.linalg.norm(C[10:, 3:]):.3f}")
    
    # Check block structure is enforced
    assert jnp.allclose(C[:10, 3:], 0.0, atol=1e-6), "Type 0 neurons leak to type 1 dims"
    assert jnp.allclose(C[10:, :3], 0.0, atol=1e-6), "Type 1 neurons leak to type 0 dims"
    
    # Check non-negativity
    assert_nonnegative(C, "C", tol=-1e-6)


def test_C_update_orientation():
    """Test that C update has correct matrix orientation (N x D)."""
    N = 8
    D = 4
    T = 60
    
    key = jax.random.PRNGKey(999)
    keys = jax.random.split(key, 5)
    
    # Generate data with specific structure
    latent = jax.random.normal(keys[0], (T, D))
    
    # C with clear row structure
    C_true = jnp.array([
        [1.0, 0.0, 0.0, 0.0],  # Neuron 0 only sees dim 0
        [0.0, 1.0, 0.0, 0.0],  # Neuron 1 only sees dim 1
        [0.5, 0.5, 0.0, 0.0],  # Neuron 2 sees dims 0,1
        [0.0, 0.0, 1.0, 0.0],  # Neuron 3 only sees dim 2
        [0.0, 0.0, 0.0, 1.0],  # Neuron 4 only sees dim 3
        [0.0, 0.0, 0.5, 0.5],  # Neuron 5 sees dims 2,3
        [0.3, 0.3, 0.3, 0.0],  # Neuron 6 sees dims 0,1,2
        [0.25, 0.25, 0.25, 0.25],  # Neuron 7 sees all
    ])
    C_true = generate_nonnegative_C(N, D, keys[1])
    
    obs = (C_true @ latent.T).T + jax.random.normal(keys[1], (T, N)) * 0.05
    
    # Compute statistics
    Mxx = jnp.sum(latent[:, :, None] * latent[:, None, :], axis=0)
    Ytil = obs.T @ latent  # Should be (N, D)
    
    print(f"\nYtil shape: {Ytil.shape} (should be (N={N}, D={D}))")
    assert Ytil.shape == (N, D), f"Ytil has wrong shape: {Ytil.shape}"
    
    # Solve for C
    P = 2.0 * Mxx
    q_matrix = -2.0 * Ytil
    lb = jnp.zeros((N, D))
    ub = jnp.full((N, D), jnp.inf)
    
    C_init = jnp.ones((N, D)) * 0.1
    vmap_solver = jax.vmap(_boxCDQP.run, in_axes=(0, (None, 1), (0, 0)))
    C = vmap_solver(C_init, (P, q_matrix.T), (lb, ub)).params
    
    print(f"\nC shape: {C.shape} (should be (N={N}, D={D}))")
    assert C.shape == (N, D), f"C has wrong shape: {C.shape}"
    print(f"\nC {C}")
    
    print("\nTrue C[0:3]:")
    print(C_true[:3])
    print("\nRecovered C[0:3]:")
    print(C[:3])
    
    # Check that structure is approximately recovered
    # Neuron 0 should have most weight on dim 0
    assert jnp.argmax(C[0]) == 0, "Neuron 0 structure not recovered"
    # Neuron 1 should have most weight on dim 1
    assert jnp.argmax(C[1]) == 1, "Neuron 1 structure not recovered"


def test_C_update_with_regularization():
    """Test C update stability with regularized Mxx."""
    N = 12
    D = 5
    T = 40  # Small T
    
    key = jax.random.PRNGKey(111)
    keys = jax.random.split(key, 5)
    
    # Generate data
    latent = jax.random.normal(keys[0], (T, D)) * 0.5
    C_true = generate_nonnegative_C(N, D, keys[1])
    obs = (C_true @ latent.T).T + jax.random.normal(keys[2], (T, N)) * 0.1
    
    # Compute statistics with regularization
    Mxx = jnp.sum(latent[:, :, None] * latent[:, None, :], axis=0)
    Mxx = Mxx + 1e-5 * jnp.eye(D)  # Add regularization
    Ytil = obs.T @ latent
    
    # Check Mxx is well-conditioned
    cond_Mxx = jnp.linalg.cond(Mxx)
    print(f"\nCondition number of Mxx: {cond_Mxx:.2e}")
    
    # Solve for C
    P = 2.0 * Mxx
    q_matrix = -2.0 * Ytil
    lb = jnp.zeros((N, D))
    ub = jnp.full((N, D), jnp.inf)
    
    C_init = jnp.ones((N, D)) * 0.1
    vmap_solver = jax.vmap(_boxCDQP.run, in_axes=(0, (None, 1), (0, 0)))
    C = vmap_solver(C_init, (P, q_matrix.T), (lb, ub)).params
    
    # Check no NaNs or Infs
    assert not jnp.any(jnp.isnan(C)), "C contains NaN"
    assert not jnp.any(jnp.isinf(C)), "C contains Inf"
    
    # Check non-negativity
    assert_nonnegative(C, "C", tol=-1e-6)
