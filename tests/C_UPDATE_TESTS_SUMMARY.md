# C Update Block Structure Tests - Summary

## Overview
Added comprehensive tests to verify that the C (emission matrix) update correctly preserves cell-type block structure and enforces non-negativity constraints.

## New Tests Added

### 1. `test_emission_bounds_structure()`
**Purpose:** Verify that `create_emission_bounds()` produces correct block-diagonal structure.

**What it tests:**
- Bounds have correct shape (N, D)
- All lower bounds are 0
- Upper bounds are `inf` for same cell type (unconstrained)
- Upper bounds are `0` for different cell types (forced to zero)
- Explicit verification of all 3×3 blocks for 3 cell types

**Key assertion:**
```python
# For each (neuron, latent_dim) pair:
if neuron_type == latent_type:
    assert ub[i, j] == inf  # Unconstrained
else:
    assert ub[i, j] == 0.0  # Forced to zero
```

### 2. `test_C_update_preserves_block_structure()`
**Purpose:** Verify that C update maintains block-diagonal structure with multiple cell types.

**What it tests:**
- Uses 3 cell types (10 neurons each, 2+3+5 latent dims)
- Verifies zeros in off-diagonal blocks (element-by-element check)
- Checks non-negativity in allowed diagonal blocks
- Confirms learned weights are non-zero in allowed blocks

**Key assertions:**
```python
# For every entry in C:
for neuron_idx in range(N):
    for latent_idx in range(D):
        if neuron_type != latent_type:
            assert |C[i,j]| < 1e-8  # Must be exactly zero

# Block-level checks:
assert C[:10, 2:] ≈ 0    # Type 0 neurons don't leak to other types
assert C[10:20, :2] ≈ 0  # Type 1 neurons don't leak to type 0
assert C[20:, :5] ≈ 0    # Type 2 neurons don't leak to types 0,1
```

### 3. `test_C_update_nonnegativity_with_bounds()`
**Purpose:** Test non-negativity enforcement even with varying data scales.

**What it tests:**
- Different scales for latent states and observations
- Unequal cell type sizes (8 neurons vs 7 neurons)
- Strict non-negativity: `C >= 0` everywhere
- Forbidden blocks are exactly zero
- Allowed blocks learn positive weights

**Key assertion:**
```python
assert jnp.all(C >= 0.0), f"C has negative entries: min = {jnp.min(C)}"
```

### 4. `test_C_update_bounds_enforcement()`
**Purpose:** Verify that bounds are strictly enforced even with violating initialization.

**What it tests:**
- Starts from initialization that violates block structure
- Generates data where cross-type correlations exist
- Confirms solver enforces bounds despite bad initialization
- Verifies upper bounds of 0 force entries to exactly 0
- Confirms inf bounds allow positive learning

**Key concept:**
```python
C_init = C_violating  # All entries positive (violates structure)
# After BoxCDQP solve:
assert C[:6, 2:] ≈ 0  # Bounds forced these to zero
assert C[6:, :2] ≈ 0  # Bounds forced these to zero
```

### 5. `visualize_C_block_structure()`
**Purpose:** Helper function to visualize C matrix block structure for debugging.

**What it does:**
- Creates ASCII visualization of C matrix
- Shows which entries are:
  - `.` = zero (correct, forbidden block)
  - `+` = positive (correct, allowed block)
  - `0` = zero (in allowed block, might indicate underfitting)
  - `X` = **ERROR** - non-zero in forbidden block
- Annotates each neuron with its cell type
- Shows latent dimension to cell type mapping

**Example output:**
```
C Matrix Block Structure:
Shape: (12, 6)
Latent dims by type: [0, 0, 0, 1, 1, 1]

    0 1 2 3 4 5
    -----------
n 0| + + + . . .  (type 0)
n 1| + + + . . .  (type 0)
...
n 6| . . . + + +  (type 1)
n 7| . . . + + +  (type 1)
...
```

### 6. `test_visualize_block_structure()`
**Purpose:** Test the visualization helper with correct block structure.

## What These Tests Verify

### Block Structure Properties
1. **Diagonal blocks** (same cell type): Non-negative, can be positive
2. **Off-diagonal blocks** (different cell types): Exactly zero (< 1e-8)
3. **Bounds correctness**: `lb=0, ub=inf` for diagonal, `lb=0, ub=0` for off-diagonal

### Non-Negativity Properties
1. All entries `C[i,j] >= 0` (no negative values anywhere)
2. Non-negativity preserved even with:
   - Varying data scales
   - Unequal cell type sizes
   - Large latent states
   - High noise levels

### Solver Properties
1. BoxCDQP respects bounds exactly (not approximately)
2. Bounds are enforced regardless of initialization
3. Upper bound of 0 forces entries to be exactly 0
4. Upper bound of inf allows learning of positive weights

## How to Use These Tests

### Run all C update tests:
```bash
pytest tests/test_C_update.py -v
```

### Run specific block structure tests:
```bash
pytest tests/test_C_update.py::test_emission_bounds_structure -v
pytest tests/test_C_update.py::test_C_update_preserves_block_structure -v
```

### Debug with visualization:
```bash
pytest tests/test_C_update.py::test_visualize_block_structure -v -s
```

### Check if C update is working correctly:
If you see NaN in your EM iterations, run these tests to verify:
1. Bounds are created correctly: `test_emission_bounds_structure`
2. C update preserves structure: `test_C_update_preserves_block_structure`
3. Non-negativity is enforced: `test_C_update_nonnegativity_with_bounds`

## Integration with models.py

These tests verify the correctness of:

1. **`create_emission_bounds()`** (lines 20-56 in models.py):
   - Creates `(N, D)` lb and ub arrays
   - Uses `cell_type_mask` and `cell_type_dimensions`
   - Enforces block structure via bounds

2. **C update in `m_step()`** (around line 497 in models.py):
   ```python
   lb, ub = create_emission_bounds(constraints, D, N)
   P = 2.0 * Mxx
   q_matrix = -2.0 * Ytil
   vmap_solver = jax.vmap(_boxCDQP.run, in_axes=(0, (None, 1), (0, 0)))
   C = vmap_solver(C_init, (P, q_matrix.T), (lb, ub)).params
   ```

## Common Issues These Tests Catch

1. **Leakage across cell types**: C[i,j] ≠ 0 when neuron i and latent dim j have different cell types
2. **Negative entries**: C[i,j] < 0 (violates non-negativity)
3. **Incorrect bounds**: `ub` not matching cell type structure
4. **Solver not respecting bounds**: BoxCDQP producing values outside [lb, ub]

## Test Coverage

- ✅ Bounds generation (`create_emission_bounds`)
- ✅ Block structure preservation (3 cell types)
- ✅ Non-negativity enforcement
- ✅ Bounds enforcement with violating initialization
- ✅ Multiple cell type scenarios
- ✅ Varying data scales
- ✅ Unequal cell type sizes
- ✅ Visualization for debugging

## Next Steps

If tests pass but you still see NaN:
1. Check A update (dynamics matrix) - might be going unstable
2. Check Q update - might be singular or negative eigenvalues
3. Check R update - verify formula is correct (should be `Y @ Y.T`, not `Y @ obs`)
4. Add regularization to Q and R
