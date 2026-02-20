# CTDS EM Algorithm Test Suite

Complete pytest test suite for the Cell-Type Dynamical Systems (CTDS) constrained EM algorithm.

## Installation

```bash
pip install pytest scipy
```

## Running Tests

### Run all tests
```bash
pytest tests/
```

### Run specific test files
```bash
# Structural invariants
pytest tests/test_invariants.py -v

# A-update tests
pytest tests/test_A_update.py -v

# C-update tests
pytest tests/test_C_update.py -v

# Q/R update tests
pytest tests/test_QR_update.py -v

# EM convergence tests
pytest tests/test_em_convergence.py -v

# Numerical stability tests
pytest tests/test_numerical_stability.py -v
```

### Run specific test functions
```bash
pytest tests/test_invariants.py::test_Q_is_psd_after_m_step -v
pytest tests/test_A_update.py::test_A_update_kkt_conditions -v
```

### Run with output
```bash
pytest tests/ -v -s  # -s shows print statements
```

### Run in parallel (faster)
```bash
pip install pytest-xdist
pytest tests/ -n auto  # Use all CPU cores
```

## Test Organization

### `test_helpers.py`
- Assertion utilities: `assert_psd`, `assert_dale_columns`, `assert_nonnegative`
- KKT condition checker: `check_kkt_conditions`
- Data generation: `generate_synthetic_ssm`, `generate_stable_A`, `generate_nonnegative_C`
- Geometry utilities: `principal_angles`, `subspace_distance`

### `test_invariants.py`
Tests that all constraints and structural properties are maintained after M-step:
- Parameter shapes (A, C, Q, R)
- Q and R are PSD
- Dale's law constraints on A
- C non-negativity
- No NaNs or Infs
- R diagonal minimum values

### `test_A_update.py`
Tests the dynamics matrix A update in isolation:
- KKT optimality conditions
- Dale's law constraint satisfaction
- Comparison with closed-form (unconstrained) solution
- Correct transpose orientation
- Stability with regularization

### `test_C_update.py`
Tests the emission matrix C update in isolation:
- KKT optimality conditions for each row
- Comparison with scipy NNLS
- Non-negativity constraints
- Block-diagonal structure (cell-type constraints)
- Correct matrix orientation (N x D)
- Stability with regularization

### `test_QR_update.py`
Tests covariance updates Q and R:
- Q and R are PSD
- Symmetry
- Minimum eigenvalue enforcement
- Minimum diagonal values for R
- Stability with small T
- Conditioning

### `test_em_convergence.py`
Tests full EM algorithm:
- Log-likelihood monotonicity
- No NaNs throughout EM
- Constraint maintenance
- Parameter recovery from perturbations
- Predictive performance improvement
- Stability with small T
- Batch processing (multiple sequences)
- Convergence detection

### `test_numerical_stability.py`
Tests edge cases and numerical stability:
- Q inverse normalization prevents overflow
- Nearly singular Q and R handling
- All excitatory / all inhibitory cases
- Mixed cell types
- P_A conditioning
- Very small T (T=3)
- Regularization prevents singularity

## Debugging Failed Tests

### If you see NaN log-likelihoods:

1. **Run structural invariants first:**
   ```bash
   pytest tests/test_invariants.py -v -s
   ```
   This will show which M-step update produces NaN.

2. **Check specific update in isolation:**
   ```bash
   # If A update is suspect:
   pytest tests/test_A_update.py::test_A_update_kkt_conditions -v -s
   
   # If C update is suspect:
   pytest tests/test_C_update.py::test_C_update_vs_scipy_nnls -v -s
   
   # If Q/R update is suspect:
   pytest tests/test_QR_update.py -v -s
   ```

3. **Run numerical stability tests:**
   ```bash
   pytest tests/test_numerical_stability.py -v -s
   ```

### Common Issues Detected:

- **R update formula wrong:** `test_C_update.py` will fail with dimension mismatches
- **Q not PSD:** `test_QR_update.py::test_Q_update_is_psd` will fail
- **Dale constraints violated:** `test_invariants.py::test_dale_constraints_on_A` will fail
- **Matrix transpose errors:** `test_A_update.py::test_A_update_orientation` or `test_C_update.py::test_C_update_orientation` will fail
- **Insufficient regularization:** `test_numerical_stability.py::test_nearly_singular_Q` will fail

## Expected Test Output

All tests should PASS. Example successful run:
```
tests/test_invariants.py::test_shapes_after_m_step PASSED
tests/test_invariants.py::test_Q_is_psd_after_m_step PASSED
tests/test_A_update.py::test_A_update_kkt_conditions PASSED
...
=================== 50 passed in 45.2s ===================
```

## Interpreting Failures

### KKT Violations
If `test_A_update_kkt_conditions` or similar fails:
- Check that `P_A` and `q_A` are computed correctly
- Verify bounds (lb, ub) match the constraints
- Ensure BoxCDQP tolerance is appropriate

### Non-PSD Matrices
If Q or R tests fail:
- Check symmetrization: `(M + M.T) / 2`
- Verify regularization: `M + 1e-4 * I`
- Check minimum eigenvalue enforcement

### Constraint Violations
If Dale's law or non-negativity fails:
- Verify constraint bounds in QP setup
- Check column-major vs row-major ordering
- Ensure BoxCDQP converged (check result.state)

## Performance

Expected runtimes on modern CPU:
- Fast tests (invariants, individual updates): ~0.1-0.5s each
- EM convergence tests: ~1-5s each
- Full suite: ~30-60s

For faster iteration during development:
```bash
# Run only fast tests
pytest tests/test_invariants.py tests/test_A_update.py -v

# Skip slow parameter recovery tests
pytest tests/ -v -m "not slow"
```
