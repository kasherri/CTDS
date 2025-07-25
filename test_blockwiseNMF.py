import pytest
import jax
import numpy as np
import jax.numpy as jnp
from jax import random
from jax.numpy.linalg import norm
from jax import config
config.update("jax_enable_x64", True)

from models.components.intialize import NMF  # Replace with actual module path
from models.components.intialize import NNLS  # Needed to test inner solver

key = random.PRNGKey(0)

def random_nonnegative_matrix(key, shape):
    return jnp.abs(random.uniform(key, shape))

def test_nmf_shapes_and_types():
    key1, key2, key3 = random.split(key, 3)
    J = random_nonnegative_matrix(key1, (10, 8))
    U_init = random_nonnegative_matrix(key2, (10, 3))
    V_init = random_nonnegative_matrix(key3, (8, 3))

    U, V = NMF(U_init, V_init, J, max_iterations=500)

    assert U.shape == (10, 3)
    assert V.shape == (8, 3)
    assert jnp.issubdtype(U.dtype, jnp.floating)
    assert jnp.issubdtype(V.dtype, jnp.floating)

def test_nmf_non_negativity():
    key1, key2, key3 = random.split(key, 3)
    J = random_nonnegative_matrix(key1, (12, 7))
    U_init = jnp.abs(random.exponential(key2, (12, 4)))
    V_init =  jnp.abs(random.exponential(key3, (7, 4)))

    U, V = NMF(U_init, V_init, J, max_iterations=300)

    assert jnp.all(U >= -1e-6)  # allow small numerical error
    assert jnp.all(V >= -1e-6)

def test_nmf_reconstruction_error_decreases():
    key1, key2, key3 = random.split(key, 3)
    J = random_nonnegative_matrix(key1, (20, 10))
    U_init =jnp.abs(random.exponential(key2, (20, 5)))
    V_init = jnp.abs(random.exponential(key3, (10, 5)))

    # Initial error
    error_init = norm(J - U_init @ V_init.T, ord='fro')

    U, V = NMF(U_init, V_init, J, max_iterations=1000)
    error_final = norm(J - U @ V.T, ord='fro')

    assert error_final < error_init  

def test_nmf_converges_small_problem():
    J = jnp.array([[1.0, 0.5], [0.2, 0.8]])
    U_init = jnp.array([[0.9, 0.1], [0.1, 0.9]])
    V_init = jnp.array([[0.5, 0.2], [0.3, 0.7]])

    U, V = NMF(U_init, V_init, J, max_iterations=100)

    recon = U @ V.T
    rel_error = norm(J - recon, 'fro') / norm(J, 'fro')
    assert rel_error < 0.6

def test_nmf_runs_under_jit():
    key1, key2, key3 = random.split(key, 3)
    J = random_nonnegative_matrix(key1, (6, 6))
    U_init = random_nonnegative_matrix(key2, (6, 2))
    V_init = random_nonnegative_matrix(key3, (6, 2))

    jitted_NMF = jax.jit(NMF)
    U, V = jitted_NMF(U_init, V_init, J)

    assert U.shape == (6, 2)
    assert V.shape == (6, 2)



def test_nnls_output_nonnegative():
    key = random.PRNGKey(0)
    D = 5
    Q = jnp.eye(D)
    c = -random.uniform(key, shape=(D,), minval=0.0, maxval=1.0)

    x = NNLS(Q, c)
    assert jnp.all(x >= 0.0), "NNLS output contains negative values."


def test_nnls_reconstruction_error_small():
    # Ground truth x ≥ 0
    x_true = jnp.array([1.0, 0.0, 2.0])
    A = jnp.array([[3.0, 0.0, 1.0],
                   [0.0, 2.0, 0.0],
                   [1.0, 0.0, 1.0]])
    b = A @ x_true
    Q = A.T @ A
    c = -A.T @ b

    x_est = NNLS(Q, c)
    b_est = A @ x_est

    rel_error = norm(b - b_est) / norm(b)
    assert rel_error < 1e-3, f"Reconstruction error too high: {rel_error}"


def test_nnls_output_shape():
    D = 10
    Q = jnp.eye(D)
    c = -jnp.arange(D, dtype=jnp.float32)
    x = NNLS(Q, c)
    assert x.shape == (D,), "Output shape mismatch."


@pytest.mark.parametrize("D", [5, 10, 50])
def test_nnls_vs_scipy_nnls(D):
    from scipy.optimize import nnls

    np.random.seed(0)
    A = np.abs(np.random.randn(D, D))
    x_true = np.abs(np.random.randn(D))
    b = A @ x_true
    Q = A.T @ A
    c = -A.T @ b

    x_est_jax = NNLS(jnp.array(Q), jnp.array(c))
    x_est_scipy, _ = nnls(A, b)

    error_jax = norm(A @ x_est_jax - b) / norm(b)
    error_scipy = norm(A @ x_est_scipy - b) / norm(b)

    assert error_jax <= error_scipy * 1.5, "JAX NNLS underperforming relative to SciPy."


def test_nnls_benchmark(benchmark):
    D = 50
    key = random.PRNGKey(123)
    Q = jnp.eye(D)
    c = -random.uniform(key, shape=(D,))

    def run():
        return NNLS(Q, c)

    result = benchmark(run)
    assert jnp.all(result >= 0.0)
