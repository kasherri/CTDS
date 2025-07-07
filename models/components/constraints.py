import jax.numpy as jnp

def apply_dale_constraint(A: jnp.ndarray, cell_types: jnp.ndarray) -> jnp.ndarray:
    """
    Enforce Dale's law on the weight matrix A:
      - Excitatory latents (cell_types == 1): column must be ≥ 0
      - Inhibitory latents (cell_types == 2): column must be ≤ 0
      - Unknown/other (cell_types == 0): no constraint applied

    Diagonal elements are left unchanged.

    Args:
        A: (D, D) weight matrix (dynamics or emission weights)
        cell_types: (D,) integer array with values 0 (unknown), 1 (excitatory), 2 (inhibitory)

    Returns:
        A_new: constrained (D, D) array
    """
    D = A.shape[0]
    A_new = A.copy()
    for j in range(D):  # COLUMN index: latent j
        for i in range(D):  # ROW index: who is affected
            if i == j:
                continue  # do not constrain diagonal
            w = A_new[i, j]
            if cell_types[j] == 1:
                # E latent: entire column must be ≥ 0
                A_new = A_new.at[i, j].set(jnp.abs(w))
            elif cell_types[j] == 2:
                # I latent: entire column must be ≤ 0
                A_new = A_new.at[i, j].set(-jnp.abs(w))
            # cell_types[j] == 0: no constraint
    return A_new



def apply_block_sparsity(A: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """
    Enforce block-level sparsity by zeroing out entries where mask == 0.

    Args:
        A: (K, K) weight or connectivity matrix
        mask: (K, K) binary mask (0 = zero-out, 1 = keep)

    Returns:
        Elementwise product A * mask, preserving dtype.
    """
    return A * mask


def project_to_unit_norm(C: jnp.ndarray, axis: int = 1, eps: float = 1e-8) -> jnp.ndarray:
    """
    Normalize rows or columns of C to unit Euclidean norm.

    Args:
        C: array of shape (D, K) or (K, D)
        axis: axis along which to normalize (1 = row-wise, 0 = column-wise)
        eps: small constant to avoid division by zero

    Returns:
        Array of same shape as C, with specified axis normalized.
    """
    norm = jnp.linalg.norm(C, axis=axis, keepdims=True)
    norm = jnp.maximum(norm, eps)
    return C / norm


def clip_matrix(A: jnp.ndarray, min_val: float = -1.0, max_val: float = 1.0) -> jnp.ndarray:
    """
    Clip all entries of A into the interval [min_val, max_val].

    Args:
        A: array of any shape
        min_val: lower bound
        max_val: upper bound

    Returns:
        Clipped array with same shape as A.
    """
    return jnp.clip(A, min_val, max_val)
