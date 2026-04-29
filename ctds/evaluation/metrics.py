import jax
from jax import jit
import jax.numpy as jnp
from jax import random, vmap, jit
from functools import partial

# Enable 64-bit precision for better numerical stability
jax.config.update("jax_enable_x64", True)

def generate_well_conditioned_nonnegative_svd(key, m, n, condition_number=10.0):
    """
    Generate well-conditioned nonnegative matrix using SVD approach
    """
    key1, key2, key3 = random.split(key, 3)
    
    rank = min(m, n)
    
    # Generate orthogonal matrices via QR decomposition
    U_raw = random.normal(key1, (m, rank))
    U, _ = jnp.linalg.qr(U_raw)
    
    V_raw = random.normal(key2, (n, rank))  
    V, _ = jnp.linalg.qr(V_raw)
    
    # Control singular values for conditioning
    s_max = 1.0
    s_min = s_max / condition_number
    singular_values = jnp.linspace(s_max, s_min, rank)
    
    # Construct matrix
    if m >= n:
        A = U @ jnp.diag(singular_values) @ V.T
    else:
        A = U @ jnp.diag(singular_values) @ V[:rank, :].T
    
    # Make nonnegative by shifting and scaling
    A_min = jnp.min(A)
    A_nonneg = A - A_min + 0.1
    
    return A_nonneg

 
def generate_via_exponential(key, m, n, condition_number=10.0, scale=0.5):
    """
    Generate nonnegative matrix via matrix exponential
    """
    A = random.normal(key, (m, n))
    
    # Control conditioning through the base matrix
    # For rectangular matrices, work with the Gram matrix
    if m >= n:
        AtA = A.T @ A
        eigenvals = jnp.linalg.eigvals(AtA)
        max_eig = jnp.max(jnp.real(eigenvals))
        min_eig = jnp.min(jnp.real(eigenvals))
        
        # Add regularization if needed
        reg = jnp.maximum(0.0, max_eig / condition_number - min_eig)
        reg_matrix = reg * jnp.eye(n) * 0.1
        A_reg = A @ jnp.linalg.cholesky(jnp.eye(n) + reg_matrix)
    else:
        AAt = A @ A.T  
        eigenvals = jnp.linalg.eigvals(AAt)
        max_eig = jnp.max(jnp.real(eigenvals))
        min_eig = jnp.min(jnp.real(eigenvals))
        
        reg = jnp.maximum(0.0, max_eig / condition_number - min_eig)
        reg_matrix = reg * jnp.eye(m) * 0.1
        A_reg = jnp.linalg.cholesky(jnp.eye(m) + reg_matrix) @ A
    
    # Apply exponential for guaranteed non-negativity
    return jnp.exp(A_reg * scale)

##@partial(jit, static_argnums=(1, 2, 3))
def generate_neural_like_matrix(key, n_neurons, n_timepoints, n_factors=None, 
                               condition_number=5.0, sparsity=0.1):
    """
    Generate neural data-like matrix with realistic structure
    """
    if n_factors is None:
        n_factors = min(n_neurons, n_timepoints) // 3
        
    key1, key2, key3, key4 = random.split(key, 4)
    
    # Generate factor loadings (neurons x factors) 
    W = random.exponential(key1, (n_neurons, n_factors))
    
    # Generate factor time courses (factors x time)
    H_base = random.exponential(key2, (n_factors, n_timepoints)) * 0.5
    
    # Add temporal correlations using convolution
    def smooth_factor(h):
        # Gaussian kernel for smoothing
        kernel_size = 11
        x = jnp.arange(-(kernel_size//2), kernel_size//2 + 1)
        kernel = jnp.exp(-x**2 / 8.0)
        kernel = kernel / jnp.sum(kernel)
        
        # Pad and convolve
        h_padded = jnp.pad(h, kernel_size//2, mode='edge')
        return jnp.convolve(h_padded, kernel, mode='valid')
    
    H = vmap(smooth_factor)(H_base)
    
    # Base reconstruction  
    X = W @ H
    
    # Add sparse noise
    noise_mask = random.bernoulli(key3, sparsity, (n_neurons, n_timepoints))
    noise = random.exponential(key4, (n_neurons, n_timepoints)) * 0.1
    X = X + noise_mask * noise
    
    # Ensure good conditioning via SVD regularization
    U, s, Vt = jnp.linalg.svd(X, full_matrices=False)
    s_ratio = s[0] / s[-1] 
    
    # Regularize singular values if condition number is too high
    cutoff = s[0] / condition_number
    s_reg = jnp.maximum(s, cutoff)
    X_reg = U @ jnp.diag(s_reg) @ Vt * 0.05

    # Ensure strict positivity
    return jnp.maximum(X_reg, 0.01)

#@partial(jit, static_argnums=(1, 2))
def generate_gamma_based(key, m, n, condition_number=10.0):
    """
    Generate using Gamma distributions with controlled conditioning
    """
    key1, key2 = random.split(key)
    
    # Varying shape parameters for diversity
    rank = min(m, n)
    shape_params = jnp.linspace(2.0, 10.0, rank)
    
    # Generate base matrix
    A = jnp.zeros((m, n))
    
    def generate_column(i, A):
        col_key = random.fold_in(key1, i)
        shape = shape_params[i % len(shape_params)]
        scale = 1.0 / shape
        col = random.gamma(col_key, shape, (m,)) * scale
        return A.at[:, i].set(col)
    
    A = jax.lax.fori_loop(0, n, generate_column, A)
    
    # Post-process for exact conditioning control
    U, s, Vt = jnp.linalg.svd(A, full_matrices=False)
    s_new = jnp.linspace(1.0, 1.0/condition_number, len(s))
    A_conditioned = U @ jnp.diag(s_new) @ Vt * 0.05

    # Ensure non-negativity
    return jnp.maximum(A_conditioned, 0.01)

#@jit
def validate_conditioning(A):
    """
    Compute condition number for rectangular matrix
    """

    return jnp.linalg.cond(A)



# Example usage and testing
def demo_generation():
    """Demonstrate all methods"""
    key = random.PRNGKey(42)
    m, n = 100, 80
    target_condition = 5.0
    
    print("Generating well-conditioned nonnegative matrices...")
    
    # Method 1: SVD-based
    key1, key = random.split(key)
    A1 = generate_well_conditioned_nonnegative_svd(key1, m, n, target_condition)
    print(A1)
    cond1 = validate_conditioning(A1)
    print(f"SVD method - Condition: {cond1:.2f}, Shape: {A1.shape}, Range: [{A1.min():.3f}, {A1.max():.3f}]")
    
    # Method 2: Exponential 
    key2, key = random.split(key)
    A2 = generate_via_exponential(key2, m, n, target_condition)
    cond2 = validate_conditioning(A2)
    print(f"Exponential method - Condition: {cond2:.2f}, Shape: {A2.shape}, Range: [{A2.min():.3f}, {A2.max():.3f}]")
    
    # Method 3: Neural-like
    key3, key = random.split(key)
    A3 = generate_neural_like_matrix(key3, m, n, condition_number=target_condition)
    cond3 = validate_conditioning(A3)
    print(A3)
    print(f"Neural-like method - Condition: {cond3:.2f}, Shape: {A3.shape}, Range: [{A3.min():.3f}, {A3.max():.3f}]")
    
    # Method 4: Gamma-based
    key4, key = random.split(key) 
    A4 = generate_gamma_based(key4, m, n, target_condition)
    cond4 = validate_conditioning(A4)
    print(f"Gamma method - Condition: {cond4:.2f}, Shape: {A4.shape}, Range: [{A4.min():.3f}, {A4.max():.3f}]")

    
    return A1, A2, A3, A4

demo_generation()