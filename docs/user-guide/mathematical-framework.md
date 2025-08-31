# Mathematical Framework

## Overview

Cell-Type Dynamical Systems (CTDS) model neural population dynamics using linear state-space models with biological constraints. This page provides the mathematical foundation for understanding how CTDS works.

## State-Space Formulation

### Core Equations

The CTDS model consists of two main equations:

**Latent State Dynamics:**
$$x_{t+1} = A x_t + B u_t + w_t$$

where:
- $x_t \in \mathbb{R}^D$ is the latent state at time $t$
- $A \in \mathbb{R}^{D \times D}$ is the dynamics matrix (recurrent connectivity)
- $B \in \mathbb{R}^{D \times U}$ is the input matrix (currently not implemented)
- $u_t \in \mathbb{R}^U$ are exogenous inputs
- $w_t \sim \mathcal{N}(0, Q)$ is process noise with covariance $Q \in \mathbb{R}^{D \times D}$

**Observation Model:**
$$y_t = C x_t + v_t$$

where:
- $y_t \in \mathbb{R}^N$ are the observed neural activities
- $C \in \mathbb{R}^{N \times D}$ is the emission matrix (observation model)
- $v_t \sim \mathcal{N}(0, R)$ is observation noise with covariance $R \in \mathbb{R}^{N \times N}$

### Initial Conditions

The initial state follows a Gaussian distribution:
$$x_1 \sim \mathcal{N}(\mu_1, \Sigma_1)$$

## Cell-Type Structure

### Connectivity Factorization

The key innovation of CTDS is factorizing the dynamics matrix to respect cell-type structure:

$$A = U V^T$$

where:
- $U \in \mathbb{R}^{D \times K}$ contains cell-type factors
- $V \in \mathbb{R}^{D \times K}$ contains connectivity weights
- $K$ is the number of cell types (typically $K=2$ for excitatory/inhibitory)

### Dale's Law Constraints

Dale's law states that neurons release the same neurotransmitter at all synapses. In CTDS, this translates to sign constraints:

**For dynamics matrix $A$:**
- $A_{ij} \geq 0$ if source neuron $j$ is excitatory
- $A_{ij} \leq 0$ if source neuron $j$ is inhibitory

**For emission matrix $C$:**
- $C_{ij} \geq 0$ (non-negative emission weights preserve Dale signs)

### Cell-Type Organization

Neurons are organized into cell types with dimensions:
- $D_E$: number of excitatory neurons in latent space
- $D_I$: number of inhibitory neurons in latent space
- Total: $D = D_E + D_I$

## Parameter Estimation

### EM Algorithm

CTDS uses the Expectation-Maximization (EM) algorithm for parameter estimation:

**E-step:** Compute posterior statistics using the Kalman smoother
$$p(x_{1:T} | y_{1:T}, \theta^{(k)})$$

**M-step:** Update parameters using constrained optimization
$$\theta^{(k+1)} = \arg\max_\theta \mathbb{E}[\log p(x_{1:T}, y_{1:T} | \theta)]$$

### Sufficient Statistics

The E-step computes sufficient statistics:
- $\hat{x}_t = \mathbb{E}[x_t | y_{1:T}]$: posterior means
- $\hat{P}_t = \text{Cov}[x_t | y_{1:T}]$: posterior covariances  
- $\hat{P}_{t,t-1} = \text{Cov}[x_t, x_{t-1} | y_{1:T}]$: lag-1 cross-covariances

### Constrained M-step Updates

**Dynamics Matrix $A$:**
Solved via constrained quadratic programming:
$$\min_A \|A\|_F^2 \text{ s.t. Dale's law constraints}$$

**Emission Matrix $C$:**
Block-wise non-negative least squares (NNLS):
$$\min_C \|Y - CX\|_F^2 \text{ s.t. } C \geq 0$$

**Covariances $Q, R$:**
Residual covariance estimation:
$$Q = \frac{1}{T-1} \sum_{t=2}^T \mathbb{E}[(x_t - Ax_{t-1})(x_t - Ax_{t-1})^T]$$
$$R = \frac{1}{T} \sum_{t=1}^T \mathbb{E}[(y_t - Cx_t)(y_t - Cx_t)^T]$$

## Inference

### Kalman Filtering/Smoothing

CTDS uses exact Bayesian inference via Kalman filtering:

**Forward Pass (Filtering):**
$$p(x_t | y_{1:t}) = \mathcal{N}(x_t | \mu_{t|t}, \Sigma_{t|t})$$

**Backward Pass (Smoothing):**
$$p(x_t | y_{1:T}) = \mathcal{N}(x_t | \mu_{t|T}, \Sigma_{t|T})$$

### Likelihood Computation

The marginal likelihood is computed as:
$$\log p(y_{1:T}) = \sum_{t=1}^T \log p(y_t | y_{1:t-1})$$

This provides a principled way to compare models and assess fit quality.

## Biological Interpretation

### Network Connectivity

The learned dynamics matrix $A$ represents effective connectivity between neural populations. The factorization $A = UV^T$ allows interpretation of:

- **$U$ matrix**: Cell-type mixing coefficients
- **$V$ matrix**: Dale-constrained connectivity patterns

### Neural Subspaces

The latent states $x_t$ represent activity in a low-dimensional neural subspace that captures the essential dynamics while respecting biological constraints.

### Population Dynamics

The model captures both:
- **Fast timescales**: Observation noise and rapid fluctuations
- **Slow timescales**: Latent dynamics and connectivity structure

## Extensions and Limitations

### Current Limitations

- **Linear dynamics**: No support for nonlinear dynamics
- **Gaussian noise**: Assumes Gaussian process and observation noise
- **Single region**: No multi-region connectivity
- **Static parameters**: Time-invariant dynamics and emission matrices

### Future Extensions

- **Nonlinear dynamics**: Neural network components for $f(x_t)$
- **Multi-region models**: Cross-region connectivity patterns
- **Time-varying parameters**: Adaptive dynamics and emissions
- **Non-Gaussian noise**: Robust noise models
