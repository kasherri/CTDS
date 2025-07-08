import jax.numpy as jnp
import dynamax
from dynamax.linear_gaussian_ssm import LinearGaussianSSM, ParamsLGSSM
from dynamax.linear_gaussian_ssm.inference import filter, smoother
from models.components.ctds_dynamics import CTDSDynamics
from models.components.ctds_emissions import CTDSEmissions


def em(model: LinearGaussianSSM, 
              params: ParamsLGSSM, 
              emissions: jnp.ndarray, 
              dynamics: CTDSDynamics, 
              emissions_module: CTDSEmissions, 
              num_iters: int = 10):
    """
    custom EM loop for CTDS 
        1. E step Kalman Smoothing
        2. M step QP for A,Q
    

    Params:
        model: Dynamax LGSSM model instance
        params: current model parameters (ParamsLGSSM)
        emissions: (T, N) array of observed data
        dynamics: instance of CTDSDynamics for constrained M-step
        emissions_module: instance of CTDSEmissions for constrained M-step
        num_iters: number of EM iterations

    returns:
        dict{Updated params, final smoothed posterior, list of log-likelihoods}
    """
    lls = []
    for i in range(num_iters):
        # ------------------ E STEP ------------------
        smoothed = smoother(model, params, emissions) # returns PosteriorGSSMSmoothed() object which is tuple-like object?
        Ex = smoothed.smoothed_means #mean of latent at time t
        Exx = smoothed.smoothed_covariances + jnp.einsum("ti,tj->tij", Ex, Ex) #Cov[z_t] + Ex ⊗ Ex	
        Exnx = smoothed.smoothed_cross_covariances + jnp.einsum("ti,tj->tij", Ex[1:], Ex[:-1])  #Cov[z_t, z_{t-1}] + Ex[1:] ⊗ Ex[:-1]	
        posterior = {
            "Ex": Ex ,                  # (T, D)
            "Exx": Exx,                    # (T, D, D)
            "Exnx": Exnx                    # (T-1, D, D)
        }

        # ------------------ M STEP ------------------
        A, Q = dynamics.m_step(posterior)           # (D, D), (D, D)
        C, R = emissions_module.m_step(posterior, emissions)  # (N, D), (N, N)

        assert A.shape == params.dynamics.A.shape
        assert Q.shape == params.dynamics.Q.shape
        assert C.shape == params.emissions.C.shape
        assert R.shape == params.emissions.R.shape

        params = params._replace(
            dynamics=params.dynamics._replace(A=A, Q=Q),
            emissions=params.emissions._replace(C=C, R=R)
        )

        # eval
        ll = model.marginal_loglik(params, emissions)
        lls.append(ll)

        print(f"[EM] Iter {i+1}/{num_iters}  |  Log Likelihood: {ll:.2f}")

    #finalize
    model.set_params(params)
    return {
        "params": params,
        "posterior": smoothed,
        "log_likelihoods": lls
        }

   

