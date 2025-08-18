# CTDS Single-Region Implementation — Status Overview

**Current status**

* ✅ Implemented a single-region CTDS model with EM.
* ✅ Simulation pipeline: generate synthetic data from known parameters, fit with EM.
* ✅ EM converges reliably on well-conditioned synthetic data (condition number < \~20).
* ✅ GPU JIT works; code runs jittable on GPU.

**Limitations / Open issues**

* ❌ CPU JIT blocked due to XLA layout constraints for dot products.
* ❌ Parameter recovery: haven’t yet implemented blockwise permutation + scaling alignment (needed to fairly compare recovered vs true parameters).
* ⚠️ Ill-conditioning: jaxopt CDQP solver becomes unstable when fitting with higher condition numbers.

**Metrics computed so far**

* Log-likelihood traces during EM.
* Frobenius errors between fitted and true parameters (not yet adjusted for non-identifiability).


