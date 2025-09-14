
# Cell-Type Dynamical Systems (CTDS)

A JAX implementation of Cell-Type Dynamical Systems from Jha et al. (2024)
for modeling neural population dynamics with biologically-constrained connectivity.


CTDS models neural population activity using linear dynamical systems that respect cell-type constraints and Dale's law. The model decomposes neural connectivity into cell-type-specific factors while maintaining Dale's law (excitatory neurons have non-negative outgoing weights, inhibitory neurons have non-positive outgoing weights).

### Key Features

- **Cell-type structure**: Models connectivity through cell-type-specific factorization
- **Dale's law constraints**: Enforces biologically plausible sign constraints  
- **JAX-native**: Full JAX compatibility for GPU acceleration and JIT compilation
- **EM training**: Efficient parameter estimation via expectation-maximization
- **Dynamax integration**: Uses Dynamax for exact inference in linear Gaussian models

## Mathematical Framework

The CTDS model describes neural dynamics as:

**Latent dynamics:**

$$x_{t+1} = A x_t +B u_t+ w_t, \quad  w_t ~ N(0, Q)$$


**Observations:**  

$$
y_t = C x_t + v_t,  \quad   v_t ~ N(0, R)
$$

**EI RNN Initialization**

TODO



### Installation
TODO



#### Requirements

- Python 3.9+
- JAX 0.4.0+
- See `pyproject.toml` for complete dependency list



## License

TODO
## Contributing

TODO
