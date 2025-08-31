#!/usr/bin/env python3
"""
CTDS Quick Start Demo
=====================

This script demonstrates basic CTDS functionality after installation.
Run this to verify your installation and see CTDS in action.

Usage:
    python quickstart.py
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import ctds

def main():
    print("CTDS Quick Start Demo")
    print("=" * 50)
    
    # Set up random key
    key = jr.PRNGKey(42)
    
    # Generate synthetic neural data
    print(" Generating synthetic neural data...")
    
    # Use the simulation utilities to create test data
    from ctds.simulation_utilis import generate_synthetic_data
    
    # Parameters for a small example
    num_samples = 1
    num_timesteps = 50
    state_dim = 6
    emission_dim = 15
    cell_types = 2
    
    # Generate data
    states, emissions, model, params = generate_synthetic_data(
        num_samples=num_samples,
        num_timesteps=num_timesteps,
        state_dim=state_dim,
        emission_dim=emission_dim,
        cell_types=cell_types,
        key=key
    )
    
    print(f"✅ Generated data: {emissions.shape[0]} timesteps, {emissions.shape[1]} neurons")
    print(f"✅ Model: {state_dim}D latent space, {cell_types} cell types")
    
    # Test basic inference
    print("\n Testing inference...")
    
    # Run smoother
    smoothed_means, smoothed_covariances = model.smoother(params, emissions, None)
    
    print(f" Posterior inference complete")
    print(f"   - Smoothed means shape: {smoothed_means.shape}")
    print(f"   - Smoothed covariances shape: {smoothed_covariances.shape}")
    
    # Test forecasting
    print("\n Testing forecasting...")
    
    # Use first half to forecast second half
    T_obs = num_timesteps // 2
    history = emissions[:T_obs]
    forecast_steps = num_timesteps - T_obs
    
    # Generate forecast
    forecasts = model.forecast(
        params,
        history,
        num_steps=forecast_steps,
        inputs=None,
        key=None  # Deterministic forecast
    )
    
    print(f"✅ Forecast complete")
    print(f"   - History length: {T_obs} timesteps")
    print(f"   - Forecast length: {forecast_steps} timesteps")
    print(f"   - Forecast shape: {forecasts.shape}")
    
    # Test model constraints
    print("\n🔒 Testing Dale's law constraints...")
    
    constraints = model.constraints
    print(f"✅ Cell types: {len(constraints.cell_types)}")
    print(f"✅ Cell signs: {constraints.cell_sign}")
    print(f"✅ Cell type mask length: {len(constraints.cell_type_mask)}")
    
    # Verify sign constraints are valid
    valid_signs = jnp.all(jnp.isin(constraints.cell_sign, jnp.array([-1, 1])))
    print(f"✅ Valid Dale signs: {valid_signs}")
    
    print("\n🎉 Quick start demo completed successfully!")
    print("\n Next steps:")
    print("   - Explore examples in Jupyter notebooks")
    print("   - Read the documentation: README.MD")
    print("   - Run full test suite: python -m pytest tests/")
    print("   - Check out the API reference in the docs")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        print("\n🔧 Troubleshooting:")
        print("   - Make sure you activated the virtual environment")
        print("   - Try running: python -c 'import ctds; print(\"CTDS works!\")'")
        print("   - Check installation: ./setup.sh")
        raise
