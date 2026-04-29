# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-01-01

### Added
- EM algorithm with Dale's Law and cell-type block-structure constraints on the dynamics matrix (`A`) and emission matrix (`C`)
- JAX/Dynamax inference backend (`DynamaxLGSSMBackend`) with Kalman filtering and smoothing
- NMF-based initialization (`fa_initialize_ctds`) and PCA-based initialization (`pca_initialize_ctds`) in `ctds.initialization`
- `ParamsCTDS` dataclass hierarchy for model parameters with typed constraint objects
- Simulation utilities (`ctds.simulation_utils`) for generating synthetic data and evaluating recovery
- Evaluation metrics module (`ctds.evaluation.metrics`)
- Installable package structure with `pyproject.toml` and hatchling build system
- Full test suite in `tests/`
