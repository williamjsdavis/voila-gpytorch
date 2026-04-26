# voila-gpytorch

A Python/[GPyTorch](https://gpytorch.ai/) reimplementation of [`voila`](https://github.com/citiususc/voila), an R package for non-parametric estimation of Langevin equations (stochastic differential equations) from densely-observed time series.

## Background

The original `voila` package estimates the drift and diffusion terms of a Langevin equation

$$\mathrm{d}X_t = f(X_t)\,\mathrm{d}t + g(X_t)\,\mathrm{d}W_t$$

by modelling $f$ and $g$ as Gaussian processes and learning sparse approximations via variational inference with inducing variables. The method is described in:

> García, C.A., Otero, A., Félix, P., Presedo, J. & Márquez D.G. **Non-parametric Estimation of Stochastic Differential Equations with Sparse Gaussian Processes.** *Phys. Rev. E 96 (2017), 022104.* [Article](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.96.022104) · [Preprint](https://arxiv.org/abs/1704.04375)

## Goals

- Reproduce `voila`'s SDE inference results in Python on top of [GPyTorch](https://gpytorch.ai/) and [PyTorch](https://pytorch.org/).
- Modernize the implementation: GPU acceleration, autodiff for the variational lower bound, modular kernels, and integration with the broader PyTorch ecosystem.
- Validate against the original R implementation on the published examples (Ornstein–Uhlenbeck, Dansgaard–Oeschger events, multivariate cases).

## Status

Early scratch / planning stage.

## Reference implementation

The original R package by García et al. lives at [citiususc/voila](https://github.com/citiususc/voila) and is licensed under GPL v3.
