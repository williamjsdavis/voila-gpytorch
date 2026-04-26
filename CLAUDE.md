# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

Scratch / planning stage. There is **no Python code in the repo yet** — only `README.md` and gitignored reference material. Treat tasks here as greenfield design work, not maintenance of an existing codebase. There are no build, lint, or test commands to run yet; if you add them, document them here.

## What this project is

A Python / [GPyTorch](https://gpytorch.ai/) reimplementation of [`voila`](https://github.com/citiususc/voila) (García et al., R package). The method does non-parametric estimation of one- and multi-dimensional Langevin SDEs

```
dX_t = f(X_t) dt + g(X_t) dW_t
```

from densely-observed time series, by modelling drift `f` and diffusion `g` as Gaussian processes and learning sparse approximations via variational inference with inducing variables (García et al., *Phys. Rev. E* 96, 022104, 2017).

The port's design goals (per `README.md`): GPU acceleration, autograd-driven variational lower bound, modular GPyTorch kernels, and reproduction of the published examples (Ornstein–Uhlenbeck, Dansgaard–Oeschger events, multivariate cases).

## Reference material on disk (gitignored, kept locally)

The `.gitignore` excludes `voila/` and `paper/` from version control, but they are present in the working tree and are the primary specification for what to build:

- `voila/` — original R package source. Authoritative reference for the algorithm. Key files:
  - `voila/R/sde_vi.R` — top-level variational-inference driver (`sde_vi`, the function the README example calls).
  - `voila/src/sde_variational_inferencer.{h,cpp}` — C++/Armadillo implementation of the VI updates and lower-bound computation. This is where the math actually lives.
  - `voila/R/kernel.R` + `voila/src/kernel.{h,cpp}`, `common_kernels.{h,cpp}`, `kernel_modules.cpp` — kernel definitions exposed via Rcpp modules (`rq_kernel`, `exp_const_kernel`, etc.). The diffusion uses a log-normal GP, with a constant + exponential kernel, to enforce positivity — this is non-obvious and easy to miss when porting.
  - `voila/R/select_diffusion_parameters.R` — heuristic for setting the diffusion kernel's amplitude and the log-normal mean `v` from the data; used for initialization in the README example.
  - `voila/R/simulate_sde.R` — SDE simulation utility (delegates to the `yuima` R package). Useful for generating the OU validation data.
  - `voila/R/sde_prediction.R` — posterior prediction for drift/diffusion (note the `log = TRUE` flag for diffusion).
  - `voila/vignettes/` and `voila/demo/` — runnable examples (DO events, multivariate). These are the validation targets.
  - `voila/README.md` — end-to-end OU example with expected lower-bound trajectory; useful as a smoke-test target.
- `paper/` — the published paper (`1704.04375v2.pdf`) and its LaTeX source (`Report.tex`, figures). Definitive source for notation and the variational lower-bound derivation.

When implementing a piece of the algorithm, prefer cross-checking the C++ source (`sde_variational_inferencer.cpp`) against the paper rather than relying on the R wrappers alone — the wrappers mostly marshal arguments.

## Architectural notes for the port

A few decisions are implied by the goals and the reference implementation; surface them explicitly when designing new code:

- **Drift vs. diffusion are not symmetric.** Drift is a plain GP; diffusion is a log-normal GP (GP on `log g²`) with a constant-plus-exponential kernel. Any abstraction over "the two GPs" must keep this asymmetry first-class.
- **Inducing-point variational inference, not exact GPs.** Pseudo-input locations and variational parameters are *both* learned. GPyTorch's `VariationalStrategy` / `InducingPointKernel` machinery is the natural target, but the lower bound in the paper is specific to the SDE likelihood (Euler–Maruyama-style increments) and will need a custom marginal log-likelihood, not GPyTorch's stock ELBO.
- **Validation, not just running.** A change is "done" when it reproduces an R-package result on one of the published examples (OU lower bound trajectory, DO bistable drift, multivariate fields), not merely when it runs without error.
