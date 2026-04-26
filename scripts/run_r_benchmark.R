#!/usr/bin/env Rscript
# Time voila::sde_vi on the bundled OU realization and dump JSON.
#
# This is the R-side counterpart to scripts/run_device_benchmark.py. It is
# DEFERRED until the device-threading work lands; do not execute as part of the
# Phase 1-3 verification.
#
# One-time install (Linux; on Apple Silicon use Homebrew's `brew install r`):
#     sudo apt install r-base r-base-dev libarmadillo-dev
#     R -e 'install.packages(c("Rcpp", "RcppArmadillo", "yuima", "jsonlite"))'
#     R CMD INSTALL voila/                       # build the vendored package
#
# Run:
#     Rscript scripts/run_r_benchmark.R --output bench_r.json
#
# Output schema mirrors scripts/run_device_benchmark.py so the comparison
# notebook can ingest both with a single loader.
#
# Scope: ou_small + ou_medium only. Lorenz / multivariate are skipped because
# voila R doesn't expose `MultiSDEVI` cleanly — that's a research extension we
# added in voila-gp.

suppressPackageStartupMessages({
  library(voila)
  library(jsonlite)
})

REPS <- 3

args <- commandArgs(trailingOnly = TRUE)
output_path <- NULL
if (length(args) >= 2 && args[1] == "--output") {
  output_path <- args[2]
}

# ---- load fixture ----------------------------------------------------------
# The OU realization is the same one used by tests/data/ornstein.npz, taken
# from voila/data/ornstein.rda.
data("ornstein", package = "voila")
x_base <- ornstein$x
h <- ornstein$samplingPeriod  # 0.001

# ---- helpers ---------------------------------------------------------------
median_time <- function(times) {
  times <- as.numeric(times)
  list(wall_clock_s = times, median_s = median(times))
}

run_ou_small <- function() {
  times <- numeric(REPS)
  for (i in seq_len(REPS)) {
    set.seed(0)
    drift_kernel <- rationalQuadraticKernel(amplitude = 5.0,
                                            alpha = 1.0, lengthScale = 1.5,
                                            epsilon = 1e-5)
    diff_params <- selectDiffusionParameters(x_base, samplingPeriod = h,
                                             priorOnSd = 5.0)
    diff_kernel <- exponentialConstantKernel(maxAmplitude = diff_params$kernelAmplitude,
                                             expAmplitude = diff_params$kernelAmplitude * 1e-3,
                                             lengthScales = c(1.5),
                                             epsilon = 1e-5)
    inducing <- matrix(seq(min(x_base), max(x_base), length.out = 10), ncol = 1)
    t0 <- Sys.time()
    fit <- sde_vi(
      targetIndex = 1, x = x_base, samplingPeriod = h,
      inducingPoints = inducing, v = diff_params$v,
      driftKernel = drift_kernel, diffKernel = diff_kernel,
      maxIterations = 5, relTol = 1e-6
    )
    times[i] <- as.numeric(Sys.time() - t0, units = "secs")
    final_L <- tail(fit$lowerBoundHistory, 1)
    iters <- length(fit$lowerBoundHistory) - 1L
  }
  c(list(problem = "ou_small", final_L = final_L, iterations = iters),
    median_time(times))
}

run_ou_medium <- function() {
  x_med <- as.matrix(rep(as.numeric(x_base), 4))   # n ≈ 80004
  times <- numeric(REPS)
  for (i in seq_len(REPS)) {
    set.seed(0)
    drift_kernel <- rationalQuadraticKernel(amplitude = 5.0,
                                            alpha = 1.0, lengthScale = 1.5,
                                            epsilon = 1e-5)
    diff_params <- selectDiffusionParameters(x_med, samplingPeriod = h,
                                             priorOnSd = 5.0)
    diff_kernel <- exponentialConstantKernel(maxAmplitude = diff_params$kernelAmplitude,
                                             expAmplitude = diff_params$kernelAmplitude * 1e-3,
                                             lengthScales = c(1.5),
                                             epsilon = 1e-5)
    inducing <- matrix(seq(min(x_med), max(x_med), length.out = 30), ncol = 1)
    t0 <- Sys.time()
    fit <- sde_vi(
      targetIndex = 1, x = x_med, samplingPeriod = h,
      inducingPoints = inducing, v = diff_params$v,
      driftKernel = drift_kernel, diffKernel = diff_kernel,
      maxIterations = 5, relTol = 1e-6
    )
    times[i] <- as.numeric(Sys.time() - t0, units = "secs")
    final_L <- tail(fit$lowerBoundHistory, 1)
    iters <- length(fit$lowerBoundHistory) - 1L
  }
  c(list(problem = "ou_medium", final_L = final_L, iterations = iters),
    median_time(times))
}

# ---- run -------------------------------------------------------------------
cat("Running R benchmark (REPS =", REPS, ")\n")
results <- list()

cat("\n--- ou_small ---\n")
res <- run_ou_small()
cat("  wall_clock_s =", paste(round(res$wall_clock_s, 3), collapse = ", "), "\n")
cat("  median_s     =", round(res$median_s, 3), "\n")
cat("  final_L      =", res$final_L, "\n")
results <- c(results, list(res))

cat("\n--- ou_medium ---\n")
res <- run_ou_medium()
cat("  wall_clock_s =", paste(round(res$wall_clock_s, 3), collapse = ", "), "\n")
cat("  median_s     =", round(res$median_s, 3), "\n")
cat("  final_L      =", res$final_L, "\n")
results <- c(results, list(res))

payload <- list(
  device = "r-cpu",
  dtype = "double",
  machine = list(
    hostname = Sys.info()[["nodename"]],
    platform = paste(Sys.info()[c("sysname", "release", "machine")], collapse = " "),
    r_version = paste(R.version$major, R.version$minor, sep = ".")
  ),
  results = results
)

payload_json <- toJSON(payload, auto_unbox = TRUE, pretty = TRUE)
cat("\n", payload_json, "\n", sep = "")

if (!is.null(output_path)) {
  writeLines(payload_json, output_path)
  cat("\nWrote", output_path, "\n")
}
