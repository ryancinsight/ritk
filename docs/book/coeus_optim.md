# Coeus Nonlinear Least-Squares Solver

`coeus-optim` owns the Levenberg-Marquardt (damped Gauss-Newton) solver used
by the nonlinear diffusion models in
[ADR 0036 decision 2](https://github.com/ryancinsight/atlas/blob/main/docs/adr/0036-neuroimaging-and-mr-ownership.md).
DKI, NODDI, and IVIM fit their parameters through this solver; RITK adds no
local nonlinear optimiser.

## Why a separate solver contract?

The gradient-descent optimisers in `coeus-optim` (`SGD`, `Adam`, `AdamW`,
`RMSProp`, `AdaGrad`) step on gradients already accumulated into parameters,
which suits network training. A least-squares solver instead re-evaluates the
model at trial points to decide whether a step is acceptable, and exploits
the Gauss-Newton curvature approximation \\(J^{\\!T}\\!J \\approx H\\) that a
bare gradient does not expose. The two are different contracts, not two
spellings of one.

## LeastSquaresProblem trait

`LeastSquaresProblem<T>` is the contract between a model and the solver. It
requires:

| Method | Purpose |
|---|---|
| `residual_count() -> usize` | Number of residual components \\(m\\) |
| `parameter_count() -> usize` | Number of free parameters \\(n\\) |
| `residuals(&self, &[T], &mut [T])` | Evaluate \\(r(p)\\) — \\(m\\) values |
| `jacobian(&self, &[T], &mut [T])` | Evaluate \\(J(p) = \\partial r/\\partial p\\) — row-major, \\(m \\times n\\) |

The Jacobian is row-major: entry \\((i, j)\\) at index \\(i \\cdot n + j\\) is
\\(\\partial r_i / \\partial p_j\\).

The solver requires \\(m \\ge n\\) — fewer residuals than parameters makes
\\(J^{\\!T}\\!J\\) singular by construction, and damping would mask that
rather than solve it.

### Domain errors

`ProblemError::Domain` signals that the model is undefined at the trial
parameters (e.g. a negative diffusivity under a square root). The solver
treats this as a rejected step — it increases damping to pull the next trial
back toward the last accepted point — rather than a failure.
`ProblemError::Evaluation` is for unrecoverable failures.

## Levenberg-Marquardt algorithm

Each iteration solves the damped normal equations:

\\[
(J^{\\!T}\\!J + \\lambda \\cdot \\operatorname{diag}(J^{\\!T}\\!J)) \\delta
= -J^{\\!T}r
\\]

Damping is scaled by \\(\\operatorname{diag}(J^{\\!T}\\!J)\\) rather than the
identity (Marquardt's modification), making the step invariant to rescaling
of individual parameters — a diffusion model mixing diffusivities near
\\(10^{-3}\\) with signal amplitudes near \\(10^3\\) is exactly the
badly-scaled case that motivates it.

The algorithm proceeds as follows:

1. Evaluate residuals and Jacobian at the current parameters.
2. Check gradient convergence: \\(\\|J^{\\!T}r\\|_\\infty \\le\\) `gradient_tolerance`.
3. Build \\(J^{\\!T}\\!J\\) and solve the damped system via Cholesky
   decomposition (the matrix is symmetric positive-definite for \\(\\lambda > 0\\)).
4. If Cholesky fails, increase \\(\\lambda\\) and retry (up to a cap).
5. Accept the step when cost decreases; otherwise reject and increase
   \\(\\lambda\\). On acceptance, decrease \\(\\lambda\\).
6. Check step-tolerance and cost-tolerance convergence.

### Usage

```rust,ignore
use coeus_optim::{
    LeastSquaresProblem, LeastSquaresReport, LevenbergMarquardtConfig,
    ProblemError, levenberg_marquardt,
};

struct MyModel { /* data */ }

impl LeastSquaresProblem<f64> for MyModel {
    fn residual_count(&self) -> usize { /* ... */ }
    fn parameter_count(&self) -> usize { /* ... */ }
    fn residuals(&self, p: &[f64], r: &mut [f64]) -> Result<(), ProblemError> { /* ... */ }
    fn jacobian(&self, p: &[f64], j: &mut [f64]) -> Result<(), ProblemError> { /* ... */ }
}

let problem = MyModel { /* ... */ };
let initial = vec![0.0; problem.parameter_count()];
let config = LevenbergMarquardtConfig::default();
let report: LeastSquaresReport<f64> = levenberg_marquardt(
    &problem, &initial, &config,
)?;
```

## Configuration

`LevenbergMarquardtConfig<T>` carries seven tuning parameters:

| Parameter | Default | Meaning |
|---|---|---|
| `gradient_tolerance` | \\(\\sqrt{\\varepsilon}\\) | Stop when \\(\\|J^{\\!T}r\\|_\\infty\\) falls to this |
| `step_tolerance` | \\(\\sqrt{\\varepsilon}\\) | Stop when \\(\\|\\delta\\| \\le \\tau \\cdot (\\|p\\| + \\tau)\\) |
| `cost_tolerance` | \\(\\sqrt{\\varepsilon}\\) | Stop when relative cost reduction falls below this |
| `max_iterations` | 100 | Runaway guard (typical problems converge in single digits) |
| `initial_damping` | \\(10^{-3}\\) | Starting \\(\\lambda\\) |
| `damping_increase` | 10 | Multiply \\(\\lambda\\) on rejection |
| `damping_decrease` | 10 | Divide \\(\\lambda\\) on acceptance |

The tolerance defaults are \\(\\sqrt{\\varepsilon}\\) for the working scalar
type — the standard choice for a first-order criterion in floating point: a
residual computed to relative accuracy \\(\\varepsilon\\) cannot certify
stationarity below roughly \\(\\sqrt{\\varepsilon}\\).

## Report and termination

`LeastSquaresReport<T>` carries the result:

| Field | Meaning |
|---|---|
| `parameters` | Best accepted parameters |
| `cost` | \\(0.5\\|r\\|^2\\) at the solution |
| `gradient_norm` | \\(\\|J^{\\!T}r\\|_\\infty\\) at the solution |
| `iterations` | Iterations executed |
| `termination` | Why the solver stopped |

`Termination` has four variants:

| Variant | Certifies a minimum? |
|---|---|
| `GradientTolerance` | Yes — the point is stationary to tolerance |
| `StepTolerance` | No — progress stopped but the gradient may not be small |
| `CostTolerance` | No — cost stopped decreasing |
| `IterationLimit` | No — the solver ran out of budget |

`termination.is_converged()` is `true` for all variants except
`IterationLimit`.

## Error types

`SolverError` covers solver-level failures:

| Variant | Condition |
|---|---|
| `ParameterCount` | Initial parameters length ≠ `parameter_count()` |
| `Underdetermined` | Fewer residuals than parameters |
| `Problem(Evaluation)` | Unrecoverable evaluation failure |
| `Singular` | Normal equations unsolvable at maximum damping |
| `NonFinite` | NaN or infinity in residuals or Jacobian |

`ProblemError::Domain` is handled internally by the solver (treated as a
rejected step) and never propagates to the caller.

## First-order optimisers

`coeus-optim` also provides five gradient-descent optimisers — `SGD`,
`Adam`, `AdamW`, `RMSProp`, `AdaGrad` — used by the Coeus network training
stack. These implement the `Optimizer<T, B>` trait and are not used by the
diffusion pipeline, which requires the residual/curvature contract of
`LeastSquaresProblem`.

The DKI and NODDI models in `ritk-diffusion` each implement
`LeastSquaresProblem<f64>` and call `levenberg_marquardt` directly. See
the [Diffusion Models](ritk_diffusion.md) chapter for the model-specific
usage.
