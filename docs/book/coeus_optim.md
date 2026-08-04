# Coeus Nonlinear Least-Squares Solver

`coeus-optim` owns the Levenberg-Marquardt (damped Gauss-Newton) solver used
by the nonlinear diffusion models in
[ADR 0036 decision 2](../../docs/adr/0036-neuroimaging-and-mr-ownership.md).
DKI, NODDI, and IVIM fit their parameters through this solver; RITK adds no
local nonlinear optimiser.

## Why a separate solver contract?

The gradient-descent optimisers in `coeus-optim` (`SGD`, `Adam`, `AdamW`,
`RMSProp`, `AdaGrad`) step on gradients already accumulated into parameters,
which suits network training. A least-squares solver instead re-evaluates the
model at trial points to decide whether a step is acceptable, and exploits
the Gauss-Newton curvature approximation `JᵀJ ≈ H` that a
bare gradient does not expose. The two are different contracts, not two
spellings of one.

## LeastSquaresProblem trait

`LeastSquaresProblem<T>` is the contract between a model and the solver. It
requires:

| Method | Purpose |
|---|---|
| `residual_count() -> usize` | Number of residual components `m` |
| `parameter_count() -> usize` | Number of free parameters `n` |
| `residuals(&self, &[T], &mut [T])` | Evaluate `r(p)` — `m` values |
| `jacobian(&self, &[T], &mut [T])` | Evaluate `J(p) = ∂r/∂p` — row-major, `m × n` |

The Jacobian is row-major: entry `(i, j)` at index `i · n + j` is
`∂rᵢ / ∂pⱼ`.

The solver requires `m ≥ n` — fewer residuals than parameters makes
`JᵀJ` singular by construction, and damping would mask that
rather than solve it.

### Domain errors

`ProblemError::Domain` signals that the model is undefined at the trial
parameters (e.g. a negative diffusivity under a square root). The solver
treats this as a rejected step — it increases damping to pull the next trial
back toward the last accepted point — rather than a failure.
`ProblemError::Evaluation` is for unrecoverable failures.

## Levenberg-Marquardt algorithm

Each iteration solves the damped normal equations:

```text
(JᵀJ + λ · diag(JᵀJ)) δ = −Jᵀr
```

Damping is scaled by `diag(JᵀJ)` rather than the
identity (Marquardt's modification), making the step invariant to rescaling
of individual parameters — a diffusion model mixing diffusivities near
`10⁻³` with signal amplitudes near `10³` is exactly the
badly-scaled case that motivates it.

The algorithm proceeds as follows:

1. Evaluate residuals and Jacobian at the current parameters.
2. Check gradient convergence: `‖Jᵀr‖∞ ≤ gradient_tolerance`.
3. Build `JᵀJ` and solve the damped system via Cholesky
   decomposition (the matrix is symmetric positive-definite for `λ > 0`).
4. If Cholesky fails, increase `λ` and retry (up to a cap).
5. Accept the step when cost decreases; otherwise reject and increase
   `λ`. On acceptance, decrease `λ`.
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
| `gradient_tolerance` | `√ε` | Stop when `‖Jᵀr‖∞` falls to this |
| `step_tolerance` | `√ε` | Stop when `‖δ‖ ≤ τ · (‖p‖ + τ)` |
| `cost_tolerance` | `√ε` | Stop when relative cost reduction falls below this |
| `max_iterations` | 100 | Runaway guard (typical problems converge in single digits) |
| `initial_damping` | `10⁻³` | Starting `λ` |
| `damping_increase` | 10 | Multiply `λ` on rejection |
| `damping_decrease` | 10 | Divide `λ` on acceptance |

The tolerance defaults are `√ε` for the working scalar
type — the standard choice for a first-order criterion in floating point: a
residual computed to relative accuracy `ε` cannot certify stationarity below
roughly `√ε`.

## Report and termination

`LeastSquaresReport<T>` carries the result:

| Field | Meaning |
|---|---|
| `parameters` | Best accepted parameters |
| `cost` | `0.5‖r‖²` at the solution |
| `gradient_norm` | `‖Jᵀr‖∞` at the solution |
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
