# Leto Linear Algebra Operations

`leto-ops` owns the dense linear algebra, matrix decompositions, and
constrained solvers used by the diffusion pipeline. DTI and ODF estimation
route through `solve_least_squares`; CSD routes through `nnls`; DKI and
NODDI's damped normal equations route through `cholesky_solve`. RITK adds
no local solve, no local decomposition, and no local eigendecomposition.

## Least-squares solve

`solve_least_squares` solves the overdetermined linear system
\\(\\min \\|Ax - b\\|_2\\) via QR decomposition. It is the workhorse for
linear model fitting:

```rust,ignore
use leto_ops::solve_least_squares;
use leto::{Array1, Array2};

let solution = solve_least_squares(&design.view(), &rhs.view())?;
```

The DTI log-linear fit assembles one design matrix row per DWI volume and
calls this function once per voxel. The ODF spherical harmonic estimator
assembles an augmented design matrix (with Laplace–Beltrami penalty rows)
and calls the same function.

## NNLS — Non-Negative Least Squares

`nnls` implements the Lawson–Hanson active-set algorithm: it iteratively
adds variables to an active set, solves the unconstrained subproblem on the
active set, and removes variables with negative coefficients. The result is
a solution vector \\(x \\ge 0\\) that minimises \\(\\|Ax - b\\|_2\\).

### Configuration

`NnlsConfig` controls convergence:

| Parameter | Default | Meaning |
|---|---|---|
| `max_iterations` | `3 * n` | Iteration cap (active-set problems converge faster than this) |
| `tolerance` | \\(\\sqrt{\\varepsilon}\\) | Stop when the relative residual change falls below this |

### Result

`NnlsResult` carries:

| Field | Meaning |
|---|---|
| `solution` | Non-negative coefficient vector |
| `residual_norm` | \\(\\|Ax - b\\|_2\\) |
| `iterations` | Active-set iterations |
| `converged` | Whether the tolerance was met |

### Usage

```rust,ignore
use leto_ops::{nnls, NnlsConfig};

let config = NnlsConfig::default();
let result = nnls(&design.view(), &rhs.view(), &config)?;
assert!(result.solution.iter().all(|&x| x >= 0.0));
```

CSD uses NNLS to enforce the non-negativity constraint on the fibre
orientation distribution. The deconvolution matrix \\(B_{\\text{resp}}\\) is
assembled by rescaling the Apollo SH design matrix with the rotational
harmonics of the response function. See the [Diffusion
Models](ritk_diffusion.md#csd-constrained-spherical-deconvolution)
chapter.

## Cholesky decomposition and solve

`cholesky_decompose` factorises a symmetric positive-definite matrix as
\\(A = LL^{\\!T}\\). `cholesky_solve` solves \\(Ax = b\\) using the
Cholesky factorisation:

```rust,ignore
use leto_ops::cholesky_solve;

let step = cholesky_solve(&damped_normal.view(), &rhs.view())?;
```

This is the inner solve inside each Levenberg-Marquardt iteration. The
damped matrix \\(J^{\\!T}\\!J + \\lambda \\cdot \\operatorname{diag}(J^{\\!T}\\!J)\\)
is symmetric positive-definite for \\(\\lambda > 0\\), so Cholesky is the
correct factorisation. When it fails, damping is increased and the solve
is retried — see the [Coeus Solver](coeus_optim.md) chapter.

`cholesky_inv` and `cholesky_det` provide the inverse and determinant
from the Cholesky factor.

## QR decomposition

`qr_decompose` factorises \\(A = QR\\) into an orthogonal matrix \\(Q\\) and
an upper-triangular matrix \\(R\\). `solve_least_squares` uses QR
internally. `col_piv_qr` provides column-pivoted QR for rank-deficient
problems:

```rust,ignore
use leto_ops::qr_decompose;

let (q, r) = qr_decompose(&matrix.view())?;
```

## Pseudoinverse

`pinv` computes the Moore–Penrose pseudoinverse \\(A^+\\) via SVD:

```rust,ignore
use leto_ops::pinv;

let a_pinv = pinv(&matrix.view())?;
```

## Eigendecomposition

`symmetric_eigen_jacobi` computes all eigenvalues and eigenvectors of a
real symmetric matrix via the Jacobi algorithm. `eigenvalues` computes
eigenvalues of a general square matrix:

```rust,ignore
use leto_ops::symmetric_eigen_jacobi;

let decomposition = symmetric_eigen_jacobi(&matrix.view())?;
// decomposition.eigenvalues, decomposition.eigenvectors
```

RITK's DTI estimator uses its own analytic 3×3 symmetric eigendecomposition
(via the cubic characteristic polynomial) because the 3×3 case admits a
closed form. Larger problems would route through these Leto functions.

## Matrix norms and properties

| Function | Computes |
|---|---|
| `norm` / `norm_l1` / `norm_l2` / `norm_max` | Matrix norms |
| `det` | Determinant via LU |
| `trace` | Matrix trace |
| `matrix_rank` / `matrix_rank_with_tolerance` | Numerical rank via SVD |
| `inv` | Matrix inverse |
| `solve` | Solve \\(Ax = b\\) (general square) |

## Singular value decomposition

`svd_decompose` computes the full SVD \\(A = U \\Sigma V^{\\!T}\\). Variants
provide rank-revealing, tolerance-controlled, and bidiagonal-reduction
paths:

```rust,ignore
use leto_ops::svd_decompose;

let svd = svd_decompose(&matrix.view())?;
// svd.u, svd.singular_values, svd.vt
```

## Iterative solvers

`leto-ops` no longer ships Krylov recurrences: stack ownership of the
iterative-solver layer moved to Athena (Atlas ADR 0033), which provides
CG, BiCGSTAB, restarted GMRES, and damped LSQR over backend-generic
operators. The diffusion pipeline documented here needs none of them —
every solve routes through the direct dense layer in the table above
(`solve_least_squares`, `nnls`, `cholesky_solve`).

## Relationship to the diffusion pipeline

| Module | Leto function | Used by |
|---|---|---|
| `ritk-diffusion::dti` | `solve_least_squares` | Log-linear tensor fit |
| `ritk-diffusion::odf` | `solve_least_squares` | Regularised SH coefficient fit |
| `ritk-diffusion::csd` | `nnls` | Non-negative fODF deconvolution |
| `ritk-diffusion::dki` | `cholesky_solve` (via LM) | Damped normal equations |
| `ritk-diffusion::noddi` | `cholesky_solve` (via LM) | Damped normal equations |

Every solve routes through Leto; RITK owns only the design-matrix assembly
and post-processing.
