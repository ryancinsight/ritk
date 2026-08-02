# Apollo Spherical Harmonic Basis

`apollo-sht` owns the spherical harmonic basis used by the ODF and CSD models
in `ritk-diffusion`. The real, even-order, antipodally symmetric basis is an
upstream provider capability per [ADR 0036 decision
2](../../docs/adr/0036-neuroimaging-and-mr-ownership.md). RITK assembles the
design matrix and calls `apollo_sht` for basis evaluation; it implements no
local spherical-harmonic path, no local associated-Legendre recurrence, and no
local SH normalization.

## Why a separate SH provider?

The complex spherical harmonic transform in `apollo-sht` predates the
diffusion pipeline — it was built for spectral analysis on the sphere
(Gauss-Legendre grids, forward/inverse transforms, Parseval verification).
The real even-order basis is a *specialization* of that general machinery:
same associated-Legendre recurrence, same Condon-Shortley convention, same
normalization constants, different output convention (real, antipodally
symmetric, only even degrees). Placing the real basis in `apollo-sht`
alongside the complex SHT means:

- The recurrence kernel and normalization are verified once, not duplicated.
- The Gauss-Legendre quadrature that certifies SHT exactness also certifies
  the real basis's orthonormality on the sphere.
- Every consumer (ODF, CSD, and future FOD models) shares one basis instance.

## Real Basis Convention

The real, orthonormal convention follows the [MRtrix3
formulation](https://mrtrix.readthedocs.io/en/dev/concepts/spherical_harmonics.html#formulation-used-in-mrtrix3).
From the complex orthonormal harmonics \\(Y_l^m(\\theta, \\phi)\\) with
Condon-Shortley phase:

\\[
\\begin{aligned}
R_l^0(\\theta, \\phi)    &= \\operatorname{Re}\\big(Y_l^0(\\theta, \\phi)\\big) \\\\
R_l^m(\\theta, \\phi)    &= \\sqrt{2}\\, \\operatorname{Re}\\big(Y_l^m(\\theta, \\phi)\\big) \\quad (m > 0) \\\\
R_l^{-m}(\\theta, \\phi) &= \\sqrt{2}\\, \\operatorname{Im}\\big(Y_l^m(\\theta, \\phi)\\big) \\quad (m > 0)
\\end{aligned}
\\]

The basis is orthonormal: \\(\\int_{S^2} R_l^m R_{l'}^{m'}\\,d\\Omega = \\delta_{l,l'}\\delta_{m,m'}\\).
Only even degrees \\(l = 0, 2, 4, \\ldots, l_{\\max}\\) are included, because
the diffusion signal is antipodally symmetric (\\(S(\\mathbf{g}) = S(-\\mathbf{g})\\))
and odd-degree harmonics vanish.

The coefficient count is \\((l_{\\max} + 1)(l_{\\max} + 2) / 2\\) — for
\\(l_{\\max} = 8\\) that is 45 coefficients.

## RealSphericalHarmonicBasis

`RealSphericalHarmonicBasis` is the main type consumed by the diffusion
pipeline. It stores degree-major, order-minor metadata over even degrees:
`l=0: m=0`, `l=2: m=-2..=2`, `l=4: m=-4..=4`, and so on.

### Construction

```rust,ignore
use apollo_sht::RealSphericalHarmonicBasis;

// Create a basis for even degrees 0, 2, 4, 6, 8
let basis = RealSphericalHarmonicBasis::new(8)?;
```

Construction validates:
- `l_max` is even (odd degrees have no place in an antipodally symmetric basis).
- `l_max >= 2` (trivial basis not useful for orientation estimation).
- `l_max <= 85` (maximum stable degree in binary64 — the
  normalization product \\((2l)!\\) stays finite until degree 85).

### Accessors

| Method | Returns |
|---|---|
| `l_max() -> usize` | Maximum even degree |
| `num_coefficients() -> usize` | \\((l_{\\max} + 1)(l_{\\max} + 2) / 2\\) |
| `index_to_lm(index) -> Option<(usize, isize)>` | Maps flattened coefficient index to `(degree, order)` |
| `iter_lm() -> impl Iterator<Item = (usize, usize, isize)>` | Iterates `(index, degree, order)` triples |

## Pointwise Evaluation

`real_spherical_harmonic(degree, order, theta, phi)` evaluates one basis
function at spherical coordinates \\(\\theta \\in [0, \\pi]\\),
\\(\\phi \\in [0, 2\\pi)\\):

```rust,ignore
use apollo_sht::real_spherical_harmonic;

// R_2^2 at the equator, φ = 0
let value = real_spherical_harmonic(2, 2, std::f64::consts::FRAC_PI_2, 0.0)?;
```

For Cartesian directions, use the convenience method on the basis:

```rust,ignore
// Evaluate all 45 basis functions at a unit direction
let direction = [1.0_f64 / 3.0_f64.sqrt(); 3];
let row = basis.evaluate_at_direction(&direction)?;
// row.len() == 45
```

`evaluate_at_direction` validates that the input vector has unit-length
squared norm (within \\(32\\varepsilon\\)) and converts to spherical
coordinates.

To evaluate all basis functions at spherical angles:

```rust,ignore
let row = basis.evaluate(theta, phi)?;
```

## Design Matrix

`design_matrix(directions)` builds the full \\(N \\times K\\) design matrix
\\(B\\) where \\(B_{i,k} = R_{l(k)}^{m(k)}(\\theta_i, \\phi_i)\\) for
\\(N\\) gradient directions and \\(K\\) coefficients:

```rust,ignore
let directions: Vec<[f64; 3]> = scheme.directions.iter()
    .map(|d| d.unit_direction().to_array())
    .collect();
let design: leto::Array2<f64> = basis.design_matrix(&directions)?;
// design.shape() == [N, K]
```

The operation costs \\(O(N \\cdot K \\cdot l_{\\max})\\). No temporary row
vectors are allocated — each basis value is pushed directly into the
row-major buffer. The result is a Leto `Array2<f64>` ready for linear solves.

## Error Types

`RealShError` is a comprehensive, non-exhaustive error enum:

| Variant | Condition |
|---|---|
| `OddLMax(usize)` | `l_max` is not even |
| `TooSmall(usize)` | `l_max < 2` |
| `DegreeOutOfRange { degree, maximum }` | Degree exceeds `MAX_REAL_SH_DEGREE` (85) |
| `CoefficientCountOverflow(usize)` | Coefficient-count arithmetic overflowed |
| `MatrixSizeOverflow { rows, columns }` | Design-matrix element-count overflowed |
| `AllocationFailed { element_count }` | Could not reserve storage |
| `InvalidOrder { degree, order }` | `\|order\| > degree` |
| `InvalidTheta(f64)` / `InvalidPhi(f64)` | Angle out of domain |
| `NonFiniteDirection { axis, value }` | Cartesian component non-finite |
| `NonUnitDirection { norm_squared, tolerance }` | Direction not unit-length |
| `NonFiniteEvaluation { degree, order }` | Basis value is NaN or infinity |
| `MatrixShape { rows, columns }` | Leto rejected the matrix shape |

All errors are typed — no panics, no `.unwrap()` in the public API.

## Underlying Complex Machinery

The real basis sits on top of `apollo-sht`'s complex spherical harmonic
infrastructure:

| Component | Role |
|---|---|
| `spherical_harmonic(l, m, theta, phi)` | Complex \\(Y_l^m\\) with Condon-Shortley phase |
| `associated_legendre(l, m, x)` | \\(P_l^m(\\cos\\theta)\\) via Bonnet recurrence at fixed order |
| `normalization_constant(l, m)` | \\(N_{lm} = \\sqrt{(2l+1)/(4\\pi) \\cdot (l-m)!/(l+m)!}\\) |
| `gauss_legendre_nodes_weights(n)` | GL quadrature nodes and positive weights on \\([-1, 1]\\) |
| `ShtPlan` | Forward/inverse complex SHT on Gauss-Legendre product grids |
| `SphericalHarmonicCoefficients` | Dense \\((l_{\\max}+1) \\times (2l_{\\max}+1)\\) coefficient matrix |

The complex transform machinery is verified by:
- **Theorem 1** — Bonnet's recurrence converges for all \\(|x| \\leq 1\\)
  (numerically stable upward recurrence at fixed order).
- **Theorem 2** — The \\(n\\)-point GL rule integrates polynomials of degree
  \\(\\leq 2n-1\\) exactly (Golub & Welsch, 1969).
- **Theorem 3** — The complex harmonics are orthonormal on \\(S^2\\)
  (Driscoll & Healy, 1994).
- **Theorem 4** — Forward-inverse SHT round-trips exactly for band-limited
  fields when the grid satisfies the Shannon-Nyquist condition.
- **Parseval verification** — Energy \\(\\sum |a_{lm}|^2\\) is conserved
  through the transform (proptest-verified with random band-limited fields).

## Relationship to the Diffusion Pipeline

The real SH basis is consumed by two models in `ritk-diffusion`:

### ODF — Orientation Distribution Function

The analytical Q-ball ODF estimator:
1. Builds the design matrix \\(B\\) via `basis.design_matrix(&scheme.directions)`.
2. Appends Laplace-Beltrami penalty rows: \\(\\lambda \\cdot l^2(l+1)^2\\)
   on the diagonal for each \\((l, m)\\).
3. Solves the augmented system via `leto_ops::solve_least_squares`.
4. Evaluates the ODF on a display grid: \\(\\Psi(\\theta, \\phi) = \\sum c_{lm} R_l^m(\\theta, \\phi)\\).

```rust,ignore
use apollo_sht::RealSphericalHarmonicBasis;
use leto_ops::solve_least_squares;

let basis = RealSphericalHarmonicBasis::new(config.l_max())?;
let design = basis.design_matrix(&directions)?;
// ... add Laplace-Beltrami rows ...
let coeffs = solve_least_squares(&augmented_design.view(), &augmented_rhs.view())?;
```

### CSD — Constrained Spherical Deconvolution

The fibre ODF estimator:
1. Builds the SH design matrix as above.
2. Computes the response function's rotational harmonics
   \\(r_l = 2\\pi \\int_{-1}^{1} R(\\cos\\alpha)\\,P_l(\\cos\\alpha)\\,d(\\cos\\alpha)\\).
3. Forms the convolution matrix \\(B_{\\text{resp}}\\) by scaling each SH
   column by the corresponding \\(r_l\\).
4. Solves via `leto_ops::nnls` to enforce non-negative fODF coefficients.
5. Extracts peaks from the reconstructed fODF for tractography.

```rust,ignore
let basis = RealSphericalHarmonicBasis::new(config.l_max())?;
let design = basis.design_matrix(&directions)?;
let conv_matrix = build_deconvolution_matrix(&design, &response, &basis)?;
let result = nnls(&conv_matrix.view(), &signal.view(), &nnls_config)?;
// result.solution contains non-negative fODF coefficients
```

The design matrix is the same object in both models — only the subsequent
linear system changes (Tikhonov-regularized least-squares for ODF,
non-negative least-squares for CSD).

## Boundary

Under ADR 0036 decision 2, the real SH basis and its design matrix belong to
Apollo, never to RITK. A RITK-local associated-Legendre recurrence or SH
normalization path is a boundary violation. The chapter in
[ritk_diffusion.md](ritk_diffusion.md) documents the *consumption* side; this
chapter documents the *provider* contract that RITK calls into.

## References

- Driscoll, J.R. & Healy, D.M. (1994). "Computing Fourier transforms and
  convolutions on the 2-sphere." *Advances in Applied Mathematics*, 15(2),
  202-250.
- Golub, G.H. & Welsch, J.H. (1969). "Calculation of Gauss quadrature rules."
  *Mathematics of Computation*, 23(106), 221-230.
- MRtrix3 spherical harmonic formulation:
  <https://mrtrix.readthedocs.io/en/dev/concepts/spherical_harmonics.html>
- Tournier, J.D., Calamante, F., & Connelly, A. (2007). "Robust determination
  of the fibre orientation distribution in diffusion MRI." *NeuroImage*,
  35(4), 1459-1472.
