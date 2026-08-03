# Diffusion Models

`ritk-diffusion` owns orientation models derived from voxel signals. Every
model consumes a validated [`GradientScheme`](diffusion_scheme.md) and one
voxel's signal vector; every model produces a typed result that carries its
coordinate frame forward. This chapter documents all five implemented models.

## Model-to-solver map

| Model | Solver | Solver owner | Key output |
|---|---|---|---|
| DTI — Diffusion Tensor Imaging | `leto_ops::solve_least_squares` | Leto | FA, MD, PEV |
| DKI — Diffusion Kurtosis Imaging | `coeus_optim::levenberg_marquardt` | Coeus | MK, AK, RK |
| ODF — Analytical Q-ball | `leto_ops::solve_least_squares` | Leto | ODF coefficients, peaks |
| CSD — Constrained Spherical Deconvolution | `leto_ops::nnls` | Leto | Non-negative fODF, fibre peaks |
| NODDI — Neurite Orientation Dispersion | `coeus_optim::levenberg_marquardt` | Coeus | NDI, ODI, fISO, direction |

The model type owns only the estimation and the derived metrics. Every
solver is dispatched to its owner (Leto for linear algebra, Coeus for
nonlinear optimisation, Apollo for the SH basis) — RITK adds no local
solver, no local gradient, and no local SH implementation.

## Relationship to the gradient scheme

Every model in this crate assumes a validated `GradientScheme`. The
[Diffusion Gradient Schemes](diffusion_scheme.md) chapter establishes
the guarantees this chapter depends on:

1. Every direction is validated (b = 0 → zero vector, weighted → unit vector).
2. Every direction has one declared frame (`ImageAxis` or `Lps`).
3. Gradient reorientation is a scheme-level operation — it must be called
   *before* the scheme enters any model fitter.
4. Shell grouping is provided by the scheme (`b0_indices`, `dwi_indices`,
   `shells`).

No model in this chapter re-validates direction norms or re-derives shell
structure. The scheme chapter owns format-codec concerns; this chapter owns
the transformation from validated scheme + signals → fitted field.

## DTI — Diffusion Tensor Imaging

DTI is the baseline diffusion model. It assumes a 3-D Gaussian
displacement distribution characterised by a symmetric 3×3 diffusion
tensor \\(D\\). The normalised signal at gradient direction \\(\\mathbf g\\)
and b-value \\(b\\) is

\\[
\\frac{S}{S_0} = \\exp\\!\\left(-b\\,\\mathbf g^{\\!T} D\\,\\mathbf g\\right).
\\]

Taking the log gives a linear system in the six unique tensor elements:

\\[
\\ln(S/S_0) = -b \\cdot [g_x^2, g_y^2, g_z^2, 2g_x g_y, 2g_x g_z, 2g_y g_z]
\\cdot \\mathbf d
\\]

where \\(\\mathbf d = [D_{xx}, D_{yy}, D_{zz}, D_{xy}, D_{xz}, D_{yz}]^{\\!T}\\).
RITK assembles the design matrix over all weighted acquisitions and solves
via `leto_ops::solve_least_squares`. Fractional anisotropy, mean
diffusivity, and the principal eigenvector are derived from the fitted
tensor through a closed-form 3×3 symmetric eigendecomposition, per
[ADR 0036 decision 2](../../docs/adr/0036-neuroimaging-and-mr-ownership.md).

### Configuration

`DtiConfig` carries one parameter: the `b0_threshold` separating reference
and weighted volumes. The default is 50 s/mm², the same threshold used by
`GradientScheme::from_seconds_per_square_millimeter`.

```rust,ignore
use ritk_diffusion::dti::{estimate_dti, DtiConfig};

let config = DtiConfig::default();
let tensor = estimate_dti(&scheme, &signals, config)?;
```

### Output

`DiffusionTensor` provides:

| Method | Unit | Meaning |
|---|---|---|
| `elements()` | mm²/s | Six Voigt elements `[Dxx, Dyy, Dzz, Dxy, Dxz, Dyz]` |
| `matrix()` | mm²/s | Full 3×3 symmetric matrix |
| `eigenvalues()` | mm²/s | `λ₀ ≥ λ₁ ≥ λ₂`, sorted descending |
| `fa()` | — | Fractional anisotropy ∈ [0, 1] |
| `md()` | mm²/s | Mean diffusivity, `(λ₀ + λ₁ + λ₂) / 3` |
| `principal_eigenvector()` | — | Unit PEV in the scheme's frame |
| `baseline_signal()` | — | Mean S₀ over b0 acquisitions |
| `residual_norm()` | — | ‖design · d − ln(S/S₀)‖₂ |
| `frame()` | — | Coordinate frame |

### Error conditions

`DtiError` covers signal/scheme length mismatch, non-finite signals,
missing b0 or weighted volumes, fewer than six DWI directions
(`Underdetermined`), invalid baseline or normalised signals, a failed
linear solve, and non-positive eigenvalues in the fitted tensor.

## DKI — Diffusion Kurtosis Imaging

DKI extends DTI with the kurtosis tensor \\(W\\), a fourth-order symmetric
tensor that captures non-Gaussian diffusion. The normalised signal is

\\[
\\frac{S}{S_0}= \\exp\\!\\left(-b\\,D(\\mathbf g)
+ \\frac{b^2}{6}\\,\\mathrm{MD}^2\\,W(\\mathbf g)\\right)
\\]

where \\(D(\\mathbf g)=\\mathbf g^{\\!T}D\\mathbf g\\) is the apparent
diffusion coefficient, \\(\\mathrm{MD} = \\operatorname{tr}(D)/3\\), and
\\(W(\\mathbf g)\\) is the full contraction of the kurtosis tensor with
the gradient direction.

The model has 21 parameters (6 for D, 15 for W). Estimation proceeds in
two stages:

1. A log-linear DTI fit supplies the initial D guess and baseline S₀.
2. Levenberg-Marquardt refines D and W simultaneously, using the analytic
   Jacobian of the DKI residual.

The solver requires at least 21 DWI directions distributed across multiple
shells — a single-shell scheme cannot disambiguate the quadratic D term
from the quartic W term.

### Configuration

`KtiConfig` carries a `b0_threshold` and a `LevenbergMarquardtConfig`.
The default tolerances are appropriate for synthetic data; in-vivo fitting
typically requires relaxed convergence criteria.

```rust,ignore
use ritk_diffusion::dki::{estimate_dki, KtiConfig};

let config = KtiConfig::default();
let kt = estimate_dki(&scheme, &signals, &config)?;
```

### Kurtosis metrics

| Metric | Symbol | Definition |
|---|---|---|
| Mean kurtosis | MK | \\(\\langle K(\\mathbf g)\\rangle\\) over 200 quasi-uniform sphere directions |
| Axial kurtosis | AK | \\(K(\\mathbf e_1)\\) along the principal eigenvector |
| Radial kurtosis | RK | \\(\\langle K(\\mathbf g)\\rangle\\) over directions perpendicular to \\(\\mathbf e_1\\) |

The apparent kurtosis coefficient is \\(K(\\mathbf g) = (\\mathrm{MD}/D(\\mathbf g))^2
\\cdot W(\\mathbf g)\\).

### Output

`DiffusionKurtosisTensor` provides `elements_d()` (six D elements),
`elements_w()` (fifteen W elements), `eigenvalues()`, `fa()`, `md()`,
`mk()`, `ak()`, `rk()`, `principal_eigenvector()`, `baseline_signal()`,
`residual_norm()`, `converged()`, `iterations()`, `gradient_norm()`,
`frame()`, `predict_signal()`, `quadratic_form()`, and
`kurtosis_at_direction()`.

### Error conditions

`KtiError` covers signal/scheme length mismatch, non-finite signals,
missing b0/weighted volumes, invalid baseline or normalised signals, a
failed DTI initial fit, and Levenberg-Marquardt solver failures.

## ODF — Analytical Q-ball

Q-ball imaging estimates an orientation distribution function (ODF)
without assuming a single diffusion tensor. RITK implements the analytical
construction of Descoteaux et al. (2007): the normalised signal is
expanded in Apollo's real, even-degree spherical harmonic basis, and the
Funk–Radon transform converts signal coefficients to ODF coefficients.

### Signal model

\\[
E(\\mathbf g)=\\sum_{l=0,2,4,\\dots}^{l_{\\max}} \\sum_{m=-l}^{l}
c_{lm}Y_{lm}(\\mathbf g)
\\]

Only even degrees appear because diffusion is antipodally symmetric. The
design matrix is evaluated at the scheme's scattered gradient directions
by `apollo_sht::RealSphericalHarmonicBasis::design_matrix`.

### Regularised least squares

The coefficients are estimated by solving

\\[
\\hat{\\mathbf c} = (B^{\\!T} B + \\lambda L)^{-1} B^{\\!T} \\mathbf e
\\]

where \\(\\lambda \\ge 0\\) is the Laplace–Beltrami penalty weight and
\\(L\\) has diagonal entries \\(l(l+1)\\) for each degree-\\(l\\)
coefficient. The solve routes through `leto_ops::solve_least_squares`.

### Funk–Radon transform

The fitted signal coefficients become Q-ball ODF coefficients through

\\[
\\psi_{lm}=2\\pi P_l(0)c_{lm}
\\]

where \\(P_l(0)\\) is the Legendre polynomial of degree \\(l\\) at zero.
This produces an **ODF**, not a fibre orientation density — the two must
not be interpreted interchangeably.

### Single-shell constraint

Analytical Q-ball fits one q-space shell. `OdfConfig` carries a
`shell_tolerance` (maximum absolute b-value difference among weighted
volumes). A multi-shell acquisition returns `MixedShells` instead of
entering a physically invalid angular fit.

### Configuration

```rust,ignore
use ritk_diffusion::odf::{estimate_odf, OdfConfig};
use ritk_diffusion_scheme::DiffusionWeighting;

let config = OdfConfig::new(
    4,                                    // l_max
    0.006,                                // regularization
    DiffusionWeighting::from_seconds_per_square_millimeter(50.0)?,
    DiffusionWeighting::from_seconds_per_square_millimeter(0.0)?,
)?;
let odf = estimate_odf(&scheme, &signals, config)?;
```

### Output

`OdField` provides `coefficients()`, `l_max()`, `baseline_signal()`,
`normalized_signal_residual()`, `frame()`, `evaluate(theta, phi)`,
`evaluate_at_direction([x, y, z])`, and `evaluate_on_grid(theta_samples, phi_samples)`.

### Error conditions

`OdfError` covers signal/scheme length mismatch, non-finite signals,
missing b0/weighted volumes, mixed shells, underdetermined basis, invalid
baseline, invalid regularisation, invalid spherical grid, invalid
evaluation directions, basis errors (propagated from Apollo), and failed
solves.

## CSD — Constrained Spherical Deconvolution

CSD deconvolves the diffusion signal with an axially symmetric response
function to recover the fibre orientation distribution (fODF). In the
spherical harmonic basis the convolution becomes a diagonal rescaling of
each degree block:

\\[
s_{lm} = \\sqrt{\\frac{4\\pi}{2l+1}} \\cdot r_l \\cdot f_{lm}
\\]

where \\(r_l\\) are the rotational harmonics of the single-fibre response.
The deconvolution matrix is \\(B_{\\text{resp}} = B \\cdot
\\operatorname{diag}(\\kappa_l)\\) with \\(\\kappa_l = 4\\pi/(2l+1) \\cdot
r_l\\). The fODF coefficients are recovered by solving

\\[
\\min \\|B_{\\text{resp}} \\mathbf f - \\mathbf S/S_0\\|_2 \\quad
\\text{subject to} \\quad \\mathbf f \\ge 0
\\]

through `leto_ops::nnls` (Lawson–Hanson active-set NNLS). The result is a
non-negative fODF whose peaks correspond to fibre directions.

### Response function

`ResponseFunction` carries rotational harmonics \\(r_0, r_2, r_4, \\dots\\)
with \\(r_0 = 1.0\\) (normalised to unit baseline). The convenience
constructor `from_tensor(b_value, ad, rd, l_max)` computes them from an
axially symmetric diffusion tensor by numerical projection onto the
Legendre polynomials.

### Configuration

```rust,ignore
use ritk_diffusion::csd::{estimate_fod, CsdConfig, ResponseFunction};
use leto_ops::NnlsConfig;

let response = ResponseFunction::from_tensor(3_000.0, 0.0017, 0.0003, 8)?;
let config = CsdConfig::new(8, b0_threshold, NnlsConfig::default())?;
let fod = estimate_fod(&scheme, &signals, &response, &config)?;
```

### Output

`FodField` provides `coefficients()` (guaranteed non-negative),
`l_max()`, `baseline_signal()`, `residual_norm()`,
`nnls_iterations()`, `nnls_converged()`, `frame()`,
`evaluate(theta, phi)`, `evaluate_at_direction([x, y, z])`,
`evaluate_on_grid(theta, phi)`, and `find_peaks(grid_theta, grid_phi,
relative_threshold)`.

### Peak extraction

`find_peaks` samples the fODF on a dense equiangular grid, locates every
sample exceeding all eight neighbours and a configurable
relative-amplitude floor, then deduplicates peaks within 5° of each other
(to handle pole-adjacent duplicates from the φ-periodic grid). Peaks are
returned sorted by descending amplitude.

### Volume type

`FodVolume` stores a 3-D grid of fODF coefficients and supports trilinear
interpolation for sub-voxel direction queries during whole-brain
tractography via `direction_at(point, grid_theta, grid_phi,
relative_threshold)`.

### Error conditions

`CsdError` covers signal/scheme length mismatch, non-finite signals,
missing b0/weighted volumes, invalid baseline, underdetermined system,
response degree too low for the requested basis, invalid \\(r_0\\),
invalid evaluation/grid parameters, Apollo basis errors, NNLS failures,
and volume construction validation (shape, spacing, origin, coefficient
count).

### Comparison with ODF

| Property | `odf` (analytical Q-ball) | `csd` |
|---|---|---|
| Solver | Laplace–Beltrami-regularised least-squares | Lawson–Hanson NNLS |
| Output | ODF (can be negative) | fODF (guaranteed non-negative) |
| Response | Implicit (Funk–Radon of signal SH) | Explicit (rotational harmonics \\(r_l\\)) |
| Purpose | Orientation distribution | Fibre orientation density |
| Tractography | ODF peaks | fODF peaks (sharper angular resolution) |

## NODDI — Neurite Orientation Dispersion and Density Imaging

NODDI (Zhang et al., 2012) is a 3-compartment tissue model that separates
water diffusion into intra-neurite (restricted), extra-neurite (hindered),
and CSF (free) pools:

\\[
\\frac{S}{S_0} = (1 - f_{\\text{iso}})\\left[f_{\\text{intra}}A_{\\text{ic}}
+ (1 - f_{\\text{intra}})A_{\\text{ec}}\\right]
+ f_{\\text{iso}}A_{\\text{iso}}
\\]

- \\(A_{\\text{ic}}\\) — Watson-averaged stick signal evaluated by Monte
  Carlo quadrature over 300 quasi-uniform directions on the sphere.
- \\(A_{\\text{ec}} = \\exp(-b \\cdot d_{\\text{ec}})\\) — hindered
  extra-cellular compartment.
- \\(A_{\\text{iso}} = \\exp(-b \\cdot d_{\\text{iso}})\\) — free-water
  CSF compartment.

Biophysical constants are fixed: \\(d_{\\parallel} = 1.7 \\times 10^{-3}\\)
mm²/s, \\(d_{\\text{ec}} = 0.8 \\times 10^{-3}\\) mm²/s, \\(d_{\\text{iso}}
= 3.0 \\times 10^{-3}\\) mm²/s.

### Parameters and metrics

| Parameter | Symbol | Range | Metric |
|---|---|---|---|
| Intra-cellular fraction | \\(f_{\\text{intra}}\\) | [0, 1] | NDI — neurite density index |
| CSF fraction | \\(f_{\\text{iso}}\\) | [0, 1] | fISO — free water fraction |
| Orientation dispersion | ODI | [0, 1] | `(2/π)·arctan(1/κ)` |
| Polar angle | \\(\\theta\\) | [0, π] | Principal fibre direction |
| Azimuthal angle | \\(\\phi\\) | [0, 2π) | Principal fibre direction |

ODI = 0 means perfectly aligned sticks (κ → ∞); ODI = 1 means isotropic
dispersion (κ → 0). The extra-cellular fraction is computed as
\\(f_{\\text{ec}} = (1 - f_{\\text{iso}})(1 - f_{\\text{intra}})\\).

### Solver

Fitting uses Levenberg-Marquardt with five parameters. The Jacobian has
two analytic columns (\\(f_{\\text{intra}}\\), \\(f_{\\text{iso}}\\)) and
three finite-difference columns (ODI, θ, φ). The initial guess uses DTI's
PEV for the fibre direction and conservative defaults for the volume
fractions.

### Configuration

```rust,ignore
use ritk_diffusion::noddi::{estimate_noddi, NoddiConfig};

let config = NoddiConfig::default();
let fit = estimate_noddi(&scheme, &signals, &config)?;
```

### Output

`NoddiFit` provides `ndi()` / `f_intra()`, `f_iso()`, `odi()`,
`f_extra()`, `principal_direction()`, `baseline_signal()`,
`residual_norm()`, `converged()`, `iterations()`,
`gradient_norm()`, `frame()`, and `predict_signal(direction, b_value)`.

### Volume type

`NoddiVolume` stores a 3-D grid of principal directions and supports
nearest-neighbour spatial lookup for tractography via
`direction_at(point)`. Unlike `FodVolume`, no peak extraction is needed —
NODDI intrinsically yields a single fibre orientation per voxel.

### Error conditions

`NoddiError` covers signal/scheme length mismatch, non-finite signals,
missing b0/weighted volumes, invalid baseline, solver failures (including
DTI initial-fit failures), and volume construction validation.

## From model fields to streamlines

Every model in this crate produces a type that can feed a direction field
for tractography. The direction-field helpers live in `ritk-tractography`
(see the [tractography chapter](tractography.md)):

| Model | Direction source | Tractography helper |
|---|---|---|
| DTI | `principal_eigenvector()` | `dti_pev_direction_field(tensor)` |
| DKI | `principal_eigenvector()` | (same PEV, richer kurtosis metrics) |
| ODF | `OdField` peak | (manual peak extraction) |
| CSD | `FodVolume::direction_at()` | `fod_volume_direction_field(volume)` |
| NODDI | `NoddiVolume::direction_at()` | `noddi_direction_field(volume)` |

The [Deterministic Streamline Tractography](tractography.md) chapter
documents the integration algorithm that consumes these direction fields.
The [signal-to-streamlines example](examples/diffusion_tractography.md)
closes the loop end-to-end: known tensor → synthetic signals → model fit
→ direction field → streamlines.
