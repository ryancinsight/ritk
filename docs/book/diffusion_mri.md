# Diffusion MRI Acquisition and Q-ball ODFs

Diffusion-weighted MRI measures how water self-diffusion varies with gradient
direction and strength. RITK separates three concerns:

| Concern | Owner | Chapter |
|---|---|---|
| Validated gradient metadata (b-values, directions, frames, codecs, reorientation) | `ritk-diffusion-scheme` | [Diffusion Gradient Schemes](diffusion_scheme.md) |
| Orientation models derived from voxel signals | `ritk-diffusion` | This chapter (Q-ball ODFs; DTI, DKI, CSD, and NODDI follow the same pipeline pattern) |
| Streamline integration over a local orientation field | `ritk-tractography` | [Deterministic Streamline Tractography](tractography.md) |

This chapter focuses on the physics that relates gradient pulses to signal
attenuation — the bridge between the validated metadata in the scheme chapter
and the model-fitting algorithms documented in the [Diffusion
Models](ritk_diffusion.md) chapter.

## From gradient pulse to b-value

For the idealized Stejskal–Tanner pulsed-gradient experiment, the diffusion
weighting is

```text
b = γ² G² δ² (Δ − δ/3)
```

where `γ` is the gyromagnetic ratio, `G` the gradient amplitude, `δ` the pulse
duration, and `Δ` the pulse separation. The measured signal for a Gaussian
diffusion tensor `D` is

```text
S(b, g) = S₀ exp(−b gᵀ D g)
```

Thus \(b\) has dimensions of time per area, conventionally s/mm²; it is not a
diffusivity. `DiffusionWeighting` stores the canonical SI value (s/m²)
internally through the Aequitas quantity system and exposes the scanner-facing
s/mm² value at the API boundary. See [the scheme
chapter](diffusion_scheme.md#diffusion-weighting) for the type-level contract
and Stejskal and Tanner,
[“Spin Diffusion Measurements”](https://doi.org/10.1063/1.1695690).

## Relationship to the gradient scheme

The [Diffusion Gradient Schemes](diffusion_scheme.md) chapter establishes
four guarantees that every model fitting depends on: validated directions,
unambiguous coordinate frames, scheme-level reorientation, and shell
grouping. The [Diffusion Models](ritk_diffusion.md) chapter documents how
all five implemented models (DTI, DKI, ODF, CSD, NODDI) consume these
guarantees — this chapter focuses on the physics shared by all of them.

## From scheme to signal to model

The pipeline established by the three chapters in this part is:

1. **Validated scheme** — the [Diffusion Gradient Schemes](diffusion_scheme.md)
   chapter ensures every direction carries a declared frame, a physically
   typed weighting, and a validated unit-vector contract.
2. **Orientation model** — the [Diffusion Models](ritk_diffusion.md) chapter
   documents the transformation from validated scheme + voxel signals →
   fitted field. Five models are implemented: DTI (log-linear tensor),
   DKI (nonlinear kurtosis), ODF (analytical Q-ball), CSD (non-negative
   fODF), and NODDI (3-compartment biophysical).
3. **Streamline geometry** — the [Deterministic Streamline
   Tractography](tractography.md) chapter integrates the fitted orientation
   field into Gaia polyline curves.

~~~rust,ignore
use ritk_diffusion_scheme::{GradientFrame, GradientScheme};
use ritk_spatial::Vector;

let scheme = GradientScheme::from_seconds_per_square_millimeter(
    vec![
        (0.0, Vector::new([0.0, 0.0, 0.0])),
        (1_000.0, Vector::new([1.0, 0.0, 0.0])),
        (1_000.0, Vector::new([0.0, 1.0, 0.0])),
    ],
    GradientFrame::ImageAxis,
)?;
assert_eq!(scheme.len(), 3);
# Ok::<(), ritk_diffusion_scheme::GradientSchemeError>(())
~~~

## Format coordinate contracts

- FSL `bvec` directions are image-axis coordinates. The three rows are x, y,
  and z components; the companion `bval` sequence supplies s/mm² values.
  `DiffusionMaps` retains this `ImageAxis` frame. When a fitted field is placed
  in `DtiVolume`, the volume validates that frame and converts the external
  `[column, row, depth]` direction order to RITK's image-index
  `[depth, row, column]` order exactly once. LPS maps are rejected at this
  boundary because physical-to-index conversion requires image geometry that
  `DtiVolume` does not own.
- Standard DICOM Diffusion Gradient Orientation `(0018,9089)` is in the
  patient coordinate system and RITK labels it LPS. Diffusion b-value
  `(0018,9087)` is s/mm². The current reader accepts top-level attributes from
  classic one-instance-per-volume inputs; it does not guess private vendor
  fields or enhanced functional groups. See [DICOM PS3.3
  C.8.13.5.9](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_c.8.13.5.9.html).
- NA-MIC DWI NRRD stores one nominal `DWMRI_b-value`. A gradient's squared norm
  relative to the largest gradient norm scales that volume's effective
  b-value. The measurement frame maps the gradient into world coordinates,
  after which RAS is converted once to RITK's LPS physical frame. See the
  [NA-MIC DWI NRRD convention](https://www.na-mic.org/wiki/NAMIC_Wiki%3ADTI%3ANrrd_format).

These distinctions prevent a numerically valid direction from being applied
in the wrong physical frame.

## Analytical Q-ball estimation

Q-ball imaging estimates an orientation distribution function (ODF) without
assuming a single diffusion tensor. RITK first normalizes weighted signals by
the mean b0 signal and fits an even-degree real spherical-harmonic expansion:

```text
E(g) = Σₗ,ₘ cₗₘ Yₗₘ(g)
```

The weighted samples must occupy one q-space shell. `OdfConfig` makes the
allowed absolute b-value tolerance explicit; an off-shell acquisition returns
`MixedShells` instead of entering a physically invalid angular fit. The fitted
`OdField` retains the scheme frame, so evaluation directions and reported
peaks keep an unambiguous coordinate meaning.

The least-squares system is augmented with a Laplace–Beltrami penalty. For
coefficient degree `l`, the diagonal penalty factor is `l(l + 1)`, so the
configured nonnegative weight suppresses high-degree oscillation. The
Funk–Radon transform then acts degree by degree:

```text
ψₗₘ = 2π Pₗ(0) cₗₘ
```

This is the analytical Q-ball construction of Descoteaux et al.,
[“Regularized, Fast, and Robust Analytical Q-Ball Imaging”](https://doi.org/10.1002/mrm.21277).
It produces an ODF, not a constrained-spherical-deconvolution fiber orientation
density; those quantities must not be interpreted interchangeably.

`OdfConfig` validates harmonic degree, regularization, the b0 threshold, and
the single-shell tolerance.
Estimation fails on non-finite signals, non-finite values created by baseline
normalization or the numerical solve, missing b0 or weighted samples,
nonpositive baseline, an underdetermined basis, or a failed solve. Evaluation
also rejects overflow created by finite coefficients and finite basis values.
`OdField` stores coefficients contiguously and reuses its basis metadata for
repeated evaluation. `evaluate_on_grid` returns one flat row-major allocation
rather than a pointer-chasing grid of rows.

The [signal-to-streamlines example](examples/diffusion_tractography.md) uses a
known tensor to verify that the recovered antipodal ODF peak matches its
analytical x axis.

The [signal-to-streamlines example](examples/diffusion_tractography.md)
closes the loop: known tensor → synthetic signals → model fit → direction
field → streamlines.
