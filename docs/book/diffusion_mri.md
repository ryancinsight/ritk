# Diffusion MRI Acquisition and Q-ball ODFs

Diffusion-weighted MRI changes the signal according to the direction and
strength of a diffusion-sensitizing gradient. RITK separates this acquisition
description from image storage: `ritk-diffusion-scheme` owns validated gradient
directions, physical b-values, and coordinate frames, while
`ritk-diffusion` owns orientation models derived from voxel signals.

## From gradient pulse to b-value

For the idealized Stejskal–Tanner pulsed-gradient experiment, the diffusion
weighting is

\[
b = \gamma^2 G^2 \delta^2\left(\Delta-\frac{\delta}{3}\right),
\]

where \(\gamma\) is the gyromagnetic ratio, \(G\) the gradient amplitude,
\(\delta\) the pulse duration, and \(\Delta\) the pulse separation. The measured
signal for a Gaussian diffusion tensor \(D\) is

\[
S(b,\mathbf g)=S_0\exp\!\left(-b\,\mathbf g^T D\mathbf g\right).
\]

Thus \(b\) has dimensions of time per area, conventionally s/mm²; it is not a
diffusivity. RITK's `DiffusionWeighting` is a validated quantity with that
dimension and stores the canonical SI value internally. The scanner-facing
constructor and accessor use s/mm² explicitly. See Stejskal and Tanner,
[“Spin Diffusion Measurements”](https://doi.org/10.1063/1.1695690).

## A gradient scheme is more than two arrays

Every acquisition entry combines a weighting and direction under these
invariants:

- b-values are finite and nonnegative in both scanner-facing s/mm² and
  canonical SI storage;
- a b0 entry has the zero direction;
- a weighted entry has a finite unit direction; and
- every direction has one declared frame: image-axis or physical LPS.

At scanner and file-format boundaries, finite values at or below the default
50 s/mm² threshold are canonicalized to exact zero weighting and direction.
This treats small scanner baseline values consistently even when orientation
is absent or carries a nominal finite vector. Values above the threshold must
provide a unit direction.

`GradientScheme` keeps acquisition order and provides thresholded b0/DWI
indices and shell grouping. Reorientation accepts only a finite, proper
orthonormal rotation. Reflections and scale/shear matrices fail because they
would silently change physical orientation.

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

\[
E(\mathbf g)=\sum_{l,m} c_{lm}Y_{lm}(\mathbf g).
\]

The weighted samples must occupy one q-space shell. `OdfConfig` makes the
allowed absolute b-value tolerance explicit; an off-shell acquisition returns
`MixedShells` instead of entering a physically invalid angular fit. The fitted
`OdField` retains the scheme frame, so evaluation directions and reported
peaks keep an unambiguous coordinate meaning.

The least-squares system is augmented with a Laplace–Beltrami penalty. For
coefficient degree \(l\), the diagonal penalty factor is \(l(l+1)\), so the
configured nonnegative weight suppresses high-degree oscillation. The
Funk–Radon transform then acts degree by degree:

\[
\psi_{lm}=2\pi P_l(0)c_{lm}.
\]

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
