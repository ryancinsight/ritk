# ADR 0017: Physically typed diffusion MRI pipeline

- Status: Accepted
- Date: 2026-07-31
- Board item: [FEAT-686-01](../../backlog.md#feat-686-01-minorarch---establish-a-physically-typed-diffusion-mri-and-tractography-pipeline)

## Context

The acquisition-series increment introduces diffusion-gradient metadata,
spherical-harmonic fitting, and deterministic streamline integration. Its
recovered public contracts are not safe to publish as written:

- b-values in seconds per square millimeter are represented by an Aequitas
  dimensionless quantity named `Diffusivity`, although diffusion weighting has
  the reciprocal dimension of diffusivity;
- the NRRD reader interprets `DWMRI_b-value` as a list, while the NRRD DWI
  convention defines one nominal value and encodes per-volume weighting in the
  squared gradient magnitude;
- DICOM gradient orientation is defined in the patient frame, but the reader
  labels it as an image-axis vector;
- the spherical-harmonic coefficients fitted to the normalized signal are
  reported directly as an orientation distribution function without applying
  the Funk-Radon transform;
- public configuration fields admit NaN, infinity, zero step sizes, invalid
  angles, and an FA threshold that the algorithm does not consume; and
- streamline integration appends a proposed point before testing whether the
  direction field defines trackable tissue there, producing an out-of-domain
  terminal segment.

These are contract defects rather than documentation gaps. A plausible figure
cannot validate a physically mislabeled pipeline.

## Decision

### Ownership and dependency direction

`ritk-diffusion-scheme` owns acquisition metadata and format-neutral
validation. Format crates convert their external conventions into that one
contract. `ritk-diffusion` owns signal and orientation models.
`ritk-tractography` owns integration policy and returns Gaia polyline geometry.
Dependencies remain one-way:

```text
format crates -> diffusion-scheme -> spatial
diffusion     -> diffusion-scheme + Apollo SHT + Leto
tractography -> spatial + Gaia
```

The three bounded contexts remain separate crates because formats, numerical
models, and geometry have different dependency and publication surfaces.

### Diffusion weighting and frames

`DiffusionWeighting` is a validating transparent newtype around
`Quantity<f64, Time/Area>`. Aequitas dimension algebra supplies `Time/Area`;
the RITK newtype supplies the MRI-domain validity boundary. It stores canonical
SI seconds per square meter and converts explicitly at codec boundaries to and
from seconds per square millimeter. Construction rejects negative and
non-finite values. No raw constructor, dimensionless alias, or `Diffusivity`
compatibility name remains.

Gradient direction is private validated state. Scanner-facing values at or
below the shared 50 s/mm² baseline threshold are canonicalized to exact zero
weighting and direction after finite-input validation; a weighted sample above
that threshold has a finite unit vector. `GradientFrame` states whether that
vector is in image axes or physical LPS. Reorientation validates a finite
orthonormal rotation before applying it and preserves unit norm within a bound
derived from f64 arithmetic.

FSL b-vectors enter in image axes. DICOM standard gradient orientation enters
in patient coordinates, which RITK represents as physical LPS. NRRD applies
the declared measurement frame and space convention once. For a nominal NRRD
weighting `b_max` and raw gradient `g_i`, the effective value is

```text
b_i = b_max (||g_i|| / max_j ||g_j||)^2.
```

The stored direction is the normalized transformed vector. Missing modality,
nominal weighting, gradients, unsupported B-matrix/NEX compression, or the
frame information required for a non-identity conversion is an explicit
error; the reader does not invent zero weightings.

### DTI volume placement

`DiffusionMaps` retains the `GradientFrame` supplied by its
`GradientScheme`. `DtiVolume` is an image-index-space consumer, so its
constructor accepts only `ImageAxis` maps and performs the one convention
conversion from external `[column, row, depth]` components to RITK's
`[depth, row, column]` voxel-index components before nearest or interpolated
queries. It rejects `Lps` maps because a physical-to-index transform requires
image geometry that the volume does not own. This keeps codec provenance and
grid placement explicit and removes the example-local reorder.

### Analytical Q-ball reconstruction

The normalized signal is fitted in Apollo's real even-order spherical-
harmonic basis. Leto solves an augmented least-squares system whose diagonal
rows implement Laplace-Beltrami regularization. For coefficient degree `l`,
the penalty factor is `sqrt(lambda) l(l+1)`, so the normal-equation penalty is
`lambda l^2(l+1)^2`.

The signal coefficients are then transformed into the Q-ball ODF with the
Funk-Hecke relation

```text
psi_lm = 2 pi P_l(0) c_lm,
```

where `P_l` is the Legendre polynomial. Calling the result an ODF is therefore
conditional on this transform; an untransformed signal fit is not exposed as
one. Configuration construction validates an even degree of at least two, a
finite nonnegative regularization weight, a valid b0 threshold, and a
nonnegative shell tolerance. Estimation rejects non-finite signals, a
non-positive or non-finite baseline, and weighted samples outside one q-space
shell; samples from different shell radii are never treated as one angular
function. The result retains the acquisition coordinate frame, stores one
reusable basis and one contiguous coefficient allocation, and evaluates
directions in that frame. Grid evaluation returns flat contiguous samples with
explicit dimensions, not `Vec<Vec<_>>`.

### Deterministic streamline integration

`TractographyConfig` has private validated fields. It rejects a non-finite or
non-positive step, zero maximum steps, and a non-finite turn limit outside
`[0, 180]` degrees. Bidirectional selection is an enum rather than a Boolean.
The unused FA threshold is removed; tissue or anisotropy termination belongs
to the direction-field contract and is represented by absence at a queried
point.

The integrator validates every returned direction as finite and unit length.
It queries the proposed point before appending it, so an absent sample stops
at the last valid point. Direction sign is aligned to the preceding step before
the turn test. Polyline construction failures propagate through a typed public
error instead of silently dropping a streamline.

Euler integration is retained because this slice establishes the contract and
an analytical constant/curved-field oracle, not a clinical tractography claim.
Higher-order and probabilistic policies can be added through new strategy
implementations without changing the acquisition or geometry contracts.

## Consequences

- The recovered, previously uncommitted public surface changes before first
  publication; no compatibility layer is needed.
- Physical units, coordinate frames, and termination behavior are explicit and
  testable.
- Analytical Q-ball inputs are one-shell data within an explicit tolerance,
  and the resulting ODF preserves the gradient frame.
- NRRD inputs that the earlier parser interpreted incorrectly now fail rather
  than silently assigning false b-values.
- ODF evaluation avoids rebuilding basis metadata and dense spherical samples
  use one allocation.
- The supported reconstruction is analytical Q-ball imaging, not DTI, DKI,
  NODDI, constrained spherical deconvolution, or a clinical validation claim.

## Rejected alternatives

Keeping a dimensionless Aequitas quantity would make the type system certify a
false unit. Naming time-per-area `Diffusivity` would retain the reciprocal
physical meaning. Adding the missing alias upstream is unnecessary because
Aequitas already supplies compile-time dimension division and RITK needs a
domain-validating wrapper regardless.

Treating NRRD's nominal value as a per-volume list contradicts the established
DWI key/value convention. Defaulting absent metadata to b0 invents acquisition
facts. Calling the fitted diffusion signal an ODF without the Funk-Radon
transform contradicts analytical Q-ball reconstruction. Retaining the unused
FA field or silently discarding polyline errors would preserve misleading API
state.

## Verification

Format and value-semantic tests cover valid single- and multi-shell schemes,
frame conversion, gradient scaling, volume order, malformed/non-finite metadata,
construction invariants, and rotation norm preservation. ODF tests use an
independently synthesized single-tensor signal and assert antipodal symmetry,
isotropic behavior, finite residual reporting, shell tolerance, and a
full-sphere dominant-axis angular bound. Tractography tests cover straight and
analytical curved fields, bidirectional joining, tissue boundaries, turn
limits, invalid directions, configuration errors, and the exact bounded path
length.

The runnable book example uses a deterministic analytical fiber phantom. It
asserts every displayed metric before generating the committed SVG; CI
regenerates and diffs that artifact.

## References

- Stejskal and Tanner, "Spin Diffusion Measurements: Spin Echoes in the
  Presence of a Time-Dependent Field Gradient," *Journal of Chemical Physics*
  42, 288-292 (1965), DOI 10.1063/1.1695690:
  <https://doi.org/10.1063/1.1695690>
- Descoteaux, Angelino, Fitzgibbons, and Deriche, "Regularized, Fast, and
  Robust Analytical Q-Ball Imaging," *Magnetic Resonance in Medicine* 58,
  497-510 (2007), analytical reconstruction and Laplace-Beltrami
  regularization: <https://doi.org/10.1002/mrm.21277>
- DICOM PS3.3, Section C.8.13.5.9, MR Diffusion Macro, diffusion weighting
  units and patient-relative gradient orientation:
  <https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_c.8.13.5.9.html>
- NA-MIC, "DTI NRRD format," DWI key/value, gradient scaling, and measurement
  frame conventions:
  <https://www.na-mic.org/wiki/NAMIC_Wiki%3ADTI%3ANrrd_format>
- Kindlmann, "A self-contained explanation of image orientation and the
  measurement frame" (2010), Sections 2-3:
  <https://people.cs.uchicago.edu/~glk/unlinked/nrrd-iomf.pdf>
- Mori, Crain, Chacko, and van Zijl, "Three-Dimensional Tracking of Axonal
  Projections in the Brain by Magnetic Resonance Imaging," *Annals of
  Neurology* 45, 265-269 (1999), DOI
  10.1002/1531-8249(199902)45:2%3C265::AID-ANA21%3E3.0.CO;2-3.

## Revision history

- 2026-07-31: Initial accepted decision for FEAT-686-01.
- 2026-08-20: Revised the accepted frame decision for `FIX-DTI-VOLUME-FRAME`:
  retain the acquisition frame in `DiffusionMaps`, centralize ImageAxis grid
  placement in `DtiVolume`, and classify the public error-enum change as
  major. The driving evidence was the reusable volume and CLI path's inability
  to enforce the conversion while the book example carried its own adapter;
  focused permutation and LPS-rejection tests now pin the boundary.
- 2026-07-31: Reclassified the release impact from major to minor after
  `cargo-semver-checks` found no break against `d3d3d811`; the architecture is
  new and the existing-crate surfaces are additive.
