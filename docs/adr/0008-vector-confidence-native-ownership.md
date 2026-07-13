# ADR 0008: Vector confidence-connected native ownership

## Status

Accepted.

## Context

The vector confidence-connected algorithm already belongs to
`ritk-segmentation`, but its image wrapper widens every scalar channel into an
owned `f64` buffer. The Python binding first converts Coeus-native `PyImage`
storage into Burn images, invokes that wrapper, and converts the label image
back to Coeus. These transfers duplicate full images around a provider whose
Mahalanobis computation is otherwise independent of either tensor substrate.

## Decision

`ritk-segmentation` owns one validated filter and one statistical core. The core
reads channel samples through a monomorphized borrowed-source seam and performs
the documented double-precision mean, covariance, and Mahalanobis arithmetic
without materializing widened channel buffers. Covariance inversion delegates
to Leto's rank-revealing SVD and applies ITK's determinant policy. Legacy and
Coeus-native image entry points validate common geometry and call that core.
PyO3 passes borrowed `MoiraiBackend` image storage to the native entry point and
returns its Coeus-native label image.

Configuration becomes validated construction state. Invalid numeric
configuration, channel shapes, and samples return typed errors before indexing
or allocation. ITK intentionally ignores out-of-bounds seeds; an empty valid
seed set returns an empty mask. Seed-neighborhood statistics use zero-flux
boundary replication with full multiplicity, compressed to unique voxels and
weights so runtime remains bounded by image size for every representable radius.
The public free image wrapper and
mutable configuration shape are replaced rather than retained as forwarding
adapters.

References: [ITK vector confidence-connected implementation](https://github.com/InsightSoftwareConsortium/ITK/blob/v5.4.6/Modules/Segmentation/RegionGrowing/include/itkVectorConfidenceConnectedImageFilter.hxx),
[ITK covariance image function](https://github.com/InsightSoftwareConsortium/ITK/blob/v5.4.6/Modules/Core/ImageFunction/include/itkCovarianceImageFunction.hxx),
[ITK Mahalanobis membership function](https://github.com/InsightSoftwareConsortium/ITK/blob/v5.4.6/Modules/Numerics/Statistics/include/itkMahalanobisDistanceMembershipFunction.hxx).

## Consequences

The provider API is breaking and all in-tree callers migrate in the same
change. Native Python execution removes the two Burn image conversions and all
full-volume `f32`-to-`f64` staging. Statistical work buffers remain proportional
to channel count or voxel count as required by covariance and flood state.

Evidence combines construction-time validation, exact legacy/native
differential tests, deterministic SimpleITK mask comparison, installed-wheel
geometry/failure tests, and warning-denied package gates. This is type-level and
empirical evidence, not machine-checked proof.

## Alternatives rejected

- Adding a native forwarding wrapper around the legacy image API retains both
  full-image conversions and duplicate ownership.
- Maintaining separate legacy and Coeus algorithms permits statistical and
  singular-covariance semantics to drift.
- Keeping widened channel copies simplifies indexing but violates the
  allocation budget when the source is already CPU-addressable.
