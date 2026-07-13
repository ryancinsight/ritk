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
the documented double-precision mean, covariance, inversion, and Mahalanobis
arithmetic without materializing widened channel buffers. Legacy and
Coeus-native image entry points validate common geometry and call that core.
PyO3 passes borrowed `MoiraiBackend` image storage to the native entry point and
returns its Coeus-native label image.

Configuration and seeds become validated construction state. Invalid numeric
configuration, channel shapes, samples, and seed coordinates return typed
errors before indexing or allocation. The public free image wrapper and mutable
configuration shape are replaced rather than retained as forwarding adapters.

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
