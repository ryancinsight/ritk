# ADR 0007: Isolated-connected native ownership

## Status

Accepted.

## Context

`IsolatedConnectedImageFilter` searches one endpoint of an inclusive intensity
band until a region connected from the retained seed excludes the isolation
seed. The prior implementation rebuilt legacy images for each probe, exposed
unvalidated mutable state, and forced Python through a Burn conversion.

## Decision

`ritk-segmentation` owns one validated, flat `f32` implementation. Its
generation-tagged flood workspace reuses the visited map and queue over binary
search probes. `IsolatedConnectedConfig` selects the varying endpoint with
`IsolationThreshold`; the filter returns the final image and explicit failure
status if the band cannot contain seed one while excluding seed two. Legacy and
Coeus-native images call the same core. PyO3 calls the native path over
`MoiraiBackend` storage.

The bisection order follows ITK's source implementation: upper search retains
the last separating lower bound, while lower search retains the last separating
upper bound. Search arithmetic is `f64`; sample membership remains native
`f32`.

Reference: [ITK `IsolatedConnectedImageFilter` implementation](https://github.com/InsightSoftwareConsortium/ITK/blob/master/Modules/Segmentation/RegionGrowing/include/itkIsolatedConnectedImageFilter.hxx).

## Consequences

The old public-field construction is removed. Callers construct validated
configuration. They receive the final output and explicit thresholding-failure
status when a band cannot isolate the seeds. Allocation reuse is structural;
exact legacy/native and SimpleITK cases provide empirical differential evidence.

## Alternatives rejected

- Per-probe image reconstruction: duplicates the provider boundary and adds
  avoidable allocation.
- Separate legacy and native algorithms: allows threshold semantics to drift.
- Treating an impossible band as an execution error or an unmarked empty
  segmentation: contradicts ITK's thresholding-failure contract.
