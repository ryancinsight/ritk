# SimpleITK cmake-coverage: investigated exclusions

Per-filter reasons the **8 still-uncovered** SimpleITK cmake filters are not booked
as ritk parity. Each was probed against sitk and found to have a genuine
algorithmic / determinism / type-system difference, or a binding-surface blocker —
not a fixable bit-exact composition. No approximate or partial-parameter parity is
booked as coverage (integrity: no fabricated parity).

This list is kept in sync with `SITK_CMAKE_COVERAGE.md` "Not yet covered". Filters
that were once here and have since shipped bit/float-exact or functionally verified
have been removed.

## Absent from this SimpleITK python package build (3)

- **CoherenceEnhancingDiffusion**: ritk implements `filter.coherence_enhancing_diffusion`, but sitk lacks `sitk.CoherenceEnhancingDiffusion` in this build (no oracle exists in the Python package).
- **ContourExtractor2D**: Outputs polylines/coordinates rather than a comparable dense image, and is absent from this SimpleITK Python build.
- **LevelSetMotionRegistration**: SimpleITK has `LevelSetMotionRegistrationFilter` but lacks the procedural `LevelSetMotionRegistration` function in Python (no oracle exists).

## Unimplemented / non-bit-exact reproducible (5)

- **AntiAliasBinary**: Iterative PDE narrow-band solver whose per-step floating-point accumulation compounds and is not bit-exact. Unimplemented in ritk.
- **CannySegmentationLevelSet**: Iterative level-set solver. Unimplemented in ritk.
- **IsolatedWatershed**: Binary-searches the `Level` of `itk::WatershedImageFilter` (Vincent–Soille hierarchical segmentation — a saliency merge-tree over flooded basins). The labeling and merge order are implementation-specific, so different watershed engines cannot reproduce it.
- **PatchBasedDenoising**: Iterative patch search. Unimplemented in ritk.
- **ScalarChanAndVeseDenseLevelSet**: Iterative level-set solver. Ritk's `chan_vese` is a different algorithm and does not correspond to sitk's.
