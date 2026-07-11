Selectable image filters exposed through the viewer core.

Each variant maps 1-to-1 onto a concrete filter implementation in
`ritk_filter`. Dispatch in `ViewerCore::apply_filter` is exhaustive
and concrete — no trait objects are used.

# Variant invariants
- `BedSeparation`: body_threshold and background_threshold must be valid
  Hounsfield-range values; outside_value must be representable as f32.
- `Gaussian`: sigma > 0.0 (zero sigma is a no-op but not an error; the
  underlying filter skips dimensions where sigma ≤ 1e-6).
- `Median`: radius = 0 is identity (each voxel is its own sole neighbour).
