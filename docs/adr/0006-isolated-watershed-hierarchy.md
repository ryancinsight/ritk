# ADR 0006: Isolated watershed hierarchy ownership

- Status: Accepted
- Date: 2026-07-12
- Change class: major

## Context

`IsolatedWatershed` exposes ITK's `threshold`, `upper_value_limit`, and
`isolated_value_tolerance`, but the current gradient-descent implementation
ignores all three. ITK first computes gradient magnitude, constructs a watershed
segment hierarchy at `threshold`, and binary-searches the hierarchy level between
`threshold` and `upper_value_limit` until the two seed labels separate within
`isolated_value_tolerance`. It then emits only the two selected segment labels.

The incumbent finest-basin shortcut happens to match one symmetric fixture but
cannot satisfy parameter sensitivity or general ITK behavior. Morphological
watershed with an h-minima level is also not equivalent: differential probes
change boundary ownership and seed regions.

## Decision

RITK will own one native-precision watershed hierarchy implementation in the
watershed bounded context:

1. Clamp the gradient-magnitude lower tail at
   `min + threshold * (max - min)`, matching ITK's segmenter threshold.
2. Build the plateau-aware initial segments and their minimum boundary edges.
3. Generate the dynamic directed merge hierarchy used by ITK: a segment floods
   across its lowest edge at saliency `edge_height - segment_min`; after a merge,
   edge lists and the surviving segment minimum are consolidated before its next
   candidate is queued.
4. Relabel the initial segmentation through hierarchy merges whose saliency is
   at most `level * maximum_hierarchy_saliency`.
5. Reproduce ITK's binary-search update and lower-level fallback exactly, then
   map the two seed segments to configured output labels and all other segments
   to background.

The hierarchy core is flat-buffer Rust and serves legacy and Coeus-native image
boundaries. Configuration and seed coordinates are validated before any image
indexing. No ITK runtime dependency, compatibility adapter, or downstream
approximation is introduced.

## Rejected alternatives

- Keep gradient-descent basins and ignore hierarchy parameters: violates input
  sensitivity and the public contract.
- Substitute marker-less morphological watershed levels: close on some fixtures
  but not label-exact on adversarial random reliefs.
- Bind ITK through FFI: violates Rust ownership and adds a runtime dependency.

## Verification

- Exact differential outputs against SimpleITK for fixed 2-D and 3-D fixtures.
- Parameter-sensitivity cases for threshold, upper limit, and tolerance.
- Exact native/legacy equality and physical-geometry preservation.
- Positive, negative, boundary, non-finite, seed-range, and cardinality tests.

Evidence is differential and empirical unless a stronger type-level invariant is
identified. The decision is revisited only if current ITK source changes the
hierarchy or binary-search contract.
