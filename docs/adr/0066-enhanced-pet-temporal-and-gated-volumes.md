# ADR 0066: Reconstruct temporal and gated Enhanced PET volumes

- Status: Proposed
- Class: [major]
- Date: 2026-11-19

Group frames by the nonspatial Dimension Index Values in each single
Enhanced PET Part 10 object. Use the spatial index to reject duplicate
planes, then sort each volume by projected patient position. Use source
frame indices for pixel decode, display transforms, and quantitative
calibration. Cardiac and respiratory trigger delays carry time, and cycle
phase and respiratory amplitude carry ratios. Depend on explicit dimension
indices for dynamic and gated input. Stacks may have distinct spatial
geometry. A combined decoded-byte ceiling covers all reconstructions.

The volume-set API preserves independent spatial buffers and the viewer
swaps the selected display, geometry, and SUV atomically. Same-geometry
changes preserve the current view and annotations. The scalar-volume API
requires exactly one reconstruction. Combining dimensions into the slice
axis would mix unrelated quantitative frames. Regression tests interleave
source frames and compare calibrated voxels across time and across gates.
