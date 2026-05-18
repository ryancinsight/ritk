Contrast Limited Adaptive Histogram Equalization (CLAHE).

Applies the Zuiderveld (1994) algorithm independently to each axial
slice of the volume. Each slice is divided into `tile_grid_size[0] ×
tile_grid_size[1]` tiles; per-tile CDFs are clip-limited and then
bilinearly interpolated to compute the per-pixel mapping.

# Invariants
- `tile_grid_size[i] ≥ 1`; values are clamped to 1 on construction.
- `clip_limit ≥ 1.0` (1.0 = no clipping; clip threshold equals the
  uniform-distribution count).
- Output values lie in `[v_min_slice, v_max_slice]` for each slice.
