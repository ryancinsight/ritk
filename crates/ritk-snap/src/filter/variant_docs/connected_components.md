Connected-component labeling filter (ITK `ConnectedComponentImageFilter` parity).

Labels each connected group of foreground voxels with a unique integer
index. Output is an f32 label image where:
- `0.0` = background (pixel value equals `background_value`);
- `1.0`, `2.0`, ... = component labels in scan order.

# Connectivity
- 6-connectivity (face adjacency): standard 3-D default, matching ITK.
- 26-connectivity (face + edge + corner): set `connectivity_26 = true`.

# Invariants
- All-background input → output is all-zero.
- N foreground blobs separated by background → N unique non-zero labels.
