Multi-class Otsu threshold segmentation
(ITK `OtsuMultipleThresholdsImageFilter` parity).

Finds K − 1 thresholds that maximise total between-class variance, then
assigns each voxel a class label in {0, 1, …, K−1} as f32.

# Invariants
- `num_classes = 2` degenerates to standard single-threshold Otsu.
- Uniform input → all thresholds equal; output is all-zero.
- Output values lie in `{0.0, 1.0, …, (num_classes − 1).0}`.
