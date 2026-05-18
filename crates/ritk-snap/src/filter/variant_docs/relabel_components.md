Re-label connected components in order of decreasing size
(ITK `RelabelComponentImageFilter` parity).

Accepts a label image (output of `ConnectedComponents`) and reassigns
component indices so that label 1 = largest component, label 2 = second
largest, and so on. Components with fewer than `minimum_object_size`
voxels are removed (set to 0.0 background).

# Invariants
- `minimum_object_size = 0` (default): retains all components.
- Background voxels (0.0) remain 0.0.
- Output label 1 has the most voxels in the input.
