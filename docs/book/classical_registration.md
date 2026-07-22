# Classical Registration

The classical registration stack in ritk collects deterministic, non-ML alignment methods under one API. `ritk-registration::classical` exposes landmark-based rigid registration via Kabsch SVD, translation search with sealed metric types, mutual-information driven rigid and affine optimization, and temporal synchronization helpers for paired acquisition streams. These algorithms are deliberately CPU-centric and geometry-conscious: the code assumes that fixed and moving images have already crossed the `ritk-image` boundary with correct origin, spacing, and direction, then performs the actual solve in a form that is easy to audit and validate.

Under Atlas, this chapter is where Coeus and Leto meet most visibly. Images arrive as Coeus-backed `Image<f32, B, 3>` values, then `image_to_leto_volume` converts contiguous voxel storage into Leto arrays for the classical engine. After optimization, `leto_volume_to_image` restores the result in the original physical frame so later filters, writers, and benchmarks still operate on the standard ritk image contract. That division keeps format handling, tensor execution, and classical numerics separated without fragmenting the user-facing workflow.

## Example Summary

| Example | Status | Focus |
| --- | --- | --- |
| [Geometry Validation](examples/geometry_check.md) | Available | Confirms that file-space geometry is correct before classical registration consumes it. |
| [Deep Learning Registration](examples/dl_registration.md) | Available | Useful contrast with the differentiable path that shares the same image boundary. |
| [Deep Learning Training](examples/dl_train.md) | Available | Shows how Coeus-native training sits alongside, rather than replacing, the classical stack. |
