# Zero-copy I/O

Zero-copy in ritk means minimizing ownership churn at the boundary, not pretending that every file decode is literally allocation-free. The `ritk-image::Image` contract keeps voxel storage in a Coeus tensor, exposes checked accessors such as `data_slice()` for contiguous CPU views, and lets writers extract host data only when a format encoder truly needs it. Supporting crates follow the same pattern: `ritk-morphology::StructuringElement::offsets()` returns a borrowed slice, and many filter kernels operate on flat buffers so they do not repeatedly rebuild tensor structures in the hot path.

Atlas integration sharpens the distinction between cheap views and necessary copies. Staying inside the Coeus-backed image pipeline often means metadata moves without touching the voxels, but some seams necessarily materialize data: JPEG encode or decode crosses a compression boundary, and classical registration widens `f32` image values into `f64` Leto volumes before optimization. This chapter is about recognizing those intentional costs, keeping them localized, and preserving a single image object through as much of the workflow as possible.

## Example Summary

| Example | Status | Focus |
| --- | --- | --- |
| [Geometry Validation](examples/geometry_check.md) | Available | Reads once, then inspects geometry without repeatedly rebuilding intermediate image objects. |
| [Gradient Recursive Gaussian Benchmark](examples/bench_gradient_rg.md) | Available | Shows the benefit of extracting once and reusing flat-buffer computation. |
