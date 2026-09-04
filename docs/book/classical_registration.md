# Classical Registration

The classical registration stack in ritk collects deterministic, non-ML alignment methods under one API. `ritk-registration::classical` exposes landmark-based rigid registration via Kabsch SVD, translation search with sealed metric types, mutual-information driven rigid and affine optimization, and temporal synchronization helpers for paired acquisition streams. These algorithms are deliberately CPU-centric and geometry-conscious: the code assumes that fixed and moving images have already crossed the `ritk-image` boundary with correct origin, spacing, and direction, then performs the actual solve in a form that is easy to audit and validate.

Rigid-search transforms use physical `[z,y,x]` millimetres. Native image world
points use metadata `[x,y,z]`. Use
`rigid_physical_affine_to_native` when a classical search result must enter a
native resampler; it applies the axis permutation to both sides of the linear
part and to translation. `index_affine_to_physical` is only for transforms
whose input contract is explicitly index space.

Under Atlas, this chapter is where Coeus and Leto meet most visibly. Images arrive as Coeus-backed `Image<f32, B, 3>` values, then `image_to_leto_volume` converts contiguous voxel storage into Leto arrays for the classical engine. After optimization, `leto_volume_to_image` restores the result in the original physical frame so later filters, writers, and benchmarks still operate on the standard ritk image contract. That division keeps format handling, tensor execution, and classical numerics separated without fragmenting the user-facing workflow.

For local CT/MR soft-tissue refinement, `MindSscFixedPrep` can be evaluated by
the same rigid-search objective closure. Prepare it once before search. Its
selected centers and denominator remain fixed across poses, while moving patch
values are sampled on demand. The optimizer bounds and search policy are
unchanged.

## Example Summary

| Example | Status | Focus |
| --- | --- | --- |
| [Geometry Validation](examples/geometry_check.md) | Available | Confirms that file-space geometry is correct before classical registration consumes it. |
| [Deep Learning Registration](examples/dl_registration.md) | Available | Useful contrast with the differentiable path that shares the same image boundary. |
| [Deep Learning Training](examples/dl_train.md) | Available | Shows how Coeus-native training sits alongside, rather than replacing, the classical stack. |

Deformable registration has its own chapter because it solves for a spatially
varying displacement field rather than a finite-dimensional rigid or affine
transform. The deterministic Thirion Demons example is documented in
[Deformable Registration](demons_registration.md).
