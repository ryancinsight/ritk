# Leto Operations Integration

Leto underpins the classical numeric side of ritk. The registration crate uses `leto::Array1`, `Array2`, and `Array3` for temporal synchronization, landmark registration, mutual-information evaluation, and resampling-friendly CPU data structures; `ritk-spatial` also builds fixed-size point, vector, and direction primitives on top of Leto matrix and vector types. This chapter explains why that integration exists: Leto offers deterministic, auditable array math for algorithms that benefit more from straightforward CPU numerics than from autodiff or accelerator-style execution.

The Atlas boundary remains deliberate. RITK does not expose Leto as the main image contract; instead, images stay in `ritk-image` as Coeus-backed tensors until a classical algorithm explicitly converts them with helpers such as `image_to_leto_volume`. After the solve, results are reconstructed back into the original physical frame so later writers, filters, and validation tools still operate on the same public image abstraction. That separation keeps Leto powerful internally without leaking storage policy into the rest of the toolkit.

## Example Summary

| Example | Status | Focus |
| --- | --- | --- |
| [Registration Comparison Figure](examples/registration_compare_figure.md) | Available | Classical mutual-information alignment running on the Leto-backed engine. |
| [Geometry Validation](examples/geometry_check.md) | Available | Verifies the image-geometry assumptions that must hold before conversion to Leto volumes. |
