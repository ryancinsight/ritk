# Backend Dispatch

Backend dispatch in ritk is designed to be explicit in the type system and cheap at runtime. `ritk-image` re-exports Atlas backends such as `SequentialBackend` and `MoiraiBackend`, while `ritk-io` binds readers and writers to a chosen backend through types like `PngReader<B>`, `MetaImageWriter<B>`, and `VtkReader<B>`. Deeper in the stack, many filters expose a native host-core path and then reconstruct the same `ritk-image::Image` boundary, so callers get one public API while the compiler monomorphizes concrete backend implementations underneath.

That matters for Atlas because Coeus is not just a storage detail: it is the execution substrate that lets the same algorithm run deterministically on the sequential backend or in parallel on Moirai. The chapter therefore covers where dispatch is compile-time, where a host extraction is unavoidable, and how the unified `read_image_native` and `write_image_native` helpers keep format inference separate from compute choice.

## Example Summary

| Example | Status | Focus |
| --- | --- | --- |
| [Gradient Recursive Gaussian Benchmark](examples/bench_gradient_rg.md) | Available | Highlights wrapper overhead versus shared host-core execution across backends. |
| [Deep Learning Registration](examples/dl_registration.md) | Available | Representative Coeus-native pipeline where backend choice affects execution, not API shape. |
