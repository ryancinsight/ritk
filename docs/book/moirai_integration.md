# Moirai Execution Integration

Moirai is ritk's Atlas-native parallel execution backend, surfaced through the `ritk-image` re-exports of `MoiraiBackend` and the broader Coeus backend traits. The important design point is that Moirai does not require a separate image type or parallel-only API: readers, writers, filters, and registration code are generally parameterized over a backend, so the same `Image<f32, B, 3>` contract can be instantiated on either the sequential backend for deterministic CPU work or Moirai for broader throughput. This chapter covers where that substitution is seamless and where performance-sensitive code still extracts host buffers intentionally.

In practice, Moirai matters most once data is already inside the Atlas pipeline. File boundaries in `ritk-io` construct backend-bound images, compute-heavy filters reuse the same substrate-agnostic kernels, and Coeus-native registration or learning loops can execute on the selected backend without rewriting example logic. The net effect is a clean separation: the book can describe algorithms once, while Moirai supplies the parallel execution strategy underneath.

## Example Summary

| Example | Status | Focus |
| --- | --- | --- |
| [Gradient Recursive Gaussian Benchmark](examples/bench_gradient_rg.md) | Available | Natural place to compare sequential wrapper cost with parallel backend execution strategy. |
| [Deep Learning Registration](examples/dl_registration.md) | Available | Coeus-native registration flow that can benefit from backend substitution without API churn. |
