# Benchmarking

The benchmarking chapter focuses on measuring host-core cost separately from boundary overhead. Many ritk algorithms are written as substrate-agnostic flat-buffer kernels with lightweight wrappers that reconstruct `ritk-image::Image` values or dispatch on a backend only once. A good benchmark therefore asks more than “how long did the function call take?” It compares release builds, warm-cache versus cold-cache behavior, and native host-core execution against per-image wrapper paths so regressions can be attributed to actual algorithmic work rather than avoidable allocations or metadata shuffling.

This is where Atlas integration becomes visible at the systems level. Coeus provides the image and backend abstractions, Moirai provides a parallel execution target, and Leto may appear indirectly when classical registration converts through array-based numerics. Benchmarking keeps those layers honest by showing when the same public API is effectively zero-cost and when a boundary crossing introduces measurable overhead that deserves a dedicated optimization.

## Example Summary

| Example | Status | Focus |
| --- | --- | --- |
| [Gradient Recursive Gaussian Benchmark](examples/bench_gradient_rg.md) | Available | Compares optimized buffer-level execution against per-`Image` wrapper passes. |
| Registration microbench suite | Planned | Extend the same method to metric evaluation, resampling, and optimizer loops. |
