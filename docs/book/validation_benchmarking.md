# Validation and Benchmarking

Validation and performance are separate acceptance axes. A faster run is not an
accuracy result, and a lower metric is not proof of correct geometry.

Registration validation should combine:

- shape, spacing, origin, and direction checks;
- metric values before and after;
- overlap measures when labels exist;
- convergence state and iteration budget; and
- a labeled overlay plus an input-to-output change map.

Benchmarking should measure loading, preprocessing, metric evaluation,
resampling, and optimizer loops separately. This makes a regression
attributable to an algorithm rather than a hidden boundary copy.

## Example Summary

| Example | Status | Focus |
| --- | --- | --- |
| [Geometry Validation](examples/geometry_check.md) | Available | Baseline spatial-contract check before trusting a registration result. |
| [Validation Suite](examples/validation_suite.md) | Available | Aggregate geometry, metric, overlap, and convergence checks. |
| [Gradient Recursive Gaussian Benchmark](examples/bench_gradient_rg.md) | Available | Performance methodology for reusable filter kernels. |

The CT/MR example records both the identity-to-registered metric change and the
visible resampling change so those claims remain distinguishable.
