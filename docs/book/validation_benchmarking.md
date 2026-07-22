# Validation and Benchmarking

Registration work is only trustworthy when ritk can prove both correctness and performance. The validation side of `ritk-registration` defines `ShapeValidation`, `NumericalCheck`, `ValidationConfig`, and quality summaries such as `RegistrationQualityMetrics`, covering fiducial error, target error, mutual information, correlation, convergence state, and iteration counts. The benchmarking side asks a different question: how much of total runtime belongs to file I/O, resampling, metric evaluation, or wrapper overhead, and how does that change across backends or optimizer families? This chapter treats those as one discipline rather than two separate chores.

In Atlas terms, validation usually crosses boundaries: geometry may come from `ritk-io`, tensors may be evaluated in Coeus, and classical metrics may run on Leto arrays. Benchmarking then measures whether that boundary design is paying off or adding avoidable copies and dispatch overhead. The goal is not just to report a fast number, but to relate accuracy, convergence, and execution cost so users can choose the simplest pipeline that still meets their clinical or research requirement.

## Example Summary

| Example | Status | Focus |
| --- | --- | --- |
| [Geometry Validation](examples/geometry_check.md) | Available | Baseline spatial-contract check before trusting any registration result. |
| [Validation Suite](examples/validation_suite.md) | Planned | Planned aggregate report for geometry metrics, overlap metrics, and convergence summaries. |
| [Gradient Recursive Gaussian Benchmark](examples/bench_gradient_rg.md) | Available | Performance methodology that also informs how registration helpers should be profiled. |
