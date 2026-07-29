# Benchmarking

Benchmarking keeps algorithm cost separate from file I/O, image extraction,
metadata reconstruction, and dispatch. Many filters operate on flat buffers,
so a useful benchmark compares the public image call with the reusable buffer
core and records where any boundary copy occurs.

Use release builds, fixed inputs, and a benchmark closure that returns the
computed result. Record median and confidence intervals. A benchmark that only
checks that a call succeeds cannot detect a dead computation or wrong value.

## Registration measurements

Registration uses the same separation. Time image loading and preprocessing
outside the timed region, then measure metric evaluation, resampling, and
optimizer steps independently. Keep fixed and moving images and the optimizer
configuration in the benchmark input. Report accuracy and runtime separately.

The repository's runnable benchmark lane is the source of truth for command
names and budgets. Build it in release mode and run the smallest case before
collecting a full baseline.

## Example Summary

| Example | Status | Focus |
| --- | --- | --- |
| [Gradient Recursive Gaussian Benchmark](examples/bench_gradient_rg.md) | Available | Compares optimized buffer-level execution against image-wrapper passes. |
