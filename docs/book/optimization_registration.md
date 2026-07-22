# Optimization and Registration

Metrics only become registration algorithms once ritk couples them to an optimizer. On the differentiable side, `ritk-registration::metric::autodiff` supplies a reusable `gradient_descent` driver configured by `GradientDescentConfig`, rebuilding a transform from trainable Coeus `Var`s each iteration and stepping parameters with the native SGD helper. On the classical side, `ImageRegistration` uses `ClassicalConfig` and `MutualInformationMetric` to run deterministic CPU optimization over rigid or affine parameters. This chapter covers that seam: parameterization, iteration budgets, tolerances, step sizes, and why optimizer behavior must be read together with the chosen similarity metric.

Atlas integration is split but coherent. Coeus provides the autodiff graph, tensor execution, and backend flexibility for learning-style registration loops, while Leto supports the classical numeric path where predictable CPU array behavior is preferred. RITK keeps those implementation details behind a common image boundary so callers can reason about transforms, convergence, and output geometry without rewriting file I/O or preprocessing around each optimizer family.

## Example Summary

| Example | Status | Focus |
| --- | --- | --- |
| [Deep Learning Registration](examples/dl_registration.md) | Available | End-to-end differentiable optimization of rigid parameters with Coeus autodiff. |
| [Deep Learning Training](examples/dl_train.md) | Available | Extends the same optimization ideas to a learned registration model and training loop. |
| [Registration Comparison Figure](examples/registration_compare_figure.md) | Available | Shows the practical effect of optimizer choice on final alignment quality. |
