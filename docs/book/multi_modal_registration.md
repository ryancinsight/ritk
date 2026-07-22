# Multi-modal Registration

Multi-modal registration in ritk focuses on the practical problem that CT, MR, PET, dose, and derived label maps do not share a simple intensity relationship. The toolkit therefore leans on modality-robust objectives such as mutual information, LNCC, and NGF, plus preprocessing steps like windowing, smoothing, and resampling that normalize inputs before optimization starts. This chapter frames multi-modal registration as a pipeline problem: choose a metric that tolerates contrast differences, ensure spatial metadata is trustworthy, and compare alignment visually as well as numerically.

Atlas integration again spans both compute families. Classical mutual-information registration operates on Leto volumes after explicit conversion, while Coeus-native metrics and differentiable models can evaluate edge- or correlation-based losses directly on the tensor-backed image representation. Because both paths share the same `ritk-image` geometry contract, ritk can compare classical and learned multi-modal workflows without changing how data is loaded, validated, or written back out.

## Example Summary

| Example | Status | Focus |
| --- | --- | --- |
| [Registration Comparison Figure](examples/registration_compare_figure.md) | Available | CT-to-MR overlay that makes multi-modal alignment success or failure visually obvious. |
| [Validation Suite](examples/validation_suite.md) | Planned | Planned companion for overlap and geometry checks after multi-modal alignment. |
