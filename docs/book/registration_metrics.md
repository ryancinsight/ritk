# Registration Metrics

Registration quality in ritk is driven by two families of similarity metrics. The differentiable path in `ritk-registration::metric` exposes native MSE, NCC, LNCC, and NGF evaluators plus autodiff traits used by gradient-based optimization. The classical path complements that with mutual information for multi-modal alignment and sealed translation metrics such as mean squared difference and normalized cross-correlation. This chapter explains when each metric is appropriate: MSE and NCC for same-modality volumes, LNCC for local contrast robustness, NGF for edge-driven alignment, and mutual information when intensity relationships are not linear across modalities.

Atlas integration matters because the same public image boundary feeds both implementations. Coeus-backed tensors power autodiff metrics and learning-based registration, while classical mutual-information evaluation operates on Leto arrays after an explicit native conversion. The metric layer therefore forms the bridge between file and geometry correctness and optimizer behavior: if spatial metadata is wrong, every metric will look unreliable; if the metric is mismatched to the modality pair, the optimizer may converge to the wrong transform even when backend execution is correct.

## Example Summary

| Example | Status | Focus |
| --- | --- | --- |
| [Registration Comparison Figure](examples/registration_compare_figure.md) | Available | Visual comparison of identity, classical MI, and reference alignment behavior. |
| [Validation Suite](examples/validation_suite.md) | Planned | Pair metric values with geometry and overlap checks to interpret whether a registration really improved. |
