# Registration Metrics

Metric selection follows the intensity relationship, not the optimizer.

| Pair | Starting metric |
| --- | --- |
| Same modality, similar calibration | MSE or NCC |
| Same modality, local contrast variation | LNCC |
| Edge-driven alignment | NGF |
| CT/MR or another non-linear intensity relationship | Mutual information |

The differentiable path exposes native MSE, NCC, LNCC, and NGF evaluators. The
classical path complements them with mutual information and sealed translation
metrics. Both paths consume the RITK image geometry contract; the classical
path performs an explicit conversion to Leto arrays at its numeric boundary.

Evaluate the identity transform before optimization. A post-registration metric
is meaningful only when the identity value, transform convention, sampled
frame, and optimizer state are recorded beside it.

## Example Summary

| Example | Status | Focus |
| --- | --- | --- |
| [CT/MR Mutual-Information Registration](examples/registration_compare_figure.md) | Available | Visual comparison of identity, classical MI, and reference alignment behavior. |
| [Validation Suite](examples/validation_suite.md) | Available | Pair metric values with geometry and overlap checks to interpret whether a registration really improved. |
