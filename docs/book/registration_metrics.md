# Registration Metrics

Metric selection follows the intensity relationship, not the optimizer.

| Pair | Starting metric |
| --- | --- |
| Same modality, similar calibration | MSE or NCC |
| Same modality, local contrast variation | LNCC |
| Edge-driven alignment | NGF |
| CT/MR with preserved local anatomy | MIND-SSC |
| Multimodal global capture | Mutual information |

The differentiable path exposes native MSE, NCC, LNCC, and NGF evaluators. The
classical path complements them with mutual information, packed MIND-SSC, and
sealed translation metrics. Both paths consume the RITK image geometry
contract; the classical path performs an explicit conversion to Leto arrays at
its numeric boundary.

## Packed MIND-SSC

MIND describes anatomy by comparing nearby patches within each image instead
of comparing CT and MR intensities directly. RITK's SSC pattern uses the 12
cross-axis pairs among six axial neighbours. A 3×3×3 patch produces 12 summed
squared patch distances; local mean distance normalizes contrast, and the
minimum-distance subtraction makes the largest response one. Five unary bits
encode six levels (0–5) and pack all 12 components into the low 60 bits of one `u64`.
XOR/popcount is therefore the exact L1 distance between quantized descriptors.

`MindSscFixedPrep` stores packed descriptors only for a deterministic fixed set
of complete-support centers. The default cap is 8,192. It follows the
Hoeffding sample requirement of 6,623 for a bounded `[0,1]` mean loss with
±0.02 error at 99% confidence under uniform random sampling, rounded up to the
next power of two. RITK fixes the hash seed and stratifies in physical space for
reproducible coverage. That statistical design explains the cap; one fixed
realization does not constitute population or clinical validation.

During each pose evaluation, RITK maps every required fixed patch point into
moving physical space and trilinearly samples it. Support inside ITK's
half-voxel field uses replicate-edge interpolation; support outside is zero
background and remains in the fixed denominator. This preserves a
pose-invariant objective while keeping persistent memory proportional to
selected centers, not image voxels.

Heinrich et al. define MIND in section 3 of the 2012 paper
(<https://doi.org/10.1016/j.media.2012.05.008>) and the compact 12-component SSC
pattern in section 3.1 of the 2013 paper
(<https://doi.org/10.1007/978-3-642-40811-3_24>).

Evaluate the identity transform before optimization. A post-registration metric
is meaningful only when the identity value, transform convention, sampled
frame, and optimizer state are recorded beside it.

## Example Summary

| Example | Status | Focus |
| --- | --- | --- |
| [CT/MR Mutual-Information Registration](examples/registration_compare_figure.md) | Available | Visual comparison of identity, classical MI, and reference alignment behavior. |
| [Validation Suite](examples/validation_suite.md) | Available | Pair metric values with geometry and overlap checks to interpret whether a registration really improved. |
