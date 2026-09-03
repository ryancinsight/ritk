# Multi-modal Registration

CT, MR, PET, dose, and derived maps do not share a simple intensity
relationship. Multi-modal registration therefore needs a modality-robust
objective, trustworthy physical metadata, and visual validation.

The minimum workflow is:

1. load both volumes and inspect their physical frames;
2. choose mutual information, LNCC, or NGF rather than assuming MSE is valid;
3. resample the moving image in the fixed frame;
4. compare identity and candidate transforms; and
5. retain a numerical change map and a labeled overlay.

## Histogram estimation and overlap

`MutualInformationMetric` makes two decisions explicit. Fixed and moving
modalities receive separate intensity ranges, so a CT Hounsfield-unit window is
not interpreted as an MR signal range. `HistogramEstimator::Discrete` is the
default when exact discrete-information identities matter.
`HistogramEstimator::MovingLinearPartialVolume` keeps the fixed sample in one
bin while distributing the resampled moving intensity between adjacent bins.
The resulting objective changes continuously inside each moving-intensity bin,
which is the useful direction for transform optimization.

Masked sample evaluation accepts borrowed slices and an optional fixed mask.
For optimization, keep that mask and the accepted moving-field support fixed
across candidate poses. Recomputing the support from each candidate lets a bad
pose improve its score by cropping difficult anatomy. Report overlap separately
and reject candidates that violate the application-derived support floor.

For CT/MR soft tissue, a bounded two-stage search is preferable to a fitted
weighted sum: NMI captures the multimodal basin, then NGF refines edge
orientation inside one terminal NMI cell. A global NGF search can lock onto a
remote skull or air boundary.

The CT/MR fixture uses moving-linear partial-volume NMI plus a reference transform so the
example is reproducible rather than optimizer-seed dependent. The registered
overlay is visibly different from identity because red and green fringes expose
residual misalignment, while the MR change panel exposes where resampling
actually changed the sampled values.

## Example Summary

| Example | Status | Focus |
| --- | --- | --- |
| [CT/MR Mutual-Information Registration](examples/registration_compare_figure.md) | Available | Labeled RIRE CT-to-MR overlays with identity, registered, and MR-change diagnostic panels. |
| [Validation Suite](examples/validation_suite.md) | Available | Geometry, metric, overlap, convergence, and visible pre/post checks. |
