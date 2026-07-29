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

The CT/MR fixture uses mutual information plus a reference transform so the
example is reproducible rather than optimizer-seed dependent. The registered
overlay is visibly different from identity because red and green fringes expose
residual misalignment, while the MR change panel exposes where resampling
actually changed the sampled values.

## Example Summary

| Example | Status | Focus |
| --- | --- | --- |
| [CT/MR Mutual-Information Registration](examples/registration_compare_figure.md) | Available | Labeled RIRE CT-to-MR overlays with identity, registered, and MR-change diagnostic panels. |
| [Validation Suite](examples/validation_suite.md) | Available | Geometry, metric, overlap, convergence, and visible pre/post checks. |
