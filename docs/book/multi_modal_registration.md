# Multi-modal Registration

CT, MR, PET, dose, and derived maps do not share a simple intensity
relationship. Multi-modal registration therefore needs a modality-robust
objective, trustworthy physical metadata, and visual validation.

The minimum workflow is:

1. load both volumes and inspect their physical frames;
2. choose mutual information, MIND-SSC, LNCC, or NGF rather than assuming MSE is valid;
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

`SpatiallyConditionedMutualInformationMetric` additionally associates every
sample with a fixed zero-based region label. It computes

\[
H(F\mid R)=\sum_r p(r)H(F\mid r),\qquad
H(F,M\mid R)=\sum_r p(r)H(F,M\mid r),
\]

and applies the chosen NMI normalization to the conditional fixed, moving, and
joint entropies. This preserves coarse location information that one global
histogram discards. Region labels belong to the fixed frame and must not change
with the candidate pose. The reusable workspace allocates its regional
histograms once and clears them between evaluations. Toews and Wells derive
this local-region formulation in section 3.2, equations 8–9
(<https://doi.org/10.1007/978-3-642-02498-6_36>).

For CT/MR soft tissue, a bounded two-stage search is preferable to a fitted
weighted sum: NMI captures the multimodal basin, then a local structural metric
refines anatomy. The structural half-range defaults to one terminal NMI cell;
callers can select a larger nonzero number of cells while the original global
rigid bounds continue to constrain every candidate. A global MIND-SSC or NGF
search can lock onto remote anatomy, skull, or air boundaries.

Those bounds are refinement limits, not a substitute for orientation capture.
For a larger initial mismatch, generate normalized-cross-correlation block
matches in both directions, convert both sets to physical fixed-to-moving
correspondences, and pass them to `fit_symmetric_trimmed_rigid`. The resulting
full transform enters `search_rigid_pose` through `RigidSearchAnchor`; residual
parameters remain subject to the same finite bounds. This follows the rigid
subset of Modat et al.'s symmetric block-matching design
(<https://doi.org/10.1117/1.JMI.1.2.024003>, sections 2.1–2.3).

MIND-SSC provides a complementary local structural objective when corresponding
soft-tissue patches retain self-similarity despite modality-dependent
intensity. Prepare fixed descriptors once at complete-support centers. For each
candidate pose, only the six moving patch neighbourhoods at those centers are
sampled. The default deterministic 8,192-center cap bounds memory and runtime;
use caller-provided indices when an anatomical mask or validation protocol
defines the exact population. The image field follows ITK's half-voxel
`[-0.5,size-0.5)` convention: support inside it uses replicate-edge trilinear
interpolation, while support outside it is explicit zero background and stays
in the fixed denominator.

The CT/MR fixture uses moving-linear partial-volume NMI plus a reference
transform so the example is reproducible rather than optimizer-seed dependent.
The coarse figure grid is intentionally not used to demonstrate spatial
conditioning: its shallow axis leaves too little entropy per local region.
The registered overlay is visibly different from identity because red and
green fringes expose residual misalignment, while the MR change panel exposes
where resampling actually changed the sampled values.

## Example Summary

| Example | Status | Focus |
| --- | --- | --- |
| [CT/MR Mutual-Information Registration](examples/registration_compare_figure.md) | Available | Labeled RIRE CT-to-MR overlays with identity, registered, and MR-change diagnostic panels. |
| [Robust Rigid Capture Initializer](examples/rigid_capture_initializer.md) | Available | Symmetric 50%-trimmed rigid fitting and bounded full-anchor refinement. |
| [Validation Suite](examples/validation_suite.md) | Available | Geometry, metric, overlap, convergence, and visible pre/post checks. |
