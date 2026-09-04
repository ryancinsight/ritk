# Optimization and Registration

Metrics only become registration algorithms once ritk couples them to an optimizer. On the differentiable side, `ritk-registration::metric::autodiff` supplies a reusable `gradient_descent` driver configured by `GradientDescentConfig`, rebuilding a transform from trainable Coeus `Var`s each iteration and stepping parameters with the native SGD helper. On the classical side, `ImageRegistration` uses `ClassicalConfig` and `MutualInformationMetric` to run deterministic CPU optimization over rigid or affine parameters. This chapter covers that seam: parameterization, iteration budgets, tolerances, step sizes, and why optimizer behavior must be read together with the chosen similarity metric.

Atlas integration is split but coherent. Coeus provides the autodiff graph, tensor execution, and backend flexibility for learning-style registration loops, while Leto supports the classical numeric path where predictable CPU array behavior is preferred. RITK keeps those implementation details behind a common image boundary so callers can reason about transforms, convergence, and output geometry without rewriting file I/O or preprocessing around each optimizer family.

## Bounded rigid capture and refinement

`search_rigid_pose` searches a six-degree-of-freedom rigid transform in physical
millimetres. It starts from a validated `RigidSearchAnchor`, performs four
coarse-to-fine coordinate-descent levels, and polishes the capture objective
with bounded Nelder–Mead. A second structural objective defaults to a half-range
of one terminal capture cell. `with_structural_half_range_cells` accepts a
`NonZeroU8` when a consumer needs to test a wider local basin. The configured
radius scales both the structural bounds and initial simplex, but the original
global rigid bounds remain authoritative. The effective interval is their
intersection. When capture ends on a global boundary, each simplex edge points
toward the side with available room rather than constructing an invalid outward
vertex. This separation supports multimodal
registration where normalized mutual information (NMI) finds the broad basin
and MIND-SSC or Normalized Gradient Fields (NGF) resolves local soft-tissue
structure without enabling an unbounded second search. The tested Rustdoc on
`RigidSearchConfig::with_structural_half_range_cells` is the copyable API
example.

The capture schedule and all simplex operations preserve finite proposals and
their bounded-objective rejection semantics. If an otherwise valid finite
resolution multiplied by a schedule factor would overflow, only that
non-finite proposal is replaced by the finite endpoint in its direction before
an objective sees it. A separate finite-transform check converts overflow from
centroid and residual composition into a typed numerical failure before metric
evaluation.

`RigidSearchResult::capture_saturated` and `structural_saturated` report when an
optimum touches its permitted boundary. Structural saturation covers either the
configured local half-range or a tighter global bound. Saturation is a
quality-control signal: it means the configured search region may have clipped
the optimum, not that the transform is invalid. The caller still owns image
sampling, fixed-domain support, overlap acceptance, and the final choice between
the capture and structural candidates.

Objective errors propagate through the search result. A malformed or empty
metric sample set therefore cannot be converted into a plausible pose.

When centroids alone do not place the orientation inside this bounded search,
`fit_symmetric_trimmed_rigid` estimates the complete anchor from direction-
specific `FixedToMovingCorrespondence` and `MovingToFixedCorrespondence`
physical-space matches. It normalizes reverse matches and retains the half with
smallest Euclidean residual under a deterministic LTS fit. The fit requires a
strict majority supporting one non-collinear rigid consensus;
exactly two equal half-population consensuses are not identifiable. See the
[robust rigid capture initializer](examples/rigid_capture_initializer.md) for
the executable known-transform and outlier case.

## Example Summary

| Example | Status | Focus |
| --- | --- | --- |
| [Deep Learning Registration](examples/dl_registration.md) | Available | End-to-end differentiable optimization of rigid parameters with Coeus autodiff. |
| [Deep Learning Training](examples/dl_train.md) | Available | Extends the same optimization ideas to a learned registration model and training loop. |
| [CT/MR Mutual-Information Registration](examples/registration_compare_figure.md) | Available | Validates multi-modal MI against the RIRE CT-to-MR transform and native resampling. |
| [Robust Rigid Capture Initializer](examples/rigid_capture_initializer.md) | Available | Fits a full anchor from symmetric correspondences before bounded residual search. |
