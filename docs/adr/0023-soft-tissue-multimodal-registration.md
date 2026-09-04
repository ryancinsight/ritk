# ADR 0023: Soft-tissue multimodal rigid registration

- **Status:** Accepted
- **Board item:** `RITK-SOFT-TISSUE-REGISTRATION-2026-09-03`
- **Class:** [major] [arch]
- **Date:** 2026-09-03
- **Revision 2026-09-03:** Add the fixed-region conditioned metric after the
  downstream global-histogram RIRE result remained anatomically offset.
- **Revision 2026-09-04:** Make the structural-refinement half-range a bounded
  nonzero terminal-cell count after one cell proved too narrow for downstream
  MIND-SSC investigation; preserve one cell as the default.

## Context

The classical mutual-information path assigns each sample to one discrete
histogram bin and uses one intensity range for both modalities. The downstream
CT/MR registration duplicates this estimator and a six-parameter search because
the images have different physical grids. On RIRE Patient-001, the earlier
downstream image-only optimum reached 1.89 mm mean and 3.33 mm maximum target
registration error (TRE), but its global NMI was slightly higher than the
independent fiducial pose while its Normalized Gradient Fields (NGF) score was
lower. The estimator therefore selected a visible soft-tissue offset; more
iterations optimized the wrong local ordering.

Maes et al. report that multimodal mutual-information registration is sensitive
to interpolation, optimization, and changing overlap [1]. Ikeda et al. identify
the discontinuity of discrete histogram estimation and use Parzen windows to
obtain a continuous mutual-information objective [2]. Haber and Modersitzki
describe the local-maxima problem for mutual information and formulate NGF as
an edge-orientation measure for multimodal registration [3]. Toews and Wells
condition image entropies on local regions to retain coarse spatial
correspondence that a global histogram discards [4].

## Decision

`MutualInformationMetric` gains an explicit histogram-estimator strategy.
Discrete estimation remains the default and preserves exact discrete-information
identities. The moving-linear partial-volume estimator keeps the fixed modality
discrete and contributes each resampled moving intensity to two adjacent bins
through a first-order B-spline kernel. This makes the transform-dependent axis
piecewise continuous without blurring both marginals. The type accepts
independent fixed and moving intensity ranges and evaluates borrowed paired
samples with an optional mask, so physical-grid resamplers can use the metric
without copying full volumes into a RITK image type.

Add one bounded six-parameter rigid-pose optimizer in `classical::rigid_search`.
It owns centroid-anchored ZYX pose construction and derivative-free search but
accepts the image sampler and similarity measures as monomorphized closures.
Objective closures are fallible, so invalid samples and allocation errors
retain their registration failure instead of being replaced with a score.
The first stage captures the multimodal basin with partial-volume NMI. The
second stage refines a structural objective inside a local half-range measured
in final NMI-resolution cells. `RigidSearchConfig::try_new` preserves a one-cell
default; an additive `NonZeroU8` builder can widen that local range. The same
radius scales the structural initial simplex and saturation boundary, while the
original global rigid bounds still reject every out-of-range candidate. The
second stage therefore cannot become an unbounded global search. Callers retain
explicit coverage and overlap gates.

Morphological filters accept independent axis radii, and a physical-radius
conversion derives those radii from image spacing. A caller can therefore keep
mask support physically bounded on anisotropic acquisitions rather than making
thick slices dominate the registration region.

Add `SpatiallyConditionedMutualInformationMetric` as a reusable workspace over
`MutualInformationMetric`. Every selected sample carries a fixed zero-based
region label. The workspace computes `H(F|R)`, `H(M|R)`, and `H(F,M|R)` and then
applies the configured normalization. Its regional joint and marginal
histograms allocate once and are cleared between poses. Fixed masks and labels
must remain constant during optimization; callers represent out-of-field
moving samples explicitly rather than deleting them from the sample population.

The transform convention is row-major fixed-to-moving `[z, y, x]` millimetres.
The implementation is deterministic. The conditioned metric performs no
per-pose allocation after construction.

## Alternatives rejected

- Continue optimizing hard-bin NMI: rejected because the RIRE reference scores
  below the displaced optimum under that estimator.
- Continue using one global partial-volume histogram: rejected because it
  preserves intensity co-occurrence but not coarse anatomical location.
- Optimize NGF globally: rejected because the RIRE objective has remote edge
  maxima and a global NGF trial increased mean TRE to 8.07 mm.
- Combine NMI and NGF with a fitted scalar weight: rejected because no physical
  or statistical derivation fixes that weight; fitting it to one subject would
  leak validation data into the algorithm.
- Apply deformable registration: rejected because the same-patient CT/MR pose is
  rigid and a deformable model can hide a rigid-estimation defect.

## Verification and limits

Analytical tests cover the discrete estimator's exact identities, partial-volume
mass conservation and continuity, masking, invalid inputs, conditioned-entropy
identities, workspace clearing, invalid region labels, a manufactured global-
histogram ambiguity resolved by location conditioning, rigid centroid mapping,
bound saturation, fallible objective propagation, and recovery of a coupled
manufactured optimum. Additional value tests prove implicit/explicit one-cell
equivalence, recovery of a wider manufactured structural optimum, confinement
to the original global bounds, and local/global saturation semantics. The
downstream LeoNeuro RIRE oracle evaluates image-only registration against
held-out fiducials: the 3×3×3 conditioned capture reaches
0.8330 mm mean and 1.1324 mm maximum TRE, while an adversarial field-of-view
crop loses support and scores below the fiducial pose. The selected pose reaches
one search bound; wider trials select a remote histogram maximum and worsen
TRE. One public subject validates this regression but does not estimate
population, clinical, or FDA performance.

## References

1. Maes F, et al. “Multimodality image registration by maximization of mutual
   information.” *IEEE Transactions on Medical Imaging* 16(2), 1997.
   DOI: <https://doi.org/10.1109/42.563664>.
2. Ikeda T, et al. “Mutual Information-Based Registration Using Parzen
   Windowing.” *IEICE Transactions on Information and Systems* E91-D(1), 2008.
   DOI: <https://doi.org/10.1093/ietisy/e91-d.1.132>.
3. Haber E, Modersitzki J. “Intensity Gradient Based Registration and Fusion of
   Multi-modal Images.” *MICCAI 2006*. DOI:
   <https://doi.org/10.1007/11866763_89>.
4. Toews M, Wells WM. “Bayesian Registration via Local Image Regions.”
   *Information Processing in Medical Imaging*, 2009, section 3.2, equations
   8–9. DOI: <https://doi.org/10.1007/978-3-642-02498-6_36>.
