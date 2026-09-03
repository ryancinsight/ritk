# ADR 0023: Soft-tissue multimodal rigid registration

- **Status:** Accepted
- **Board item:** `RITK-SOFT-TISSUE-REGISTRATION-2026-09-03`
- **Class:** [major] [arch]
- **Date:** 2026-09-03

## Context

The classical mutual-information path assigns each sample to one discrete
histogram bin and uses one intensity range for both modalities. The downstream
CT/MR registration duplicates this estimator and a six-parameter search because
the images have different physical grids. On RIRE Patient-001, the image-only
optimum reaches 1.89 mm mean and 3.33 mm maximum target registration error
(TRE), but its NMI is slightly higher than the independent fiducial pose while
its Normalized Gradient Fields (NGF) score is lower. The estimator therefore
selects a visible soft-tissue offset; more iterations optimize the wrong local
ordering.

Maes et al. report that multimodal mutual-information registration is sensitive
to interpolation, optimization, and changing overlap [1]. Ikeda et al. identify
the discontinuity of discrete histogram estimation and use Parzen windows to
obtain a continuous mutual-information objective [2]. Haber and Modersitzki
describe the local-maxima problem for mutual information and formulate NGF as
an edge-orientation measure for multimodal registration [3].

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
The first stage captures the multimodal basin with partial-volume NMI. The
second stage refines NGF inside the final NMI resolution cell; it cannot perform
a second global search. Callers retain explicit coverage and overlap gates.

Morphological filters accept independent axis radii, and a physical-radius
conversion derives those radii from image spacing. A caller can therefore keep
mask support physically bounded on anisotropic acquisitions rather than making
thick slices dominate the registration region.

The transform convention is row-major fixed-to-moving `[z, y, x]` millimetres.
The implementation is deterministic and allocation-free per optimizer state;
metric histograms allocate once per evaluation until profile evidence justifies
caller-owned scratch storage.

## Alternatives rejected

- Continue optimizing hard-bin NMI: rejected because the RIRE reference scores
  below the displaced optimum under that estimator.
- Optimize NGF globally: rejected because the RIRE objective has remote edge
  maxima and a global NGF trial increased mean TRE to 8.07 mm.
- Combine NMI and NGF with a fitted scalar weight: rejected because no physical
  or statistical derivation fixes that weight; fitting it to one subject would
  leak validation data into the algorithm.
- Apply deformable registration: rejected because the same-patient CT/MR pose is
  rigid and a deformable model can hide a rigid-estimation defect.

## Verification and limits

Analytical tests cover the discrete estimator's exact identities, partial-volume
mass conservation and continuity, masking, invalid inputs, rigid centroid
mapping, bound saturation, and recovery of a coupled manufactured optimum.
The RIRE test evaluates image-only registration against held-out fiducials and
renders CT/MR overlays for inspection. One public subject validates this
regression but does not estimate population, clinical, or FDA performance.

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
