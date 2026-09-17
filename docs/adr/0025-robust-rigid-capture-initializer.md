# ADR 0025: Robust rigid capture initializer

- **Status:** Accepted
- **Board item:** `RITK-RIGID-CAPTURE-INITIALIZER-2026-09-04`
- **Class:** [major] [arch]
- **Date:** 2026-09-04

### Revision — 2026-09-16

The landed implementation follows the directional estimator selected in PR
248. This revision replaces the earlier joint-fit wording and records the
actual split schedules, endpoint conflict rule, and log-Euclidean combination.

## Context

The rigid search accepted only fixed and moving centroids. That initialization
estimates translation but cannot estimate orientation, so a downstream CT/MR
case reached the rotational and translational capture bounds despite a useful
soft-tissue objective. Expanding those bounds selected remote multimodal image
maxima rather than fixing the missing initializer.

Principal-axis alignment is not an appropriate replacement. ITK documents that
moment initialization assumes similar anatomical intensity moments and warns
that the assumption probably does not hold for multi-modality registration
[2]. The symmetric block-matching method of Modat et al. instead establishes
local normalized-cross-correlation correspondences in both image directions,
rejects the largest 50% of residuals with least-trimmed-squares (LTS), and fits
a global transformation [1, sections 2.1–2.3]. Their RIRE evaluation reports a
1.60 mm mean and 3.62 mm maximum CT-to-T1 target registration error over nine
subjects and 704 landmarks [1, section 3 and table 1].

## Decision

RITK owns two reusable registration primitives:

1. `fit_symmetric_trimmed_rigid` accepts direction-specific
   `FixedToMovingCorrespondence` and `MovingToFixedCorrespondence` values and
   fits each schedule independently with 50%-trimmed least squares. The
   reverse fit is inverted and combined with the forward fit through the
   log-Euclidean mean described by Modat et al., equations 4–5. The distinct
   types make source/target order a checked API contract, while equal treatment
   of the two retained directional halves removes direction bias.
2. `RigidSearchAnchor` replaces the two-centroid `search_rigid_pose` arguments.
   It validates a full proper rigid transform and a fixed-frame center. Search
   rotations right-compose in the fixed frame and translations act in moving-
   frame millimetres. The exact zero residual returns the supplied transform.

Each directional LTS initializer evaluates every non-collinear three-pair
elemental subset while there are at most 4,096 combinations. Larger schedules
evaluate 1,024 deterministic sampled subsets. Directed source/target sorting
makes exchanging the image roles exchange the two schedules exactly; the
unordered endpoint key is used only for conflict detection. All paths perform
at most five LTS concentration refits. At a 50% inlier fraction, 1,024
independent three-point draws would miss an all-inlier subset with probability
`(7/8)^1024`, below `f64::EPSILON^2`; the deterministic sequence provides
reproducibility, not a probabilistic guarantee against adversarially arranged
correspondence values.

Repeated correspondences with the same unordered endpoint key retain their
directional weight when every copy has one orientation. A key containing both
orientations is discarded as ambiguous: selecting either orientation would
change which correspondence identity occupies a sampled position after image
roles swap.
Auxiliary storage remains linear in correspondence count and does not allocate
a pairwise-consistency matrix.

RITK registration owns fitting and search composition. The block-matching crate
continues to own local image correspondence generation. Applications own image
resampling into compatible physical grids and modality-specific acceptance
criteria.

## Alternatives rejected

- Principal axes or intensity moments: rejected because modality-dependent
  image moments need not represent the same anatomy [2].
- Keep centroid-only search and expand bounds: rejected because it does not
  estimate rotation and increases exposure to remote objective maxima.
- Pairwise correspondence-consistency graph: rejected because quadratic
  memory conflicts with bounded operation on dense block populations.
- One joint fit after normalizing both directions: rejected because one
  schedule could contribute more retained correspondences and dominate the
  result. Independent 50%-trimmed fits preserve equal directional influence;
  the log-Euclidean mean supplies the inverse-consistent transform.

## Verification and limits

Analytical tests cover a known 20-degree transform with 40% coherent outliers,
direction swapping and inverse composition on both exhaustive and sampled
candidate paths, exact schedule exchange with independent directional outliers,
rejection of conflicting unordered-endpoint directions, input-order
normalization, non-finite points, and collinear points. The RIRE integration
test fits the eight published CT→MR fiducials with two gross outliers per
direction and verifies the 0.01 mm target-registration-error bound. Existing
capture and refinement bounds remain covered, and the runnable book example
executes the same known-transform workflow.

The fit assumes strictly more than half of the supplied correspondences support
one identifiable non-collinear rigid transform. Exactly half is intrinsically
ambiguous when two rigid consensuses have equal residual. The deterministic
large-set candidate schedule is reproducible but has no adversarial guarantee.
Clinical or FDA performance requires population validation independent of these
algorithmic tests.

## References

1. Modat M, et al. “Global image registration using a symmetric block-matching
   approach.” *Journal of Medical Imaging* 1(2), 2014, sections 2.1–2.3 and 3,
   table 1. <https://doi.org/10.1117/1.JMI.1.2.024003>.
2. ITK, `CenteredTransformInitializer` class reference, “MomentsOn” notes.
   <https://docs.itk.org/projects/doxygen/en/stable/classitk_1_1CenteredTransformInitializer.html>.
