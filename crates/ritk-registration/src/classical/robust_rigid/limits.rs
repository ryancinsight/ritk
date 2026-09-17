/// Maximum least-trimmed-squares refits used by NiftyReg's `reg_aladin`.
pub(super) const REFIT_LIMIT: usize = 5;
/// Exact elemental candidates remain bounded for small correspondence sets.
pub(super) const EXACT_CANDIDATE_LIMIT: usize = 4_096;
/// Deterministic elemental candidates for larger sets.
///
/// At the limiting 50% inlier fraction, 1,024 independent three-point draws
/// miss an all-inlier subset with probability `(7/8)^1024 < f64::EPSILON^2`.
/// The deterministic sequence makes registration reproducible; the bound
/// explains its breadth but is not claimed as a probabilistic guarantee for
/// adversarially ordered input.
pub(super) const SAMPLED_CANDIDATE_LIMIT: usize = 1_024;
/// `sqrt(f64::EPSILON)`, used as a relative rank threshold for 3-D point sets.
pub(super) const RANK_TOLERANCE: f64 = 1.490_116_119_384_765_6e-8;
/// Rotations this close to the logarithm branch cut cannot yield a stable axis.
///
/// The skew part has magnitude `sin(theta)`. Below `sqrt(epsilon)`, normalizing
/// it loses at least half the significand, so the principal logarithm fails
/// closed rather than selecting an unstable sign at `theta = pi`.
pub(super) const ROTATION_LOG_BRANCH_TOLERANCE: f64 = RANK_TOLERANCE;
