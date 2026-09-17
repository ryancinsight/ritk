use super::super::error::{RegistrationError, Result};

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct RigidCorrespondence {
    pub(super) source_mm: [f64; 3],
    pub(super) target_mm: [f64; 3],
}

#[derive(Debug)]
pub(super) struct DirectionalFit {
    pub(super) transform: crate::types::AffineTransform,
    pub(super) inlier_count: usize,
    pub(super) squared_residual_sum: f64,
}

/// One finite fixed-to-moving physical-space correspondence.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct FixedToMovingCorrespondence {
    fixed_mm: [f64; 3],
    moving_mm: [f64; 3],
}

impl FixedToMovingCorrespondence {
    /// Construct a correspondence from fixed to moving millimetres.
    ///
    /// # Errors
    ///
    /// Returns [`RegistrationError::InvalidInput`] when either point contains
    /// a non-finite coordinate.
    pub fn try_new(fixed_mm: [f64; 3], moving_mm: [f64; 3]) -> Result<Self> {
        validate_points(fixed_mm, moving_mm, "fixed-to-moving")?;
        Ok(Self {
            fixed_mm,
            moving_mm,
        })
    }

    /// Return the point in fixed-image physical coordinates.
    #[must_use]
    pub const fn fixed_mm(self) -> [f64; 3] {
        self.fixed_mm
    }

    /// Return the corresponding point in moving-image physical coordinates.
    #[must_use]
    pub const fn moving_mm(self) -> [f64; 3] {
        self.moving_mm
    }
}

/// One finite moving-to-fixed physical-space correspondence.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct MovingToFixedCorrespondence {
    moving_mm: [f64; 3],
    fixed_mm: [f64; 3],
}

impl MovingToFixedCorrespondence {
    /// Construct a correspondence from moving to fixed millimetres.
    ///
    /// The argument order follows the measured direction and therefore differs
    /// intentionally from [`FixedToMovingCorrespondence::try_new`].
    ///
    /// # Errors
    ///
    /// Returns [`RegistrationError::InvalidInput`] when either point contains
    /// a non-finite coordinate.
    pub fn try_new(moving_mm: [f64; 3], fixed_mm: [f64; 3]) -> Result<Self> {
        validate_points(moving_mm, fixed_mm, "moving-to-fixed")?;
        Ok(Self {
            moving_mm,
            fixed_mm,
        })
    }

    /// Return the point in moving-image physical coordinates.
    #[must_use]
    pub const fn moving_mm(self) -> [f64; 3] {
        self.moving_mm
    }

    /// Return the corresponding point in fixed-image physical coordinates.
    #[must_use]
    pub const fn fixed_mm(self) -> [f64; 3] {
        self.fixed_mm
    }
}

fn validate_points(first_mm: [f64; 3], second_mm: [f64; 3], direction: &str) -> Result<()> {
    if first_mm
        .iter()
        .chain(second_mm.iter())
        .any(|value| !value.is_finite())
    {
        return Err(RegistrationError::InvalidInput(format!(
            "{direction} rigid correspondence must be finite, got source {first_mm:?}, target {second_mm:?}"
        )));
    }
    Ok(())
}

pub(super) fn forward_correspondences(
    supplied: &[FixedToMovingCorrespondence],
) -> Result<Vec<RigidCorrespondence>> {
    let mut correspondences = Vec::new();
    correspondences
        .try_reserve_exact(supplied.len())
        .map_err(|error| {
            RegistrationError::InvalidInput(format!(
                "cannot allocate {} forward rigid correspondences: {error}",
                supplied.len()
            ))
        })?;
    correspondences.extend(supplied.iter().map(|pair| RigidCorrespondence {
        source_mm: pair.fixed_mm,
        target_mm: pair.moving_mm,
    }));
    correspondences.sort_by(compare_correspondences);
    Ok(correspondences)
}

pub(super) fn reverse_correspondences(
    supplied: &[MovingToFixedCorrespondence],
) -> Result<Vec<RigidCorrespondence>> {
    let mut correspondences = Vec::new();
    correspondences
        .try_reserve_exact(supplied.len())
        .map_err(|error| {
            RegistrationError::InvalidInput(format!(
                "cannot allocate {} reverse rigid correspondences: {error}",
                supplied.len()
            ))
        })?;
    correspondences.extend(supplied.iter().map(|pair| RigidCorrespondence {
        source_mm: pair.moving_mm,
        target_mm: pair.fixed_mm,
    }));
    correspondences.sort_by(compare_correspondences);
    Ok(correspondences)
}

fn compare_correspondences(
    left: &RigidCorrespondence,
    right: &RigidCorrespondence,
) -> std::cmp::Ordering {
    left.source_mm
        .into_iter()
        .chain(left.target_mm)
        .zip(right.source_mm.into_iter().chain(right.target_mm))
        .find_map(|(left, right)| {
            let ordering = left.total_cmp(&right);
            (ordering != std::cmp::Ordering::Equal).then_some(ordering)
        })
        .unwrap_or(std::cmp::Ordering::Equal)
}

/// Total order over one point, so endpoint comparison never depends on NaN.
fn compare_points(left: [f64; 3], right: [f64; 3]) -> std::cmp::Ordering {
    left.into_iter()
        .zip(right)
        .find_map(|(left, right)| {
            let ordering = left.total_cmp(&right);
            (ordering != std::cmp::Ordering::Equal).then_some(ordering)
        })
        .unwrap_or(std::cmp::Ordering::Equal)
}

/// The correspondence read back as (fixed, moving), whichever direction built
/// it. A forward schedule stores source = fixed; a reverse schedule stores
/// source = moving, so reading a reverse entry in this frame swaps it.
const fn as_fixed_moving(pair: &RigidCorrespondence, reversed: bool) -> Endpoints {
    if reversed {
        (pair.target_mm, pair.source_mm)
    } else {
        (pair.source_mm, pair.target_mm)
    }
}

type Endpoints = ([f64; 3], [f64; 3]);

struct EndpointClaim {
    endpoints: Endpoints,
    orientation: std::cmp::Ordering,
}

fn canonical_endpoints(pair: &RigidCorrespondence) -> Endpoints {
    if compare_points(pair.source_mm, pair.target_mm).is_gt() {
        (pair.target_mm, pair.source_mm)
    } else {
        (pair.source_mm, pair.target_mm)
    }
}

fn compare_endpoints(left: &Endpoints, right: &Endpoints) -> std::cmp::Ordering {
    compare_points(left.0, right.0).then_with(|| compare_points(left.1, right.1))
}

/// Drop every endpoint pair the two directions disagree about.
///
/// Read in the fixed-to-moving frame, a forward and a reverse correspondence
/// over the same two points must agree on which endpoint is fixed. When they
/// do not, both cannot hold and neither is the more credible, so the pair
/// leaves both schedules rather than one being chosen — the symmetry the fit
/// depends on is exactly what a one-sided choice would break.
pub(super) fn discard_conflicting_endpoint_pairs(
    forward: &mut Vec<RigidCorrespondence>,
    reverse: &mut Vec<RigidCorrespondence>,
) {
    let mut claims: Vec<EndpointClaim> = forward
        .iter()
        .map(|pair| (pair, false))
        .chain(reverse.iter().map(|pair| (pair, true)))
        .map(|(pair, reversed)| {
            let (fixed, moving) = as_fixed_moving(pair, reversed);
            EndpointClaim {
                endpoints: canonical_endpoints(pair),
                orientation: compare_points(fixed, moving),
            }
        })
        .collect();
    claims.sort_by(|left, right| compare_endpoints(&left.endpoints, &right.endpoints));

    let mut conflicting: Vec<Endpoints> = Vec::new();
    let mut start = 0;
    while start < claims.len() {
        let mut end = start + 1;
        while end < claims.len()
            && compare_endpoints(&claims[start].endpoints, &claims[end].endpoints).is_eq()
        {
            end += 1;
        }
        if claims[start + 1..end]
            .iter()
            .any(|claim| claim.orientation != claims[start].orientation)
        {
            conflicting.push(claims[start].endpoints);
        }
        start = end;
    }
    if conflicting.is_empty() {
        return;
    }
    let retain = |pair: &RigidCorrespondence| {
        conflicting
            .binary_search_by(|probe| compare_endpoints(probe, &canonical_endpoints(pair)))
            .is_err()
    };
    forward.retain(retain);
    reverse.retain(retain);
}

/// Result of a symmetric 50%-trimmed rigid fit.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
#[non_exhaustive]
pub struct SymmetricRigidFit {
    /// Rigid transform mapping fixed to moving physical coordinates.
    pub transform: crate::types::AffineTransform,
    /// Total number of forward and reverse correspondences fitted.
    pub correspondence_count: usize,
    /// Total number retained by the two independent 50% LTS fits.
    pub inlier_count: usize,
    /// Root-mean-square residual over both directional inlier sets, in mm.
    ///
    /// Reverse residuals are measured in fixed space before inversion. Rigid
    /// distance preservation makes their magnitudes equal in moving space.
    pub inlier_rms_mm: f64,
}
