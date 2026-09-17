use super::super::error::{RegistrationError, Result};
use super::correspondence::{
    discard_conflicting_endpoint_pairs, forward_correspondences, reverse_correspondences,
    FixedToMovingCorrespondence, MovingToFixedCorrespondence, SymmetricRigidFit,
};
use super::lie::{invert_rigid, log_euclidean_mean};
use super::trimmed::fit_trimmed_direction;

/// Fit one rigid transform to bidirectional block correspondences.
///
/// `fixed_to_moving` stores matches measured from fixed-image blocks.
/// `moving_to_fixed` stores matches measured after swapping the image order;
/// each set receives an independent LTS fit retaining the half with the
/// smallest squared residuals. The reverse transform is inverted and the two
/// fixed-to-moving transforms are combined with the log/exp mean from Modat
/// et al., equations 4–5. Repeated pairs retain their supplied weight.
///
/// Exact unordered endpoint pairs that disagree about their fixed endpoint
/// are removed from both schedules before fitting. This keeps the directional
/// schedules symmetric under exchanging the image roles.
///
/// The implementation stores two correspondence vectors plus per-direction
/// index and residual vectors, so peak auxiliary memory is linear in the
/// larger directional match count. Its five-refit cap is fixed by the
/// reference implementation rather than by validation-subject tuning.
///
/// # Errors
///
/// Returns [`RegistrationError::InvalidInput`] when either direction supplies
/// fewer than six correspondences, because retaining 50% must leave at least
/// three non-collinear points, or when no retained subset has sufficient rank.
/// Returns [`RegistrationError::NumericalFailure`] when SVD fails or a fitted
/// rotation reaches the numerically unresolved branch of the principal matrix
/// logarithm at 180 degrees.
///
/// # References
///
/// Modat M, et al. “Global image registration using a symmetric block-matching
/// approach.” *Journal of Medical Imaging* 1(2), 2014, sections 2.1–2.3.
/// <https://doi.org/10.1117/1.JMI.1.2.024003>
pub fn fit_symmetric_trimmed_rigid(
    fixed_to_moving: &[FixedToMovingCorrespondence],
    moving_to_fixed: &[MovingToFixedCorrespondence],
) -> Result<SymmetricRigidFit> {
    if fixed_to_moving.len() < 6 || moving_to_fixed.len() < 6 {
        return Err(RegistrationError::InvalidInput(format!(
            "50%-trimmed symmetric rigid fitting needs at least six correspondences per direction, got {} forward and {} reverse",
            fixed_to_moving.len(),
            moving_to_fixed.len()
        )));
    }

    let mut forward = forward_correspondences(fixed_to_moving)?;
    let mut reverse = reverse_correspondences(moving_to_fixed)?;
    discard_conflicting_endpoint_pairs(&mut forward, &mut reverse);
    let forward_fit = fit_trimmed_direction(&forward)?;
    let reverse_fit = fit_trimmed_direction(&reverse)?;
    let reverse_inverse = invert_rigid(&reverse_fit.transform)?;
    let transform = log_euclidean_mean(&forward_fit.transform, &reverse_inverse)?;
    // Retained pairs, not supplied ones: a discarded conflict contributed to
    // neither fit and must not be reported as though it had.
    let correspondence_count = forward.len().checked_add(reverse.len()).ok_or_else(|| {
        RegistrationError::InvalidInput(
            "bidirectional correspondence count overflows usize".to_owned(),
        )
    })?;
    let inlier_count = forward_fit
        .inlier_count
        .checked_add(reverse_fit.inlier_count)
        .ok_or_else(|| {
            RegistrationError::InvalidInput("bidirectional inlier count overflows usize".to_owned())
        })?;
    let mean_squared =
        (forward_fit.squared_residual_sum + reverse_fit.squared_residual_sum) / inlier_count as f64;
    Ok(SymmetricRigidFit {
        transform,
        correspondence_count,
        inlier_count,
        inlier_rms_mm: mean_squared.sqrt(),
    })
}
