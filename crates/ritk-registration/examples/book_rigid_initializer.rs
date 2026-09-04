//! Fit a robust full rigid anchor and refine only its bounded residual pose.
#![expect(
    clippy::print_stdout,
    reason = "example reports the recovered pose and retained correspondence count"
)]

use anyhow::{ensure, Result};
use ritk_registration::{
    fit_symmetric_trimmed_rigid, search_rigid_pose, AffineTransform, FixedToMovingCorrespondence,
    MovingToFixedCorrespondence, RigidSearchAnchor, RigidSearchConfig,
};

fn transform_point(point: [f64; 3]) -> [f64; 3] {
    let angle = 20.0_f64.to_radians();
    let (sine, cosine) = angle.sin_cos();
    [
        cosine * point[0] - sine * point[1] + 4.0,
        sine * point[0] + cosine * point[1] - 3.0,
        point[2] + 2.0,
    ]
}

fn transform_residual(candidate: &AffineTransform, expected: &AffineTransform) -> f64 {
    candidate
        .as_array()
        .iter()
        .zip(expected.as_array())
        .map(|(actual, expected)| (actual - expected).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn main() -> Result<()> {
    let fixed_points = [
        [-2.0, -1.0, 0.0],
        [3.0, -1.0, 1.0],
        [-1.0, 4.0, 2.0],
        [2.0, 3.0, -2.0],
        [-3.0, 1.0, 4.0],
        [4.0, 2.0, 3.0],
    ];
    let mut forward = Vec::new();
    let mut reverse = Vec::new();
    for fixed in fixed_points {
        let moving = transform_point(fixed);
        forward.push(FixedToMovingCorrespondence::try_new(fixed, moving)?);
        reverse.push(MovingToFixedCorrespondence::try_new(moving, fixed)?);
    }
    // Forty percent of each direction is deliberately inconsistent. The fit
    // retains the best half, so the correct 60% consensus remains identifiable.
    for index in 0_u32..4 {
        let offset = f64::from(index);
        let fixed = [50.0 + offset, -80.0, 20.0];
        let moving = [-70.0, 40.0 + offset, -30.0];
        forward.push(FixedToMovingCorrespondence::try_new(fixed, moving)?);
        reverse.push(MovingToFixedCorrespondence::try_new(moving, fixed)?);
    }

    let fitted = fit_symmetric_trimmed_rigid(&forward, &reverse)?;
    let expected = AffineTransform([
        20.0_f64.to_radians().cos(),
        -20.0_f64.to_radians().sin(),
        0.0,
        4.0,
        20.0_f64.to_radians().sin(),
        20.0_f64.to_radians().cos(),
        0.0,
        -3.0,
        0.0,
        0.0,
        1.0,
        2.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]);
    let gamma = 1_024.0 * f64::EPSILON / (1.0 - 1_024.0 * f64::EPSILON);
    ensure!(
        transform_residual(&fitted.transform, &expected) <= gamma * 4.0,
        "trimmed fit did not recover the manufactured rigid transform"
    );

    let anchor = RigidSearchAnchor::try_new(fitted.transform, [0.0; 3])?;
    let objective = |candidate: &AffineTransform| Ok(-transform_residual(candidate, &expected));
    let search = search_rigid_pose(
        anchor,
        RigidSearchConfig::try_new(4.0, 4.0, 0.5, 0.5, 128)?,
        objective,
        objective,
    )?;
    ensure!(
        transform_residual(&search.structural_transform, &expected) <= gamma * 4.0,
        "bounded residual search moved away from the recovered anchor"
    );

    println!(
        "retained {}/{} correspondences; inlier RMS {:.3e} mm; search saturated: {}",
        fitted.inlier_count,
        fitted.correspondence_count,
        fitted.inlier_rms_mm,
        search.capture_saturated || search.structural_saturated
    );
    Ok(())
}
