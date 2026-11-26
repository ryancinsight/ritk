//! Estimate a rigid pose between two scalar images.
use super::metric::{Input, Samples, Stage};
use super::volume::center;
use crate::geometry::affine::AffineTransform as Grid;
use crate::LoadedVolume;
use anyhow::{ensure, Result};
use ritk_registration::{search_rigid_pose, AffineTransform, RigidSearchAnchor, RigidSearchConfig};
use std::sync::atomic::AtomicBool;

pub(super) fn estimate(
    fixed: &LoadedVolume,
    moving: &LoadedVolume,
    fixed_grid: Grid,
    moving_grid: Grid,
    shared_frame: bool,
    cancelled: &AtomicBool,
) -> Result<AffineTransform> {
    let fixed = Input::new(fixed, fixed_grid)?;
    let moving = Input::new(moving, moving_grid)?;
    let mut anchor = if shared_frame {
        RigidSearchAnchor::try_new(AffineTransform::IDENTITY, fixed.center)?
    } else {
        RigidSearchAnchor::from_centers(fixed.center, moving.center)?
    };
    let mut capture = Samples::new(&fixed, &moving, anchor.transform(), Stage::Capture)?;
    let mut score = |pose: &AffineTransform| capture.score(&moving, pose, cancelled);
    if !shared_frame {
        let centered = RigidSearchAnchor::from_centers(
            center(fixed.volume, fixed.grid),
            center(moving.volume, moving.grid),
        )?;
        if score(&centered.transform())? >= score(&anchor.transform())? {
            anchor = centered;
        }
    }

    // Fix the fine sampling domains at the captured pose, leaving a margin for
    // small residual rotations. Symmetric scoring reduces interpolation bias.
    let mut refinement = None;
    let refine = |pose: &AffineTransform| {
        let inverse = inverse_rigid(pose);
        let (forward, backward) = match &mut refinement {
            Some(domains) => domains,
            None => {
                let prepare = || -> Result<_> {
                    Ok((
                        Samples::new(&fixed, &moving, *pose, Stage::Refinement)?,
                        Samples::new(&moving, &fixed, inverse, Stage::Refinement)?,
                    ))
                };
                refinement.insert(prepare().map_err(|error| {
                    ritk_registration::classical::RegistrationError::InvalidInput(error.to_string())
                })?)
            }
        };
        Ok((forward.score(&moving, pose, cancelled)?
            + backward.score(&fixed, &inverse, cancelled)?)
            * 0.5)
    };
    let fit = search_rigid_pose(
        anchor,
        RigidSearchConfig::try_new(20.0, 32.0, 0.5, 0.75, 96)?.with_structural_half_range_cells(
            std::num::NonZeroU8::new(5).expect("positive cell count"),
        ),
        score,
        refine,
    )?;
    ensure!(
        !fit.capture_saturated,
        "The rigid fit reached its search limit. Check the selected anatomy and field of view."
    );
    ensure!(
        fit.structural_score > 0.0,
        "No usable intensity correspondence was found."
    );
    Ok(fit.structural_transform)
}

fn inverse_rigid(pose: &AffineTransform) -> AffineTransform {
    let matrix = pose.as_array();
    let mut inverse = AffineTransform::IDENTITY;
    for row in 0..3 {
        for column in 0..3 {
            inverse.0[row * 4 + column] = matrix[column * 4 + row];
        }
        inverse.0[row * 4 + 3] = -(0..3)
            .map(|k| matrix[k * 4 + row] * matrix[k * 4 + 3])
            .sum::<f64>();
    }
    inverse
}
