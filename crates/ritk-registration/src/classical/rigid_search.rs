//! Bounded multimodal rigid-pose search in physical coordinates.

use super::error::{RegistrationError, Result};
use crate::types::AffineTransform;

const PARAMETER_COUNT: usize = 6;
const SIMPLEX_VERTEX_COUNT: usize = PARAMETER_COUNT + 1;
// Four levels span an eightfold coarse cell followed by three bisections to the
// requested terminal resolution.
const CAPTURE_LEVELS: u32 = 4;

/// Bounds and terminal resolution for centroid-anchored rigid registration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RigidSearchConfig {
    rotation_half_range_radians: f64,
    translation_half_range_mm: f64,
    final_rotation_resolution_radians: f64,
    final_translation_resolution_mm: f64,
    simplex_iteration_limit: usize,
}

impl RigidSearchConfig {
    /// Validate a bounded rigid-search configuration.
    ///
    /// Rotation values are degrees and translations are millimetres. The final
    /// resolution must not exceed its corresponding search half-range.
    ///
    /// # Errors
    ///
    /// Returns [`RegistrationError::InvalidInput`] when a value is non-finite,
    /// non-positive, the terminal resolution exceeds its half-range, or the
    /// simplex iteration limit is zero.
    pub fn try_new(
        rotation_half_range_deg: f64,
        translation_half_range_mm: f64,
        final_rotation_resolution_deg: f64,
        final_translation_resolution_mm: f64,
        simplex_iteration_limit: usize,
    ) -> Result<Self> {
        let values = [
            rotation_half_range_deg,
            translation_half_range_mm,
            final_rotation_resolution_deg,
            final_translation_resolution_mm,
        ];
        if values
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(RegistrationError::InvalidInput(format!(
                "rigid-search ranges and resolutions must be finite and positive, got {values:?}"
            )));
        }
        if final_rotation_resolution_deg > rotation_half_range_deg
            || final_translation_resolution_mm > translation_half_range_mm
        {
            return Err(RegistrationError::InvalidInput(format!(
                "rigid-search terminal resolution [{final_rotation_resolution_deg} deg, \
                 {final_translation_resolution_mm} mm] exceeds half-range \
                 [{rotation_half_range_deg} deg, {translation_half_range_mm} mm]"
            )));
        }
        if simplex_iteration_limit == 0 {
            return Err(RegistrationError::InvalidInput(
                "rigid-search simplex iteration limit must be positive".to_owned(),
            ));
        }
        Ok(Self {
            rotation_half_range_radians: rotation_half_range_deg.to_radians(),
            translation_half_range_mm,
            final_rotation_resolution_radians: final_rotation_resolution_deg.to_radians(),
            final_translation_resolution_mm,
            simplex_iteration_limit,
        })
    }

    fn global_bounds(self) -> [f64; PARAMETER_COUNT] {
        [
            self.rotation_half_range_radians,
            self.rotation_half_range_radians,
            self.rotation_half_range_radians,
            self.translation_half_range_mm,
            self.translation_half_range_mm,
            self.translation_half_range_mm,
        ]
    }

    fn terminal_resolution(self) -> [f64; PARAMETER_COUNT] {
        [
            self.final_rotation_resolution_radians,
            self.final_rotation_resolution_radians,
            self.final_rotation_resolution_radians,
            self.final_translation_resolution_mm,
            self.final_translation_resolution_mm,
            self.final_translation_resolution_mm,
        ]
    }
}

/// NMI-capture and local structural-refinement candidates.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct RigidSearchResult {
    /// Transform at the capture objective's optimum.
    pub capture_transform: AffineTransform,
    /// Transform after structural refinement inside the terminal capture cell.
    pub structural_transform: AffineTransform,
    /// Capture-objective value at [`Self::capture_transform`].
    pub capture_score: f64,
    /// Structural-objective value at [`Self::structural_transform`].
    pub structural_score: f64,
    /// Whether capture terminated within one resolution step of a global bound.
    pub capture_saturated: bool,
    /// Whether structural refinement terminated within one convergence width
    /// of its local capture-cell bound.
    pub structural_saturated: bool,
}

/// Search a centroid-anchored six-degree-of-freedom rigid pose.
///
/// Parameters are ZYX Euler rotations in radians followed by residual `[z, y,
/// x]` translations in millimetres. The capture objective runs coarse-to-fine
/// coordinate descent and a bounded Nelder–Mead polish. The structural objective
/// starts at that result and is confined to one terminal capture-resolution
/// cell, preventing a local edge metric from escaping to a remote edge maximum.
/// Returned matrices map fixed to moving coordinates in row-major `[z, y, x]`
/// millimetres.
///
/// # Errors
///
/// Propagates an objective error and returns
/// [`RegistrationError::NumericalFailure`] when either objective emits a
/// non-finite value.
pub fn search_rigid_pose<C, S>(
    fixed_centroid_mm: [f64; 3],
    moving_centroid_mm: [f64; 3],
    config: RigidSearchConfig,
    mut capture_objective: C,
    mut structural_objective: S,
) -> Result<RigidSearchResult>
where
    C: FnMut(&AffineTransform) -> Result<f64>,
    S: FnMut(&AffineTransform) -> Result<f64>,
{
    let matrix = |parameters: &[f64; PARAMETER_COUNT]| {
        rigid_about_centroid(
            euler_zyx(parameters[0], parameters[1], parameters[2]),
            fixed_centroid_mm,
            [
                moving_centroid_mm[0] + parameters[3],
                moving_centroid_mm[1] + parameters[4],
                moving_centroid_mm[2] + parameters[5],
            ],
        )
    };
    let bounds = config.global_bounds();
    let in_global_range = |parameters: &[f64; PARAMETER_COUNT]| {
        parameters
            .iter()
            .zip(bounds.iter())
            .all(|(&value, &bound)| value.abs() <= bound)
    };
    let mut capture_score = |parameters: &[f64; PARAMETER_COUNT]| -> Result<f64> {
        if !in_global_range(parameters) {
            return Ok(f64::NEG_INFINITY);
        }
        finite_score(capture_objective(&matrix(parameters))?, "capture")
    };

    let resolution = config.terminal_resolution();
    let mut parameters = [0.0; PARAMETER_COUNT];
    let mut best_score = capture_score(&parameters)?;
    for level in (0..CAPTURE_LEVELS).rev() {
        let scale = f64::from(1_u32 << level);
        let steps = resolution.map(|value| value * scale);
        loop {
            let mut improved = false;
            for axis in 0..PARAMETER_COUNT {
                for direction in [-1.0, 1.0] {
                    let mut candidate = parameters;
                    candidate[axis] += direction * steps[axis];
                    let candidate_score = capture_score(&candidate)?;
                    if candidate_score > best_score {
                        parameters = candidate;
                        best_score = candidate_score;
                        improved = true;
                    }
                }
            }
            if !improved {
                break;
            }
        }
    }

    let capture_simplex_step = resolution.map(|value| value * 4.0);
    let convergence_width = resolution.map(|value| value / 16.0);
    let capture_parameters = nelder_mead_maximize(
        parameters,
        capture_simplex_step,
        convergence_width,
        config.simplex_iteration_limit,
        &mut capture_score,
    )?;
    let capture_transform = matrix(&capture_parameters);
    let capture_score_value = finite_score(capture_objective(&capture_transform)?, "capture")?;

    let in_structural_range = |candidate: &[f64; PARAMETER_COUNT]| {
        in_global_range(candidate)
            && candidate
                .iter()
                .zip(capture_parameters.iter())
                .zip(resolution.iter())
                .all(|((&value, &center), &radius)| (value - center).abs() <= radius)
    };
    let mut structural_score = |candidate: &[f64; PARAMETER_COUNT]| -> Result<f64> {
        if !in_structural_range(candidate) {
            return Ok(f64::NEG_INFINITY);
        }
        finite_score(structural_objective(&matrix(candidate))?, "structural")
    };
    let structural_step = resolution.map(|value| value / 2.0);
    let structural_parameters = nelder_mead_maximize(
        capture_parameters,
        structural_step,
        convergence_width,
        config.simplex_iteration_limit,
        &mut structural_score,
    )?;
    let structural_transform = matrix(&structural_parameters);
    let structural_score_value =
        finite_score(structural_objective(&structural_transform)?, "structural")?;
    let capture_saturated = touches_bound(
        capture_parameters,
        [0.0; PARAMETER_COUNT],
        bounds,
        resolution,
    );
    let structural_saturated = touches_bound(
        structural_parameters,
        capture_parameters,
        resolution,
        convergence_width,
    );

    Ok(RigidSearchResult {
        capture_transform,
        structural_transform,
        capture_score: capture_score_value,
        structural_score: structural_score_value,
        capture_saturated,
        structural_saturated,
    })
}

fn touches_bound(
    parameters: [f64; PARAMETER_COUNT],
    center: [f64; PARAMETER_COUNT],
    half_range: [f64; PARAMETER_COUNT],
    tolerance: [f64; PARAMETER_COUNT],
) -> bool {
    parameters
        .iter()
        .zip(center.iter())
        .zip(half_range.iter())
        .zip(tolerance.iter())
        .any(|(((&value, &origin), &bound), &width)| {
            ((value - origin).abs() - bound).abs() <= width
        })
}

fn finite_score(score: f64, objective: &str) -> Result<f64> {
    if score.is_finite() {
        Ok(score)
    } else {
        Err(RegistrationError::NumericalFailure(format!(
            "{objective} rigid-search objective returned {score}"
        )))
    }
}

fn euler_zyx(alpha: f64, beta: f64, gamma: f64) -> [[f64; 3]; 3] {
    let (sin_alpha, cos_alpha) = alpha.sin_cos();
    let (sin_beta, cos_beta) = beta.sin_cos();
    let (sin_gamma, cos_gamma) = gamma.sin_cos();
    let z = [
        [1.0, 0.0, 0.0],
        [0.0, cos_alpha, -sin_alpha],
        [0.0, sin_alpha, cos_alpha],
    ];
    let y = [
        [cos_beta, 0.0, sin_beta],
        [0.0, 1.0, 0.0],
        [-sin_beta, 0.0, cos_beta],
    ];
    let x = [
        [cos_gamma, -sin_gamma, 0.0],
        [sin_gamma, cos_gamma, 0.0],
        [0.0, 0.0, 1.0],
    ];
    multiply_3x3(multiply_3x3(z, y), x)
}

fn multiply_3x3(left: [[f64; 3]; 3], right: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut product = [[0.0; 3]; 3];
    for row in 0..3 {
        for column in 0..3 {
            product[row][column] = (0..3)
                .map(|inner| left[row][inner] * right[inner][column])
                .sum();
        }
    }
    product
}

fn rigid_about_centroid(
    rotation: [[f64; 3]; 3],
    fixed_centroid: [f64; 3],
    moving_centroid: [f64; 3],
) -> AffineTransform {
    let rotated_centroid = [
        rotation[0][0] * fixed_centroid[0]
            + rotation[0][1] * fixed_centroid[1]
            + rotation[0][2] * fixed_centroid[2],
        rotation[1][0] * fixed_centroid[0]
            + rotation[1][1] * fixed_centroid[1]
            + rotation[1][2] * fixed_centroid[2],
        rotation[2][0] * fixed_centroid[0]
            + rotation[2][1] * fixed_centroid[1]
            + rotation[2][2] * fixed_centroid[2],
    ];
    let translation = [
        moving_centroid[0] - rotated_centroid[0],
        moving_centroid[1] - rotated_centroid[1],
        moving_centroid[2] - rotated_centroid[2],
    ];
    AffineTransform::new([
        rotation[0][0],
        rotation[0][1],
        rotation[0][2],
        translation[0],
        rotation[1][0],
        rotation[1][1],
        rotation[1][2],
        translation[1],
        rotation[2][0],
        rotation[2][1],
        rotation[2][2],
        translation[2],
        0.0,
        0.0,
        0.0,
        1.0,
    ])
}

fn nelder_mead_maximize<F>(
    start: [f64; PARAMETER_COUNT],
    step: [f64; PARAMETER_COUNT],
    convergence_width: [f64; PARAMETER_COUNT],
    iteration_limit: usize,
    objective: &mut F,
) -> Result<[f64; PARAMETER_COUNT]>
where
    F: FnMut(&[f64; PARAMETER_COUNT]) -> Result<f64>,
{
    let mut simplex = [start; SIMPLEX_VERTEX_COUNT];
    for axis in 0..PARAMETER_COUNT {
        simplex[axis + 1][axis] += step[axis];
    }
    let mut values = [0.0; SIMPLEX_VERTEX_COUNT];
    for (value, vertex) in values.iter_mut().zip(simplex.iter()) {
        *value = objective(vertex)?;
    }

    for _ in 0..iteration_limit {
        let mut order = [0, 1, 2, 3, 4, 5, 6];
        order.sort_by(|&left, &right| values[right].total_cmp(&values[left]));
        let best = order[0];
        let second_worst = order[PARAMETER_COUNT - 1];
        let worst = order[PARAMETER_COUNT];
        if simplex_converged(&simplex, best, convergence_width) {
            break;
        }

        let mut centroid = [0.0; PARAMETER_COUNT];
        for (vertex_index, vertex) in simplex.iter().enumerate() {
            if vertex_index == worst {
                continue;
            }
            for axis in 0..PARAMETER_COUNT {
                centroid[axis] += vertex[axis] / PARAMETER_COUNT as f64;
            }
        }
        let reflected = along(centroid, simplex[worst], 1.0);
        let reflected_value = objective(&reflected)?;
        if reflected_value > values[second_worst] && reflected_value <= values[best] {
            simplex[worst] = reflected;
            values[worst] = reflected_value;
        } else if reflected_value > values[best] {
            let expanded = along(centroid, simplex[worst], 2.0);
            let expanded_value = objective(&expanded)?;
            if expanded_value > reflected_value {
                simplex[worst] = expanded;
                values[worst] = expanded_value;
            } else {
                simplex[worst] = reflected;
                values[worst] = reflected_value;
            }
        } else {
            let coefficient = if reflected_value > values[worst] {
                0.5
            } else {
                -0.5
            };
            let contracted = along(centroid, simplex[worst], coefficient);
            let contracted_value = objective(&contracted)?;
            let threshold = if coefficient > 0.0 {
                reflected_value
            } else {
                values[worst]
            };
            if contracted_value > threshold {
                simplex[worst] = contracted;
                values[worst] = contracted_value;
            } else {
                let best_vertex = simplex[best];
                for vertex_index in 0..SIMPLEX_VERTEX_COUNT {
                    if vertex_index == best {
                        continue;
                    }
                    for axis in 0..PARAMETER_COUNT {
                        simplex[vertex_index][axis] = best_vertex[axis]
                            + 0.5 * (simplex[vertex_index][axis] - best_vertex[axis]);
                    }
                    values[vertex_index] = objective(&simplex[vertex_index])?;
                }
            }
        }
    }

    let best = (0..SIMPLEX_VERTEX_COUNT)
        .max_by(|&left, &right| values[left].total_cmp(&values[right]))
        .expect("invariant: rigid simplex has seven vertices");
    Ok(simplex[best])
}

fn along(
    centroid: [f64; PARAMETER_COUNT],
    worst: [f64; PARAMETER_COUNT],
    coefficient: f64,
) -> [f64; PARAMETER_COUNT] {
    let mut candidate = [0.0; PARAMETER_COUNT];
    for axis in 0..PARAMETER_COUNT {
        candidate[axis] = centroid[axis] + coefficient * (centroid[axis] - worst[axis]);
    }
    candidate
}

fn simplex_converged(
    simplex: &[[f64; PARAMETER_COUNT]; SIMPLEX_VERTEX_COUNT],
    best: usize,
    convergence_width: [f64; PARAMETER_COUNT],
) -> bool {
    (0..PARAMETER_COUNT).all(|axis| {
        simplex
            .iter()
            .all(|vertex| (vertex[axis] - simplex[best][axis]).abs() <= convergence_width[axis])
    })
}

#[cfg(test)]
#[path = "rigid_search_tests.rs"]
mod tests;
