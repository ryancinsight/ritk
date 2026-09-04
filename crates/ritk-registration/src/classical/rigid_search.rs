//! Bounded multimodal rigid-pose search in physical coordinates.

mod anchor;
mod config;
mod pose;

pub use anchor::RigidSearchAnchor;
pub use config::{RigidSearchConfig, RigidSearchResult};

use self::pose::{euler_zyx, multiply_3x3, rigid_about_centroid};
use super::error::{RegistrationError, Result};
use crate::types::AffineTransform;
use std::num::NonZeroU8;

pub(super) const PARAMETER_COUNT: usize = 6;
const SIMPLEX_VERTEX_COUNT: usize = PARAMETER_COUNT + 1;
// Four levels span an eightfold coarse cell followed by three bisections to the
// requested terminal resolution.
const CAPTURE_LEVELS: u32 = 4;
/// Search a six-degree-of-freedom residual around a full rigid anchor.
///
/// Parameters are ZYX Euler rotations in radians followed by residual `[z, y,
/// x]` translations in millimetres. The capture objective runs coarse-to-fine
/// coordinate descent and a bounded Nelder–Mead polish. The structural objective
/// starts at that result and is confined to the configured nonzero number of
/// terminal capture-resolution cells and the global bounds, preventing a local
/// metric from escaping to an unbounded remote maximum.
/// Returned matrices map fixed to moving coordinates in row-major `[z, y, x]`
/// millimetres.
///
/// # Errors
///
/// Propagates an objective error and returns
/// [`RegistrationError::NumericalFailure`] when a candidate cannot form a
/// finite transform or either objective emits a non-finite value.
pub fn search_rigid_pose<C, S>(
    anchor: RigidSearchAnchor,
    config: RigidSearchConfig,
    mut capture_objective: C,
    mut structural_objective: S,
) -> Result<RigidSearchResult>
where
    C: FnMut(&AffineTransform) -> Result<f64>,
    S: FnMut(&AffineTransform) -> Result<f64>,
{
    let anchor_matrix = anchor.transform.as_array();
    let anchor_rotation = [
        [anchor_matrix[0], anchor_matrix[1], anchor_matrix[2]],
        [anchor_matrix[4], anchor_matrix[5], anchor_matrix[6]],
        [anchor_matrix[8], anchor_matrix[9], anchor_matrix[10]],
    ];
    let matrix = |parameters: &[f64; PARAMETER_COUNT]| -> Result<AffineTransform> {
        if parameters.iter().all(|&parameter| parameter == 0.0) {
            return Ok(anchor.transform);
        }
        let residual_rotation = euler_zyx(parameters[0], parameters[1], parameters[2]);
        let transform = rigid_about_centroid(
            multiply_3x3(anchor_rotation, residual_rotation),
            anchor.fixed_center_mm,
            [
                anchor.moving_center_mm[0] + parameters[3],
                anchor.moving_center_mm[1] + parameters[4],
                anchor.moving_center_mm[2] + parameters[5],
            ],
        );
        if transform.as_array().iter().all(|value| value.is_finite()) {
            Ok(transform)
        } else {
            Err(RegistrationError::NumericalFailure(
                "rigid-search candidate produced a non-finite transform".to_owned(),
            ))
        }
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
        finite_score(capture_objective(&matrix(parameters)?)?, "capture")
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
                    candidate[axis] = finite_offset(
                        parameters[axis],
                        direction * steps[axis],
                        [-bounds[axis], bounds[axis]],
                    );
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
        bounds.map(|bound| [-bound, bound]),
        &mut capture_score,
    )?;
    let capture_transform = matrix(&capture_parameters)?;
    let capture_score_value = finite_score(capture_objective(&capture_transform)?, "capture")?;

    let (structural_bounds, structural_step) = structural_interval_and_step(
        capture_parameters,
        bounds,
        resolution,
        config.structural_half_range_cells,
    );
    let in_structural_range = |candidate: &[f64; PARAMETER_COUNT]| {
        candidate
            .iter()
            .zip(structural_bounds.iter())
            .all(|(&value, &[lower, upper])| value >= lower && value <= upper)
    };
    let mut structural_score = |candidate: &[f64; PARAMETER_COUNT]| -> Result<f64> {
        if !in_structural_range(candidate) {
            return Ok(f64::NEG_INFINITY);
        }
        finite_score(structural_objective(&matrix(candidate)?)?, "structural")
    };
    let structural_parameters = nelder_mead_maximize(
        capture_parameters,
        structural_step,
        convergence_width,
        config.simplex_iteration_limit,
        structural_bounds,
        &mut structural_score,
    )?;
    let structural_transform = matrix(&structural_parameters)?;
    let structural_score_value =
        finite_score(structural_objective(&structural_transform)?, "structural")?;
    let capture_saturated = touches_bound(
        capture_parameters,
        [0.0; PARAMETER_COUNT],
        bounds,
        resolution,
    );
    let structural_saturated =
        touches_interval_bound(structural_parameters, structural_bounds, convergence_width);

    Ok(RigidSearchResult {
        capture_transform,
        structural_transform,
        capture_score: capture_score_value,
        structural_score: structural_score_value,
        capture_saturated,
        structural_saturated,
    })
}

fn structural_interval_and_step(
    center: [f64; PARAMETER_COUNT],
    global_half_range: [f64; PARAMETER_COUNT],
    resolution: [f64; PARAMETER_COUNT],
    cells: NonZeroU8,
) -> ([[f64; 2]; PARAMETER_COUNT], [f64; PARAMETER_COUNT]) {
    let cell_count = f64::from(cells.get());
    let intervals = std::array::from_fn(|axis| {
        let requested_radius = resolution[axis] * cell_count;
        [
            (center[axis] - requested_radius).max(-global_half_range[axis]),
            (center[axis] + requested_radius).min(global_half_range[axis]),
        ]
    });
    let nominal_step = std::array::from_fn(|axis| {
        (resolution[axis] * (cell_count / 2.0)).min(global_half_range[axis])
    });
    let steps = signed_simplex_step(center, intervals, nominal_step);
    (intervals, steps)
}

fn signed_simplex_step(
    center: [f64; PARAMETER_COUNT],
    intervals: [[f64; 2]; PARAMETER_COUNT],
    nominal_step: [f64; PARAMETER_COUNT],
) -> [f64; PARAMETER_COUNT] {
    std::array::from_fn(|axis| {
        let [lower, upper] = intervals[axis];
        let positive_room = upper - center[axis];
        let negative_room = center[axis] - lower;
        let step_magnitude = nominal_step[axis].min(positive_room.max(negative_room));
        if positive_room >= negative_room {
            step_magnitude
        } else {
            -step_magnitude
        }
    })
}

fn finite_offset(value: f64, offset: f64, [lower, upper]: [f64; 2]) -> f64 {
    let candidate = value + offset;
    if candidate.is_finite() {
        candidate
    } else {
        debug_assert!(
            !candidate.is_nan(),
            "invariant: finite value plus signed infinite offset has a defined sign"
        );
        if candidate.is_sign_negative() {
            lower
        } else {
            upper
        }
    }
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

fn touches_interval_bound(
    parameters: [f64; PARAMETER_COUNT],
    intervals: [[f64; 2]; PARAMETER_COUNT],
    tolerance: [f64; PARAMETER_COUNT],
) -> bool {
    parameters
        .iter()
        .zip(intervals.iter())
        .zip(tolerance.iter())
        .any(|((&value, &[lower, upper]), &width)| {
            (value - lower).abs() <= width || (upper - value).abs() <= width
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

fn nelder_mead_maximize<F>(
    start: [f64; PARAMETER_COUNT],
    step: [f64; PARAMETER_COUNT],
    convergence_width: [f64; PARAMETER_COUNT],
    iteration_limit: usize,
    intervals: [[f64; 2]; PARAMETER_COUNT],
    objective: &mut F,
) -> Result<[f64; PARAMETER_COUNT]>
where
    F: FnMut(&[f64; PARAMETER_COUNT]) -> Result<f64>,
{
    let mut simplex = [start; SIMPLEX_VERTEX_COUNT];
    for axis in 0..PARAMETER_COUNT {
        simplex[axis + 1][axis] = finite_offset(start[axis], step[axis], intervals[axis]);
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
        let reflected = along(centroid, simplex[worst], 1.0, intervals);
        let reflected_value = objective(&reflected)?;
        if reflected_value > values[second_worst] && reflected_value <= values[best] {
            simplex[worst] = reflected;
            values[worst] = reflected_value;
        } else if reflected_value > values[best] {
            let expanded = along(centroid, simplex[worst], 2.0, intervals);
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
            let contracted = along(centroid, simplex[worst], coefficient, intervals);
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
                    simplex[vertex_index] =
                        towards(best_vertex, simplex[vertex_index], 0.5, intervals);
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
    intervals: [[f64; 2]; PARAMETER_COUNT],
) -> [f64; PARAMETER_COUNT] {
    let direction = std::array::from_fn(|axis| centroid[axis] - worst[axis]);
    finite_affine_offset(centroid, direction, coefficient, intervals)
}

fn towards(
    start: [f64; PARAMETER_COUNT],
    target: [f64; PARAMETER_COUNT],
    fraction: f64,
    intervals: [[f64; 2]; PARAMETER_COUNT],
) -> [f64; PARAMETER_COUNT] {
    let direction = std::array::from_fn(|axis| target[axis] - start[axis]);
    finite_affine_offset(start, direction, fraction, intervals)
}

fn finite_affine_offset(
    start: [f64; PARAMETER_COUNT],
    direction: [f64; PARAMETER_COUNT],
    coefficient: f64,
    intervals: [[f64; 2]; PARAMETER_COUNT],
) -> [f64; PARAMETER_COUNT] {
    std::array::from_fn(|axis| {
        finite_offset(start[axis], coefficient * direction[axis], intervals[axis])
    })
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
