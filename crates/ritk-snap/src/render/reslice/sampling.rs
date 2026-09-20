//! Validation and sampling helpers for physical-plane reslicing.

use super::{ResliceError, ResliceInterpolation};
use crate::geometry::affine::AffineTransform;
use crate::LoadedVolume;

const MAX_OUTPUT_BYTES: usize = 64 * 1024 * 1024;
const MAX_OUTPUT_SAMPLES: usize = MAX_OUTPUT_BYTES / std::mem::size_of::<f32>();
const MAX_SOURCE_SAMPLES: usize = 256 * 1024 * 1024;
const COORDINATE_TOLERANCE_FACTOR: f64 = 128.0;

pub(super) fn validate_volume(volume: &LoadedVolume) -> Result<AffineTransform, ResliceError> {
    if volume.channels == 0 || volume.shape.contains(&0) {
        return Err(ResliceError::EmptyVolume);
    }
    if volume.channels != 1 {
        return Err(ResliceError::UnsupportedChannels {
            channels: volume.channels,
        });
    }
    let expected = volume
        .shape
        .iter()
        .try_fold(usize::from(volume.channels), |count, extent| {
            count.checked_mul(*extent)
        })
        .ok_or(ResliceError::SampleCountOverflow {
            shape: volume.shape,
            channels: volume.channels,
        })?;
    if volume.data.len() != expected {
        return Err(ResliceError::MalformedPayload {
            expected,
            actual: volume.data.len(),
        });
    }
    AffineTransform::from_parts(volume.origin, volume.direction, volume.spacing)
        .map_err(|source| ResliceError::InvalidGeometry { source })
}

pub(super) fn validate_plane_vectors(
    origin: [f64; 3],
    horizontal: [f64; 3],
    vertical: [f64; 3],
    depth: [f64; 3],
) -> Result<(), ResliceError> {
    if !origin
        .into_iter()
        .chain(horizontal)
        .chain(vertical)
        .chain(depth)
        .all(f64::is_finite)
    {
        return Err(ResliceError::InvalidPlaneBasis);
    }
    if vector_norm(horizontal) == 0.0 || vector_norm(vertical) == 0.0 {
        return Err(ResliceError::InvalidPlaneBasis);
    }
    let cross = cross_product(horizontal, vertical);
    let scale = vector_norm(horizontal) * vector_norm(vertical);
    if vector_norm(cross) <= COORDINATE_TOLERANCE_FACTOR * f64::EPSILON.sqrt() * scale {
        return Err(ResliceError::InvalidPlaneBasis);
    }
    Ok(())
}

pub(super) fn validate_dimensions(
    dimensions: [usize; 2],
    depth_samples: usize,
) -> Result<(), ResliceError> {
    if dimensions.contains(&0) || depth_samples == 0 {
        return Err(ResliceError::EmptyRequest);
    }
    let output_samples = dimensions[0]
        .checked_mul(dimensions[1])
        .ok_or(ResliceError::OutputTooLarge { dimensions })?;
    if output_samples > MAX_OUTPUT_SAMPLES {
        return Err(ResliceError::OutputTooLarge { dimensions });
    }
    let source_samples = output_samples
        .checked_mul(depth_samples)
        .ok_or(ResliceError::WorkTooLarge)?;
    if source_samples > MAX_SOURCE_SAMPLES {
        return Err(ResliceError::WorkTooLarge);
    }
    Ok(())
}

pub(super) fn validate_volume_bounds(
    origin: [f64; 3],
    steps: [[f64; 3]; 3],
    dimensions: [usize; 2],
    depth_samples: usize,
    shape: [usize; 3],
) -> Result<(), ResliceError> {
    let max_offsets = [
        dimensions[0].saturating_sub(1) as f64,
        dimensions[1].saturating_sub(1) as f64,
        depth_samples.saturating_sub(1) as f64,
    ];
    for horizontal in [0.0, max_offsets[0]] {
        for vertical in [0.0, max_offsets[1]] {
            for depth in [0.0, max_offsets[2]] {
                let coordinate = add_scaled(
                    add_scaled(add_scaled(origin, steps[0], horizontal), steps[1], vertical),
                    steps[2],
                    depth,
                );
                validate_coordinate(coordinate, shape)?;
            }
        }
    }
    Ok(())
}

fn validate_coordinate(coordinate: [f64; 3], shape: [usize; 3]) -> Result<(), ResliceError> {
    for (value, extent) in coordinate.into_iter().zip(shape) {
        let maximum = extent.saturating_sub(1) as f64;
        let tolerance = COORDINATE_TOLERANCE_FACTOR * f64::EPSILON * maximum.abs().max(1.0);
        if !value.is_finite() || value < -tolerance || value > maximum + tolerance {
            return Err(ResliceError::OutOfVolume { coordinate });
        }
    }
    Ok(())
}

pub(super) fn sample_volume(
    volume: &LoadedVolume,
    coordinate: [f64; 3],
    interpolation: ResliceInterpolation,
) -> Result<f32, ResliceError> {
    validate_coordinate(coordinate, volume.shape)?;
    let coordinate = clamp_coordinate(coordinate, volume.shape);
    match interpolation {
        ResliceInterpolation::Nearest => {
            let index = coordinate.map(f64::round);
            let index = index.map(|value| {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "validated voxel coordinate is finite and within usize extent"
                )]
                {
                    value as usize
                }
            });
            Ok(volume.pixel_at(index[0], index[1], index[2]))
        }
        ResliceInterpolation::Linear => sample_trilinear(volume, coordinate),
    }
}

fn sample_trilinear(volume: &LoadedVolume, coordinate: [f64; 3]) -> Result<f32, ResliceError> {
    let lower = coordinate.map(f64::floor);
    let upper = [
        lower[0].min(volume.shape[0].saturating_sub(1) as f64),
        lower[1].min(volume.shape[1].saturating_sub(1) as f64),
        lower[2].min(volume.shape[2].saturating_sub(1) as f64),
    ];
    let upper_index = upper.map(|value| {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "validated voxel coordinate is finite and within usize extent"
        )]
        {
            value as usize
        }
    });
    let next = [
        (upper_index[0] + 1).min(volume.shape[0].saturating_sub(1)),
        (upper_index[1] + 1).min(volume.shape[1].saturating_sub(1)),
        (upper_index[2] + 1).min(volume.shape[2].saturating_sub(1)),
    ];
    let weights = [
        (coordinate[0] - upper[0]) as f32,
        (coordinate[1] - upper[1]) as f32,
        (coordinate[2] - upper[2]) as f32,
    ];
    let x00 = lerp(
        volume.pixel_at(upper_index[0], upper_index[1], upper_index[2]),
        volume.pixel_at(next[0], upper_index[1], upper_index[2]),
        weights[0],
    );
    let x01 = lerp(
        volume.pixel_at(upper_index[0], upper_index[1], next[2]),
        volume.pixel_at(next[0], upper_index[1], next[2]),
        weights[0],
    );
    let x10 = lerp(
        volume.pixel_at(upper_index[0], next[1], upper_index[2]),
        volume.pixel_at(next[0], next[1], upper_index[2]),
        weights[0],
    );
    let x11 = lerp(
        volume.pixel_at(upper_index[0], next[1], next[2]),
        volume.pixel_at(next[0], next[1], next[2]),
        weights[0],
    );
    let y0 = lerp(x00, x10, weights[1]);
    let y1 = lerp(x01, x11, weights[1]);
    Ok(lerp(y0, y1, weights[2]))
}

fn lerp(first: f32, second: f32, weight: f32) -> f32 {
    first + (second - first) * weight
}

fn clamp_coordinate(coordinate: [f64; 3], shape: [usize; 3]) -> [f64; 3] {
    [
        coordinate[0].clamp(0.0, shape[0].saturating_sub(1) as f64),
        coordinate[1].clamp(0.0, shape[1].saturating_sub(1) as f64),
        coordinate[2].clamp(0.0, shape[2].saturating_sub(1) as f64),
    ]
}

pub(super) fn patient_step_to_voxel(
    transform: &AffineTransform,
    origin: [f64; 3],
    step: [f64; 3],
) -> [f64; 3] {
    let shifted = [
        origin[0] + step[0],
        origin[1] + step[1],
        origin[2] + step[2],
    ];
    let first = transform.patient_to_voxel(origin);
    let second = transform.patient_to_voxel(shifted);
    [
        second[0] - first[0],
        second[1] - first[1],
        second[2] - first[2],
    ]
}

pub(super) fn voxel_step(transform: &AffineTransform, base: [f64; 3], axis: usize) -> [f64; 3] {
    let first = transform.voxel_to_patient(base);
    let mut next = base;
    next[axis] += 1.0;
    let second = transform.voxel_to_patient(next);
    [
        second[0] - first[0],
        second[1] - first[1],
        second[2] - first[2],
    ]
}

pub(super) fn axis_index(index: usize, axis: usize) -> [f64; 3] {
    let mut coordinate = [0.0; 3];
    coordinate[axis] = index as f64;
    coordinate
}

pub(super) fn add_scaled(first: [f64; 3], second: [f64; 3], scale: f64) -> [f64; 3] {
    [
        first[0] + second[0] * scale,
        first[1] + second[1] * scale,
        first[2] + second[2] * scale,
    ]
}

pub(super) fn vector_norm(vector: [f64; 3]) -> f64 {
    vector
        .into_iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt()
}

fn cross_product(first: [f64; 3], second: [f64; 3]) -> [f64; 3] {
    [
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    ]
}
