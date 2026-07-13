//! Double-precision statistics for vector confidence-connected membership.

use anyhow::Result;
use leto::{Array2, Storage};
use leto_ops::svd_rank_revealing_with_tolerance;

const SINGULAR_DETERMINANT_THRESHOLD: f64 = 1.0e-6;

/// Compute the population mean and covariance over selected voxel indices.
pub(super) fn mean_covariance<S: AsRef<[f32]>>(
    channels: &[S],
    indices: &[usize],
) -> (Vec<f64>, Vec<f64>) {
    let channel_count = channels.len();
    let sample_count = indices.len() as f64;
    let mut mean = vec![0.0; channel_count];
    let mut covariance = vec![0.0; channel_count * channel_count];
    for &sample_index in indices {
        for row in 0..channel_count {
            let row_value = f64::from(channels[row].as_ref()[sample_index]);
            mean[row] += row_value;
            for column in 0..channel_count {
                covariance[row * channel_count + column] +=
                    row_value * f64::from(channels[column].as_ref()[sample_index]);
            }
        }
    }
    for value in &mut mean {
        *value /= sample_count;
    }
    for row in 0..channel_count {
        for column in 0..channel_count {
            covariance[row * channel_count + column] =
                covariance[row * channel_count + column] / sample_count - mean[row] * mean[column];
        }
    }
    (mean, covariance)
}

/// Compute population statistics from unique samples and their ZeroFlux multiplicities.
pub(super) fn weighted_mean_covariance<S: AsRef<[f32]>>(
    channels: &[S],
    samples: &[(usize, f64)],
) -> (Vec<f64>, Vec<f64>) {
    let channel_count = channels.len();
    let sample_count = samples.iter().map(|(_, weight)| weight).sum::<f64>();
    let mut mean = vec![0.0; channel_count];
    let mut covariance = vec![0.0; channel_count * channel_count];
    for &(sample_index, weight) in samples {
        for row in 0..channel_count {
            let row_value = f64::from(channels[row].as_ref()[sample_index]);
            mean[row] += weight * row_value;
            for column in 0..channel_count {
                covariance[row * channel_count + column] +=
                    weight * row_value * f64::from(channels[column].as_ref()[sample_index]);
            }
        }
    }
    for value in &mut mean {
        *value /= sample_count;
    }
    for row in 0..channel_count {
        for column in 0..channel_count {
            covariance[row * channel_count + column] =
                covariance[row * channel_count + column] / sample_count - mean[row] * mean[column];
        }
    }
    (mean, covariance)
}

/// Build ITK's inverse covariance, including its singular-matrix rule.
pub(super) fn inverse_covariance(covariance: &[f64], channel_count: usize) -> Result<Vec<f64>> {
    let matrix = Array2::from_shape_vec([channel_count, channel_count], covariance.to_vec())?;
    // ITK decides singularity from the determinant, so an independent relative
    // rank cutoff would change its contract before that decision is applied.
    let decomposition = svd_rank_revealing_with_tolerance(&matrix.view(), 0.0)?;
    let determinant = decomposition.singular_values.iter().product::<f64>();
    if determinant <= SINGULAR_DETERMINANT_THRESHOLD {
        return Ok(singular_inverse(channel_count));
    }

    let left = decomposition.left_singular_vectors.storage().as_slice();
    let right = decomposition.right_singular_vectors.storage().as_slice();
    let mut inverse = vec![0.0; channel_count * channel_count];
    for row in 0..channel_count {
        for column in 0..channel_count {
            inverse[row * channel_count + column] = decomposition
                .singular_values
                .iter()
                .enumerate()
                .map(|(component, &singular_value)| {
                    right[row * channel_count + component]
                        * singular_value.recip()
                        * left[column * channel_count + component]
                })
                .sum();
        }
    }
    Ok(inverse)
}

/// Evaluate `(x-mean)^T inverse (x-mean)` using reusable channel scratch.
pub(super) fn mahalanobis_squared<S: AsRef<[f32]>>(
    channels: &[S],
    sample_index: usize,
    mean: &[f64],
    inverse: &[f64],
    delta: &mut [f64],
) -> f64 {
    let channel_count = channels.len();
    for channel in 0..channel_count {
        delta[channel] = f64::from(channels[channel].as_ref()[sample_index]) - mean[channel];
    }
    let mut distance = 0.0;
    for row in 0..channel_count {
        let mut product = 0.0;
        for column in 0..channel_count {
            product += inverse[row * channel_count + column] * delta[column];
        }
        distance += delta[row] * product;
    }
    distance
}

fn singular_inverse(channel_count: usize) -> Vec<f64> {
    let diagonal = f64::MAX.powf(1.0 / 3.0) / channel_count as f64;
    let mut inverse = vec![0.0; channel_count * channel_count];
    for index in 0..channel_count {
        inverse[index * channel_count + index] = diagonal;
    }
    inverse
}
