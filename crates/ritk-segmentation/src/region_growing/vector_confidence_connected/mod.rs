//! Vector confidence-connected region growing.
//!
//! Starting from seed neighborhoods, the filter estimates a vector mean and
//! population covariance. It admits face-connected voxels whose Mahalanobis
//! distance is no greater than the configured multiplier, recomputing region
//! statistics for the configured number of iterations. Statistical arithmetic
//! is `f64` to match ITK's membership-function contract; image storage remains
//! native `f32` and is read through borrowed channel slices.

mod flood;
mod statistics;

use anyhow::{bail, ensure, Result};
use ritk_core::spatial::{Direction, Point, Spacing, VoxelIndex};
use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec, rebuild};

use flood::FloodWorkspace;
use statistics::{
    inverse_covariance, mahalanobis_squared, mean_covariance, weighted_mean_covariance,
};

/// Validated vector confidence-connected configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VectorConfidenceConnectedConfig {
    multiplier: f64,
    iterations: u32,
    initial_radius: usize,
    replace_value: f32,
}

impl VectorConfidenceConnectedConfig {
    /// Construct validated vector confidence-connected configuration.
    ///
    /// # Errors
    ///
    /// Returns an error unless the multiplier is finite and nonnegative and the
    /// replacement value is finite.
    pub fn new(
        multiplier: f64,
        iterations: u32,
        initial_radius: usize,
        replace_value: f32,
    ) -> Result<Self> {
        ensure!(
            multiplier.is_finite() && multiplier > 0.0,
            "vector confidence multiplier must be finite and positive, got {multiplier}"
        );
        ensure!(
            replace_value.is_finite(),
            "vector confidence replacement must be finite, got {replace_value}"
        );
        Ok(Self {
            multiplier,
            iterations,
            initial_radius,
            replace_value,
        })
    }
}

impl Default for VectorConfidenceConnectedConfig {
    fn default() -> Self {
        Self {
            multiplier: 2.5,
            iterations: 4,
            initial_radius: 1,
            replace_value: 1.0,
        }
    }
}

/// Vector confidence-connected region-growing filter.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorConfidenceConnectedFilter {
    seeds: Box<[VoxelIndex]>,
    config: VectorConfidenceConnectedConfig,
}

impl VectorConfidenceConnectedFilter {
    /// Construct a filter from seed indices and validated configuration.
    pub fn new<I, S>(seeds: I, config: VectorConfidenceConnectedConfig) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<VoxelIndex>,
    {
        Self {
            seeds: seeds.into_iter().map(Into::into).collect(),
            config,
        }
    }

    /// Apply the filter to legacy scalar component images.
    ///
    /// # Errors
    ///
    /// Returns an error for absent channels, mismatched channel geometry,
    /// invalid image storage or non-finite samples. Seeds outside the image are
    /// ignored, matching ITK's contract.
    pub fn apply<B: Backend>(&self, channels: &[&Image<f32, B, 3>]) -> Result<Image<f32, B, 3>> {
        let Some(first) = channels.first().copied() else {
            bail!("vector confidence requires at least one channel");
        };
        let reference = Geometry::new(
            first.shape(),
            first.origin(),
            first.spacing(),
            first.direction(),
        );
        let mut samples = Vec::with_capacity(channels.len());
        for (index, &channel) in channels.iter().enumerate() {
            reference.ensure_matches(
                index,
                channel.shape(),
                channel.origin(),
                channel.spacing(),
                channel.direction(),
            )?;
            samples.push(extract_vec(channel)?.0);
        }
        let labels = segment_values(&samples, reference.dimensions, &self.seeds, self.config)?;
        Ok(rebuild(labels, reference.dimensions, first))
    }

    /// Apply the filter directly to Coeus-native scalar component images.
    ///
    /// # Errors
    ///
    /// Returns an error for absent channels, mismatched channel geometry,
    /// inaccessible storage, non-finite samples, or output construction failure.
    /// Seeds outside the image are ignored, matching ITK's contract.
    pub fn apply_native<B>(
        &self,
        channels: &[&ritk_image::native::Image<f32, B, 3>],
        backend: &B,
    ) -> Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let Some(first) = channels.first().copied() else {
            bail!("vector confidence requires at least one channel");
        };
        let reference = Geometry::new(
            first.shape(),
            first.origin(),
            first.spacing(),
            first.direction(),
        );
        let mut samples = Vec::with_capacity(channels.len());
        for (index, &channel) in channels.iter().enumerate() {
            reference.ensure_matches(
                index,
                channel.shape(),
                channel.origin(),
                channel.spacing(),
                channel.direction(),
            )?;
            samples.push(channel.data_slice()?);
        }
        crate::native_output::from_values(
            first,
            segment_values(&samples, reference.dimensions, &self.seeds, self.config)?,
            backend,
        )
    }
}

fn segment_values<S: AsRef<[f32]>>(
    channels: &[S],
    dimensions: [usize; 3],
    seeds: &[VoxelIndex],
    config: VectorConfidenceConnectedConfig,
) -> Result<Vec<f32>> {
    ensure!(
        !channels.is_empty(),
        "vector confidence requires at least one channel"
    );
    ensure!(
        dimensions.iter().all(|&extent| extent > 0),
        "vector confidence requires nonzero dimensions, got {dimensions:?}"
    );
    let voxel_count = dimensions
        .into_iter()
        .try_fold(1usize, |count, extent| count.checked_mul(extent))
        .ok_or_else(|| anyhow::anyhow!("vector confidence voxel count overflows"))?;
    validate_samples(channels, voxel_count)?;
    let seeds: Vec<_> = seeds
        .iter()
        .copied()
        .filter(|seed| {
            seed[0] < dimensions[0] && seed[1] < dimensions[1] && seed[2] < dimensions[2]
        })
        .collect();
    if seeds.is_empty() {
        return Ok(vec![0.0; voxel_count]);
    }

    let channel_count = channels.len();
    channel_count
        .checked_mul(channel_count)
        .and_then(|square| square.checked_mul(2))
        .ok_or_else(|| anyhow::anyhow!("vector confidence channel matrix size overflows"))?;
    let mut mean = vec![0.0; channel_count];
    let mut covariance = vec![0.0; channel_count * channel_count];
    let mut neighborhood = Vec::new();
    for &seed in &seeds {
        collect_neighborhood(seed, dimensions, config.initial_radius, &mut neighborhood);
        let (seed_mean, seed_covariance) = weighted_mean_covariance(channels, &neighborhood);
        for row in 0..channel_count {
            mean[row] += seed_mean[row];
            for column in 0..channel_count {
                covariance[row * channel_count + column] +=
                    seed_covariance[row * channel_count + column];
            }
        }
    }
    let seed_count = seeds.len() as f64;
    for row in 0..channel_count {
        mean[row] /= seed_count;
        for column in 0..channel_count {
            covariance[row * channel_count + column] /= seed_count;
        }
    }

    let mut inverse = inverse_covariance(&covariance, channel_count)?;
    let mut threshold = config.multiplier;
    let mut delta = vec![0.0; channel_count];
    for &seed in &seeds {
        let distance = mahalanobis_squared(
            channels,
            flatten(seed, dimensions),
            &mean,
            &inverse,
            &mut delta,
        )
        .max(0.0)
        .sqrt();
        threshold = threshold.max(distance);
    }

    let mut flood = FloodWorkspace::new(voxel_count, channel_count);
    flood.fill(channels, dimensions, &seeds, &mean, &inverse, threshold);
    for _ in 0..config.iterations {
        if flood.visit_order().is_empty() {
            break;
        }
        (mean, covariance) = mean_covariance(channels, flood.visit_order());
        inverse = inverse_covariance(&covariance, channel_count)?;
        flood.fill(channels, dimensions, &seeds, &mean, &inverse, threshold);
    }

    Ok(flood
        .mask()
        .iter()
        .map(|&inside| if inside { config.replace_value } else { 0.0 })
        .collect())
}

fn validate_samples<S: AsRef<[f32]>>(channels: &[S], voxel_count: usize) -> Result<()> {
    for (channel_index, channel) in channels.iter().enumerate() {
        let samples = channel.as_ref();
        ensure!(
            samples.len() == voxel_count,
            "vector confidence channel {channel_index} length {} != voxel count {voxel_count}",
            samples.len()
        );
        if let Some((sample_index, sample)) = samples
            .iter()
            .copied()
            .enumerate()
            .find(|(_, sample)| !sample.is_finite())
        {
            bail!(
                "vector confidence channel {channel_index} sample {sample_index} must be finite, got {sample}"
            );
        }
    }
    Ok(())
}

fn collect_neighborhood(
    seed: VoxelIndex,
    dimensions: [usize; 3],
    radius: usize,
    samples: &mut Vec<(usize, f64)>,
) {
    samples.clear();
    let z_weights = zero_flux_weights(seed[0], dimensions[0], radius);
    let y_weights = zero_flux_weights(seed[1], dimensions[1], radius);
    let x_weights = zero_flux_weights(seed[2], dimensions[2], radius);
    for &(z, z_weight) in &z_weights {
        for &(y, y_weight) in &y_weights {
            for &(x, x_weight) in &x_weights {
                samples.push((
                    (z * dimensions[1] + y) * dimensions[2] + x,
                    z_weight * y_weight * x_weight,
                ));
            }
        }
    }
}

fn zero_flux_weights(center: usize, extent: usize, radius: usize) -> Vec<(usize, f64)> {
    if extent == 1 {
        return vec![(0, 2.0 * radius as f64 + 1.0)];
    }
    let start = center.saturating_sub(radius);
    let end = center.saturating_add(radius).min(extent - 1);
    (start..=end)
        .map(|index| {
            let weight = if index == 0 {
                radius.saturating_sub(center) as f64 + 1.0
            } else if index == extent - 1 {
                radius.saturating_sub(extent - 1 - center) as f64 + 1.0
            } else {
                1.0
            };
            (index, weight)
        })
        .collect()
}

fn flatten(seed: VoxelIndex, dimensions: [usize; 3]) -> usize {
    (seed[0] * dimensions[1] + seed[1]) * dimensions[2] + seed[2]
}

#[derive(Clone, Copy)]
struct Geometry<'a> {
    dimensions: [usize; 3],
    origin: &'a Point<3>,
    spacing: &'a Spacing<3>,
    direction: &'a Direction<3>,
}

impl<'a> Geometry<'a> {
    fn new(
        dimensions: [usize; 3],
        origin: &'a Point<3>,
        spacing: &'a Spacing<3>,
        direction: &'a Direction<3>,
    ) -> Self {
        Self {
            dimensions,
            origin,
            spacing,
            direction,
        }
    }

    fn ensure_matches(
        self,
        channel_index: usize,
        dimensions: [usize; 3],
        origin: &Point<3>,
        spacing: &Spacing<3>,
        direction: &Direction<3>,
    ) -> Result<()> {
        ensure!(
            dimensions == self.dimensions,
            "vector confidence channel {channel_index} shape {dimensions:?} != {:?}",
            self.dimensions
        );
        ensure!(
            origin == self.origin && spacing == self.spacing && direction == self.direction,
            "vector confidence channel {channel_index} geometry differs from channel 0"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests;
