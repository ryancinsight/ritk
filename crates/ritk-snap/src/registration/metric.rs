//! Estimate physical sampling domains and score them with normalized mutual information.
use super::volume::center;
use crate::geometry::affine::AffineTransform as Grid;
use crate::render::reslice::sampling::sample_volume;
use crate::render::ResliceInterpolation;
use crate::LoadedVolume;
use anyhow::{ensure, Context, Result};
use ritk_registration::classical::error::{RegistrationError, Result as MetricResult};
use ritk_registration::classical::spatial::transform_point;
use ritk_registration::AffineTransform;
use ritk_registration::{
    HistogramEstimator, IntensityRange, MutualInformationMetric, NmiNormalization,
};
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Stage {
    Capture,
    Refinement,
}

pub(super) struct Samples {
    points: Vec<[f64; 3]>,
    fixed: Vec<f64>,
    metric: MutualInformationMetric,
    values: Vec<f64>,
    valid: Vec<bool>,
}

/// Estimate per-modality histogram bounds and a coarse foreground centroid.
/// Corner intensities estimate the background even for inverted contrasts.
pub(super) struct Input<'a> {
    pub(super) volume: &'a LoadedVolume,
    pub(super) grid: Grid,
    histogram: IntensityRange,
    background: f64,
    threshold: f64,
    pub(super) center: [f64; 3],
}

fn indices(n: usize, limit: usize) -> impl Iterator<Item = usize> {
    let count = n.min(limit);
    (0..count).map(move |i| (2 * i + 1) * n / (2 * count))
}

impl<'a> Input<'a> {
    pub(super) fn new(volume: &'a LoadedVolume, grid: Grid) -> Result<Self> {
        ensure!(
            volume.data.iter().all(|v| v.is_finite()),
            "Series contains non-finite voxels."
        );
        let [nz, ny, nx] = volume.shape;
        let mut voxels = Vec::new();
        for z in indices(nz, 32) {
            for y in indices(ny, 32) {
                for x in indices(nx, 32) {
                    voxels.push((volume.pixel_at(z, y, x), [z as f64, y as f64, x as f64]));
                }
            }
        }
        voxels.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        let last = voxels.len() - 1;
        let mut low = f64::from(voxels[last * 2 / 100].0);
        let mut high = f64::from(voxels[last * 98 / 100].0);
        if high <= low {
            low = voxels[0].0.into();
            high = voxels[last].0.into();
        }
        // Restrict the CT histogram to the air-to-soft-tissue contrast useful across modalities.
        if volume.modality.as_deref() == Some("CT") && low < -800. && high > 300. {
            low = -1000.;
            high = 300.;
        }
        let mut corners = [0.; 8];
        for (i, corner) in corners.iter_mut().enumerate() {
            *corner = volume.pixel_at(
                if i & 1 == 0 { 0 } else { nz - 1 },
                if i & 2 == 0 { 0 } else { ny - 1 },
                if i & 4 == 0 { 0 } else { nx - 1 },
            );
        }
        corners.sort_unstable_by(f32::total_cmp);
        let background = f64::from(corners[3].midpoint(corners[4]));
        let threshold = (high - low) * 0.15;
        let mut sum = [0.; 3];
        let mut count = 0u32;
        for &(v, coord) in &voxels {
            if (f64::from(v) - background).abs() > threshold {
                for (total, value) in sum.iter_mut().zip(coord) {
                    *total += value;
                }
                count += 1;
            }
        }
        let histogram =
            IntensityRange::try_new(low, high).context("series needs intensity variation")?;
        let center = if count > 0 {
            grid.voxel_to_patient(sum.map(|v| v / f64::from(count)))
        } else {
            center(volume, grid)
        };
        Ok(Self {
            volume,
            grid,
            histogram,
            background,
            threshold,
            center,
        })
    }
}

impl Samples {
    pub(super) fn new(
        fixed: &Input<'_>,
        moving: &Input<'_>,
        anchor: AffineTransform,
        stage: Stage,
    ) -> Result<Self> {
        let interior = stage == Stage::Refinement;
        let limit = if interior { 64 } else { 28 };
        let bins = if interior && fixed.volume.data.len() >= 64 * 64 * 4 {
            64
        } else {
            32
        };
        let metric = MutualInformationMetric::with_ranges(
            bins,
            fixed.histogram,
            moving.histogram,
            NmiNormalization::MeanEntropy,
            HistogramEstimator::MovingLinearPartialVolume,
        )?;
        let mut points = Vec::new();
        let mut values = Vec::new();
        let mut sample_index = 0usize;
        let [nz, ny, nx] = fixed.volume.shape;
        // Sample bin centres in all three dimensions to avoid locking to one row
        // or slice when the array strides share a common factor.
        for z in indices(nz, limit) {
            for y in indices(ny, limit) {
                for x in indices(nx, limit) {
                    sample_index += 1;
                    let voxel = [z as f64, y as f64, x as f64];
                    let p = fixed.grid.voxel_to_patient(voxel);
                    let coord = moving.grid.patient_to_voxel(transform_point(&p, &anchor));
                    let value = f64::from(fixed.volume.pixel_at(z, y, x));
                    let foreground = (value - fixed.background).abs() > fixed.threshold;
                    let in_core = !interior
                        || coord.iter().zip(moving.volume.shape).all(|(&v, n)| {
                            let margin = (0.1 * (n - 1) as f64).min(4.0);
                            v >= margin && v <= (n - 1) as f64 - margin
                        });
                    if in_core
                        && (foreground || sample_index.is_multiple_of(5))
                        && sample_volume(moving.volume, coord, ResliceInterpolation::Nearest)
                            .is_ok()
                    {
                        points.push(p);
                        values.push(value);
                    }
                }
            }
        }
        ensure!(
            points.len() >= 64,
            "Too few overlapping samples for 3D registration."
        );
        let count = points.len();
        ensure!(
            values.iter().any(|v| *v != values[0]),
            "No intensity variation in the overlapping region."
        );
        Ok(Self {
            points,
            fixed: values,
            metric,
            values: vec![0.; count],
            valid: vec![false; count],
        })
    }

    pub(super) fn score(
        &mut self,
        moving: &Input<'_>,
        pose: &AffineTransform,
        cancelled: &AtomicBool,
    ) -> MetricResult<f64> {
        if cancelled.load(Ordering::Relaxed) {
            return Err(RegistrationError::InvalidInput(
                "Registration cancelled.".into(),
            ));
        }
        let mut valid = 0u32;
        for ((point, sample), included) in self
            .points
            .iter()
            .zip(&mut self.values)
            .zip(&mut self.valid)
        {
            let coord = moving.grid.patient_to_voxel(transform_point(point, pose));
            match sample_volume(moving.volume, coord, ResliceInterpolation::Linear) {
                Ok(value) => {
                    valid += 1;
                    *sample = f64::from(value);
                    *included = true;
                }
                Err(crate::render::ResliceError::OutOfVolume { .. }) => {
                    *included = false;
                }
                Err(error) => return Err(RegistrationError::InvalidInput(error.to_string())),
            }
        }
        let coverage = f64::from(valid) / self.points.len() as f64;
        if coverage < 0.70 {
            return Ok(-1.0);
        }
        Ok(self
            .metric
            .compute_masked_samples(&self.fixed, &self.values, Some(&self.valid))?
            * coverage.sqrt())
    }
}
