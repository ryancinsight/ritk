use std::num::NonZeroU8;

use super::{
    discrete_bin_weights, entropy, linear_bin_weights, HistogramEstimator, MutualInformationMetric,
    NmiNormalization,
};
use crate::classical::error::{RegistrationError, Result};

/// Reusable location-conditioned mutual-information metric.
///
/// Each selected sample carries a fixed spatial-region label. Entropies are
/// conditioned on that label before normalization, preserving coarse spatial
/// correspondence that a single global joint histogram discards. Histogram
/// storage is allocated once and cleared between evaluations so an optimizer
/// performs no per-pose allocation.
///
/// The conditional entropies follow Toews and Wells, *Bayesian Registration via
/// Local Image Regions*, §3.2, equations 8–9:
/// `H(F|R) = Σ p(r) H(F|r)` and likewise for the moving and joint histograms.
/// The configured [`NmiNormalization`] is then applied to those three
/// conditional entropies. See <https://doi.org/10.1007/978-3-642-02498-6_36>.
///
/// # Examples
///
/// ```
/// use std::num::NonZeroU8;
/// use ritk_registration::classical::{
///     HistogramEstimator, IntensityRange, MutualInformationMetric,
///     NmiNormalization, SpatiallyConditionedMutualInformationMetric,
/// };
///
/// let range = IntensityRange::try_new(0.0, 1.0)?;
/// let base = MutualInformationMetric::with_ranges(
///     2,
///     range,
///     range,
///     NmiNormalization::JointEntropy,
///     HistogramEstimator::Discrete,
/// )?;
/// let mut metric = SpatiallyConditionedMutualInformationMetric::try_new(
///     base,
///     NonZeroU8::new(2).expect("two regions are nonzero"),
/// )?;
/// let nmi = metric.compute_masked_samples(
///     &[0.0, 1.0, 0.0, 1.0],
///     &[0.0, 1.0, 0.0, 1.0],
///     &[0, 0, 1, 1],
///     None,
/// )?;
/// assert_eq!(nmi, 2.0);
/// # Ok::<(), ritk_registration::classical::RegistrationError>(())
/// ```
#[derive(Debug)]
pub struct SpatiallyConditionedMutualInformationMetric {
    metric: MutualInformationMetric,
    region_count: NonZeroU8,
    joint: Vec<f64>,
    fixed_marginal: Vec<f64>,
    moving_marginal: Vec<f64>,
    mass: Vec<f64>,
}

impl SpatiallyConditionedMutualInformationMetric {
    /// Allocate a reusable metric for `region_count` fixed spatial regions.
    ///
    /// # Errors
    ///
    /// Returns [`RegistrationError::InvalidInput`] when the required histogram
    /// lengths overflow or their storage cannot be reserved.
    pub fn try_new(metric: MutualInformationMetric, region_count: NonZeroU8) -> Result<Self> {
        let regions = usize::from(region_count.get());
        let joint_per_region = metric.bins.checked_mul(metric.bins).ok_or_else(|| {
            RegistrationError::InvalidInput(format!(
                "mutual-information bin count {} overflows histogram length",
                metric.bins
            ))
        })?;
        let joint_length = joint_per_region.checked_mul(regions).ok_or_else(|| {
            RegistrationError::InvalidInput(format!(
                "{regions} spatial regions overflow the mutual-information histogram length"
            ))
        })?;
        let marginal_length = metric.bins.checked_mul(regions).ok_or_else(|| {
            RegistrationError::InvalidInput(format!(
                "{regions} spatial regions overflow the mutual-information marginal length"
            ))
        })?;
        Ok(Self {
            metric,
            region_count,
            joint: zeroed_histogram(joint_length, "joint")?,
            fixed_marginal: zeroed_histogram(marginal_length, "fixed marginal")?,
            moving_marginal: zeroed_histogram(marginal_length, "moving marginal")?,
            mass: zeroed_histogram(regions, "region mass")?,
        })
    }

    /// Return the number of fixed spatial regions.
    #[must_use]
    pub fn region_count(&self) -> NonZeroU8 {
        self.region_count
    }

    /// Evaluate samples after conditioning their entropies on spatial region.
    ///
    /// Region labels are zero-based and must be below [`Self::region_count`].
    /// Empty regions contribute no entropy; at least one sample must be selected.
    ///
    /// # Errors
    ///
    /// Returns [`RegistrationError::InvalidInput`] when input lengths differ, a
    /// selected sample is non-finite, a region label is out of range, or the
    /// mask selects no samples.
    pub fn compute_masked_samples(
        &mut self,
        fixed: &[f64],
        moving: &[f64],
        regions: &[u8],
        mask: Option<&[bool]>,
    ) -> Result<f64> {
        validate_inputs(fixed, moving, regions, mask)?;
        self.joint.fill(0.0);
        self.fixed_marginal.fill(0.0);
        self.moving_marginal.fill(0.0);
        self.mass.fill(0.0);

        let bins = self.metric.bins;
        let joint_stride = bins * bins;
        let region_count = usize::from(self.region_count.get());
        let mut total_mass = 0.0_f64;
        for (index, ((&fixed_value, &moving_value), &region_label)) in fixed
            .iter()
            .zip(moving.iter())
            .zip(regions.iter())
            .enumerate()
        {
            if mask.is_some_and(|selection| !selection[index]) {
                continue;
            }
            if !fixed_value.is_finite() || !moving_value.is_finite() {
                return Err(RegistrationError::InvalidInput(format!(
                    "mutual-information sample {index} is not finite: fixed={fixed_value}, moving={moving_value}"
                )));
            }
            let region = usize::from(region_label);
            if region >= region_count {
                return Err(RegistrationError::InvalidInput(format!(
                    "spatial-region label {region} at sample {index} exceeds region count {region_count}"
                )));
            }

            let fixed_coordinate = self.metric.fixed_range.coordinate(fixed_value, bins);
            let moving_coordinate = self.metric.moving_range.coordinate(moving_value, bins);
            let fixed_weights = discrete_bin_weights(fixed_coordinate, bins);
            let moving_weights = match self.metric.estimator {
                HistogramEstimator::Discrete => discrete_bin_weights(moving_coordinate, bins),
                HistogramEstimator::MovingLinearPartialVolume => {
                    linear_bin_weights(moving_coordinate, bins)
                }
            };
            let joint_offset = region * joint_stride;
            let marginal_offset = region * bins;
            for &(fixed_bin, fixed_weight) in &fixed_weights {
                self.fixed_marginal[marginal_offset + fixed_bin] += fixed_weight;
                for &(moving_bin, moving_weight) in &moving_weights {
                    self.joint[joint_offset + fixed_bin * bins + moving_bin] +=
                        fixed_weight * moving_weight;
                }
            }
            for &(moving_bin, moving_weight) in &moving_weights {
                self.moving_marginal[marginal_offset + moving_bin] += moving_weight;
            }
            self.mass[region] += 1.0;
            total_mass += 1.0;
        }

        if total_mass == 0.0 {
            return Err(RegistrationError::InvalidInput(
                "mutual-information mask selected no samples".to_owned(),
            ));
        }

        let mut fixed_entropy = 0.0;
        let mut moving_entropy = 0.0;
        let mut joint_entropy = 0.0;
        for (region, &region_mass) in self.mass.iter().enumerate() {
            if region_mass == 0.0 {
                continue;
            }
            let weight = region_mass / total_mass;
            let marginal_offset = region * bins;
            let joint_offset = region * joint_stride;
            fixed_entropy += weight
                * entropy(
                    &self.fixed_marginal[marginal_offset..marginal_offset + bins],
                    region_mass,
                );
            moving_entropy += weight
                * entropy(
                    &self.moving_marginal[marginal_offset..marginal_offset + bins],
                    region_mass,
                );
            joint_entropy += weight
                * entropy(
                    &self.joint[joint_offset..joint_offset + joint_stride],
                    region_mass,
                );
        }
        Ok(normalize_information(
            self.metric.normalization,
            fixed_entropy,
            moving_entropy,
            joint_entropy,
        ))
    }
}

fn validate_inputs(
    fixed: &[f64],
    moving: &[f64],
    regions: &[u8],
    mask: Option<&[bool]>,
) -> Result<()> {
    if fixed.len() != moving.len() || fixed.len() != regions.len() {
        return Err(RegistrationError::InvalidInput(format!(
            "conditioned mutual-information lengths differ: fixed={}, moving={}, regions={}",
            fixed.len(),
            moving.len(),
            regions.len()
        )));
    }
    if let Some(selection) = mask {
        if selection.len() != fixed.len() {
            return Err(RegistrationError::InvalidInput(format!(
                "mutual-information mask length {} differs from sample length {}",
                selection.len(),
                fixed.len()
            )));
        }
    }
    Ok(())
}

fn zeroed_histogram(length: usize, name: &str) -> Result<Vec<f64>> {
    let mut histogram = Vec::new();
    histogram.try_reserve_exact(length).map_err(|error| {
        RegistrationError::InvalidInput(format!(
            "cannot reserve {length} values for {name} histogram: {error}"
        ))
    })?;
    histogram.resize(length, 0.0);
    Ok(histogram)
}

fn normalize_information(
    normalization: NmiNormalization,
    fixed_entropy: f64,
    moving_entropy: f64,
    joint_entropy: f64,
) -> f64 {
    match normalization {
        NmiNormalization::MeanEntropy => {
            let denominator = fixed_entropy + moving_entropy;
            if denominator == 0.0 {
                0.0
            } else {
                2.0 * (fixed_entropy + moving_entropy - joint_entropy) / denominator
            }
        }
        NmiNormalization::JointEntropy => {
            if joint_entropy == 0.0 {
                0.0
            } else {
                (fixed_entropy + moving_entropy) / joint_entropy
            }
        }
    }
}
