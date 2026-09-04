//! Configurable histogram mutual information for classical registration.

use leto::Array3;

use super::super::error::{RegistrationError, Result};

mod conditioned;

pub use conditioned::SpatiallyConditionedMutualInformationMetric;

/// Finite, non-empty intensity interval used by histogram density estimation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IntensityRange {
    minimum: f64,
    maximum: f64,
}

impl IntensityRange {
    /// Validate an inclusive intensity interval.
    ///
    /// # Errors
    ///
    /// Returns [`RegistrationError::InvalidInput`] when either endpoint is not
    /// finite or `maximum` is not greater than `minimum`.
    pub fn try_new(minimum: f64, maximum: f64) -> Result<Self> {
        if !minimum.is_finite() || !maximum.is_finite() || maximum <= minimum {
            return Err(RegistrationError::InvalidInput(format!(
                "intensity range must be finite and increasing, got [{minimum}, {maximum}]"
            )));
        }
        Ok(Self { minimum, maximum })
    }

    fn coordinate(self, value: f64, bins: usize) -> f64 {
        let span = self.maximum - self.minimum;
        (((value - self.minimum) / span) * (bins - 1) as f64).clamp(0.0, (bins - 1) as f64)
    }
}

/// Normalization applied to mutual information.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum NmiNormalization {
    /// `2·MI/(H(fixed)+H(moving))`, in `[0, 1]` for non-degenerate inputs.
    MeanEntropy,
    /// Studholme NMI: `(H(fixed)+H(moving))/H(fixed,moving)`.
    JointEntropy,
}

/// Histogram density estimator used by mutual information.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum HistogramEstimator {
    /// Assign each intensity to its nearest bin.
    Discrete,
    /// Linearly distribute the moving intensity between adjacent bins.
    ///
    /// The fixed intensity remains discretely binned. This partial-volume
    /// estimator makes the objective piecewise continuous as a moving image is
    /// resampled without blurring the fixed-image marginal.
    MovingLinearPartialVolume,
}

/// Mutual information using a selected histogram density estimator.
///
/// [`HistogramEstimator::Discrete`] retains exact discrete-information
/// identities. [`HistogramEstimator::MovingLinearPartialVolume`] distributes
/// moving-image intensity through a first-order B-spline kernel, making the
/// objective piecewise continuous under moving-image interpolation. Fixed and
/// moving modalities have independent ranges.
///
/// The implementation streams borrowed samples into `O(bins²)` storage; it does
/// not copy or retain image voxels.
#[derive(Debug, Clone, PartialEq)]
pub struct MutualInformationMetric {
    bins: usize,
    fixed_range: IntensityRange,
    moving_range: IntensityRange,
    normalization: NmiNormalization,
    estimator: HistogramEstimator,
}

impl MutualInformationMetric {
    /// Create a metric using one intensity interval for both images.
    ///
    /// # Errors
    ///
    /// Returns [`RegistrationError::InvalidInput`] for fewer than two bins or
    /// an invalid intensity interval.
    pub fn new(bins: usize, minimum: f64, maximum: f64) -> Result<Self> {
        let range = IntensityRange::try_new(minimum, maximum)?;
        Self::with_ranges(
            bins,
            range,
            range,
            NmiNormalization::MeanEntropy,
            HistogramEstimator::Discrete,
        )
    }

    /// Create a metric with modality-specific intensity intervals.
    ///
    /// # Errors
    ///
    /// Returns [`RegistrationError::InvalidInput`] for fewer than two bins.
    pub fn with_ranges(
        bins: usize,
        fixed_range: IntensityRange,
        moving_range: IntensityRange,
        normalization: NmiNormalization,
        estimator: HistogramEstimator,
    ) -> Result<Self> {
        if bins < 2 {
            return Err(RegistrationError::InvalidInput(format!(
                "mutual-information histogram requires at least two bins, got {bins}"
            )));
        }
        Ok(Self {
            bins,
            fixed_range,
            moving_range,
            normalization,
            estimator,
        })
    }

    /// Return the number of bins on each histogram axis.
    #[must_use]
    pub fn bins(&self) -> usize {
        self.bins
    }

    /// Evaluate equally shaped volumes.
    ///
    /// # Errors
    ///
    /// Returns [`RegistrationError::InvalidInput`] for unequal shapes, empty
    /// volumes, or selected non-finite samples.
    pub fn compute(&self, fixed: &Array3<f64>, moving: &Array3<f64>) -> Result<f64> {
        if fixed.shape() != moving.shape() {
            return Err(RegistrationError::InvalidInput(format!(
                "mutual-information volume shapes differ: {:?} versus {:?}",
                fixed.shape(),
                moving.shape()
            )));
        }
        self.compute_masked_samples(
            fixed.as_slice().ok_or_else(|| {
                RegistrationError::InvalidInput(
                    "fixed mutual-information volume is not contiguous".to_owned(),
                )
            })?,
            moving.as_slice().ok_or_else(|| {
                RegistrationError::InvalidInput(
                    "moving mutual-information volume is not contiguous".to_owned(),
                )
            })?,
            None,
        )
    }

    /// Evaluate paired samples, optionally selecting them with a mask.
    ///
    /// # Errors
    ///
    /// Returns [`RegistrationError::InvalidInput`] when sample or mask lengths
    /// differ, no sample is selected, or a selected sample is not finite.
    pub fn compute_masked_samples(
        &self,
        fixed: &[f64],
        moving: &[f64],
        mask: Option<&[bool]>,
    ) -> Result<f64> {
        if fixed.len() != moving.len() {
            return Err(RegistrationError::InvalidInput(format!(
                "mutual-information sample lengths differ: {} versus {}",
                fixed.len(),
                moving.len()
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

        let mut joint = vec![0.0_f64; self.bins * self.bins];
        let mut fixed_marginal = vec![0.0_f64; self.bins];
        let mut moving_marginal = vec![0.0_f64; self.bins];
        let mut mass = 0.0_f64;

        for (index, (&fixed_value, &moving_value)) in fixed.iter().zip(moving.iter()).enumerate() {
            if mask.is_some_and(|selection| !selection[index]) {
                continue;
            }
            if !fixed_value.is_finite() || !moving_value.is_finite() {
                return Err(RegistrationError::InvalidInput(format!(
                    "mutual-information sample {index} is not finite: fixed={fixed_value}, moving={moving_value}"
                )));
            }

            let fixed_coordinate = self.fixed_range.coordinate(fixed_value, self.bins);
            let moving_coordinate = self.moving_range.coordinate(moving_value, self.bins);
            let fixed_weights = discrete_bin_weights(fixed_coordinate, self.bins);
            let moving_weights = match self.estimator {
                HistogramEstimator::Discrete => discrete_bin_weights(moving_coordinate, self.bins),
                HistogramEstimator::MovingLinearPartialVolume => {
                    linear_bin_weights(moving_coordinate, self.bins)
                }
            };
            for &(fixed_bin, fixed_weight) in &fixed_weights {
                fixed_marginal[fixed_bin] += fixed_weight;
                for &(moving_bin, moving_weight) in &moving_weights {
                    joint[fixed_bin * self.bins + moving_bin] += fixed_weight * moving_weight;
                }
            }
            for &(moving_bin, moving_weight) in &moving_weights {
                moving_marginal[moving_bin] += moving_weight;
            }
            mass += 1.0;
        }

        if mass == 0.0 {
            return Err(RegistrationError::InvalidInput(
                "mutual-information mask selected no samples".to_owned(),
            ));
        }

        let fixed_entropy = entropy(&fixed_marginal, mass);
        let moving_entropy = entropy(&moving_marginal, mass);
        let joint_entropy = entropy(&joint, mass);
        Ok(match self.normalization {
            NmiNormalization::MeanEntropy => {
                let denominator = fixed_entropy + moving_entropy;
                if denominator == 0.0 {
                    let fixed_peak = peak_bin(&fixed_marginal);
                    let moving_peak = peak_bin(&moving_marginal);
                    if fixed_peak == moving_peak {
                        1.0
                    } else {
                        0.0
                    }
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
        })
    }
}

impl Default for MutualInformationMetric {
    fn default() -> Self {
        Self {
            bins: 32,
            fixed_range: IntensityRange {
                minimum: 0.0,
                maximum: 255.0,
            },
            moving_range: IntensityRange {
                minimum: 0.0,
                maximum: 255.0,
            },
            normalization: NmiNormalization::MeanEntropy,
            estimator: HistogramEstimator::Discrete,
        }
    }
}

fn discrete_bin_weights(coordinate: f64, bins: usize) -> [(usize, f64); 2] {
    let bin = (coordinate.round() as usize).min(bins - 1);
    [(bin, 1.0), (bin, 0.0)]
}

fn linear_bin_weights(coordinate: f64, bins: usize) -> [(usize, f64); 2] {
    let lower = coordinate.floor() as usize;
    let upper = (lower + 1).min(bins - 1);
    let upper_weight = coordinate - lower as f64;
    [(lower, 1.0 - upper_weight), (upper, upper_weight)]
}

fn entropy(histogram: &[f64], mass: f64) -> f64 {
    histogram
        .iter()
        .copied()
        .filter(|&count| count > 0.0)
        .map(|count| {
            let probability = count / mass;
            -probability * probability.ln()
        })
        .sum()
}

fn peak_bin(histogram: &[f64]) -> usize {
    histogram
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map_or(0, |(index, _)| index)
}

#[cfg(test)]
#[path = "metric_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "conditioned_metric_tests.rs"]
mod conditioned_tests;
