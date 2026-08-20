//! Displacement calculators that regularize the raw matcher output.
//!
//! The raw matcher reports a displacement for every block independently, so
//! decorrelated blocks produce spurious outliers that read as strain. These
//! calculators impose a spatial prior — either a fixed Gaussian prior weighted
//! by correlation confidence, or a local least-squares slope prior — and return
//! a regularized field with the same centres and peak metadata.
//!
//! Both **condition** every block. The complementary operation, rejecting the
//! blocks whose measurement cannot be believed at all, lives in
//! [`crate::strain_window_filter`].

use anyhow::{bail, Result};

use crate::{search::MultiResolutionDisplacement, DisplacementField};

/// Reject a field whose parallel arrays disagree in length.
///
/// Every consumer indexes all three by the same position, so unequal lengths
/// are not a recoverable shape difference — they mean the field does not
/// describe a consistent set of blocks.
fn validate_field(field: &DisplacementField) -> Result<()> {
    let (centres, displacements, peaks) = (
        field.centres.len(),
        field.displacements.len(),
        field.peak_similarities.len(),
    );
    if centres != displacements || centres != peaks {
        bail!(
            "displacement field arrays disagree: {centres} centres,              {displacements} displacements, {peaks} peak similarities"
        );
    }
    Ok(())
}

/// A Gaussian prior on displacement, weighted by correlation confidence.
///
/// Each block's displacement is treated as a noisy observation of a global
/// prior. The observation precision scales with the square of the peak
/// similarity, so a high-confidence match dominates the prior and a
/// decorrelated block is pulled all the way to the prior mean instead of being
/// silently reported as a valid zero displacement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BayesianDisplacementPrior {
    /// Prior mean displacement per component, in voxels.
    pub mean: [f64; 3],
    /// Prior variance, strictly positive and finite (squared voxels).
    pub prior_variance: f64,
    /// Baseline observation variance, strictly positive and finite.
    pub observation_variance: f64,
    /// Minimum peak similarity that contributes observation precision.
    /// Values are clamped to `[0, 1]` by validation.
    pub minimum_peak_similarity: f64,
}

impl BayesianDisplacementPrior {
    /// Construct a Gaussian-prior regularizer.
    ///
    /// # Errors
    ///
    /// Returns an error when the mean is non-finite, either variance is
    /// non-positive or non-finite, or the peak threshold is outside `[0, 1]`.
    pub fn new(
        mean: [f64; 3],
        prior_variance: f64,
        observation_variance: f64,
        minimum_peak_similarity: f64,
    ) -> Result<Self> {
        if mean.iter().any(|value| !value.is_finite()) {
            bail!("Bayesian displacement prior mean must be finite");
        }
        if !prior_variance.is_finite() || prior_variance <= 0.0 {
            bail!("prior variance must be finite and positive");
        }
        if !observation_variance.is_finite() || observation_variance <= 0.0 {
            bail!("observation variance must be finite and positive");
        }
        if !minimum_peak_similarity.is_finite() || !(0.0..=1.0).contains(&minimum_peak_similarity) {
            bail!("minimum peak similarity must be finite and in [0, 1]");
        }
        Ok(Self {
            mean,
            prior_variance,
            observation_variance,
            minimum_peak_similarity,
        })
    }

    /// Re-check the stored values.
    ///
    /// [`Self::new`] already rejects invalid input, but the fields are public,
    /// so a prior can also be assembled by struct literal or deserialized.
    /// Pipeline entry points call this before doing any metric work, so a
    /// malformed stage fails before the expensive part rather than silently
    /// producing a field nobody can interpret.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::new`].
    pub fn validate(&self) -> Result<()> {
        Self::new(
            self.mean,
            self.prior_variance,
            self.observation_variance,
            self.minimum_peak_similarity,
        )
        .map(|_| ())
    }

    /// Validate both the prior and the field, then regularize.
    ///
    /// [`Self::regularize`] assumes a well-formed field because the matcher
    /// produces one. A field can also be assembled by hand or arrive from
    /// deserialization, where mismatched array lengths would otherwise index
    /// out of bounds or silently drop blocks.
    ///
    /// # Errors
    ///
    /// Returns an error when this prior is invalid, or when the field's three
    /// arrays do not have equal length.
    pub fn try_regularize(&self, field: &DisplacementField) -> Result<DisplacementField> {
        self.validate()?;
        validate_field(field)?;
        Ok(self.regularize(field))
    }

    /// Regularize a displacement field, preserving centres and peak metadata.
    ///
    /// For a finite peak `c >= minimum_peak_similarity`, observation precision
    /// is `c² / observation_variance`; otherwise it is zero and the posterior
    /// mean is the prior mean exactly. The posterior mean per component is
    /// `(μ₀/σ₀² + c²·d/σ_d²) / (1/σ₀² + c²/σ_d²)`.
    #[must_use]
    pub fn regularize(&self, field: &DisplacementField) -> DisplacementField {
        let displacements = field
            .displacements
            .iter()
            .zip(&field.peak_similarities)
            .map(|(&displacement, &peak)| self.posterior_displacement(displacement, peak))
            .collect();

        DisplacementField {
            centres: field.centres.clone(),
            displacements,
            peak_similarities: field.peak_similarities.clone(),
        }
    }

    /// Regularize the final result of a pyramid match.
    ///
    /// The finest-level displacement is treated as one observation, with its
    /// finest-level peak similarity supplying confidence. Coarse-to-fine level
    /// diagnostics and the reported peak are preserved verbatim, so callers can
    /// compare the raw and regularized final estimate without losing execution
    /// evidence.
    #[must_use]
    pub fn regularize_pyramid(
        &self,
        result: &MultiResolutionDisplacement,
    ) -> MultiResolutionDisplacement {
        let mut regularized = result.clone();
        regularized.displacement =
            self.posterior_displacement(result.displacement, result.peak_similarity);
        regularized
    }

    fn posterior_displacement(&self, displacement: [f64; 3], peak: f64) -> [f64; 3] {
        let prior_precision = 1.0 / self.prior_variance;
        let confidence = if peak.is_finite() && peak >= self.minimum_peak_similarity {
            peak.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let observation_precision = confidence * confidence / self.observation_variance;
        let posterior_precision = prior_precision + observation_precision;
        std::array::from_fn(|axis| {
            (self.mean[axis] * prior_precision + displacement[axis] * observation_precision)
                / posterior_precision
        })
    }
}

/// A displacement prior that smooths toward a local least-squares slope
/// (the Kallel–Ophir estimator).
///
/// For each block, the neighbouring axial displacements within `window`
/// blocks are fit to a straight line by ordinary least squares; the fitted
/// value at the block's position becomes the prior. The block's own measured
/// displacement is then blended with that prior by `regularization_strength`
/// in `[0, 1]` (1 = trust the measurement entirely, 0 = trust the line). This
/// is a *displacement* calculator with a strain prior — distinct from the
/// crate's `strain_from_displacement`, which estimates strain from a finished
/// displacement field.
///
/// Distinct, too, from [`crate::strain_window_filter`], despite both taking a
/// window over axial strain. This one **conditions** every block, blending it
/// toward the fitted line, so a real feature is attenuated along with the
/// noise. That one **rejects**: a block outside a plausibility bound is
/// discarded and replaced from its neighbours, and everything else is returned
/// untouched. Use this to suppress measurement noise, that to remove
/// peak-hopping artefacts; they compose in either order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LeastSquaresDisplacementPrior {
    /// Odd window size in blocks (the number of axial neighbours used for the
    /// local fit). Must be `>= 3`.
    pub window: usize,
    /// Blend between the local line fit and the measured displacement, in
    /// `[0, 1]`; higher keeps more of the raw measurement.
    pub regularization_strength: f64,
}

impl LeastSquaresDisplacementPrior {
    /// Construct a strain-window regularizer.
    ///
    /// # Errors
    ///
    /// Returns an error when the window is smaller than 3 or even, or the
    /// strength is outside `[0, 1]`.
    pub fn new(window: usize, regularization_strength: f64) -> Result<Self> {
        if window < 3 || window.is_multiple_of(2) {
            bail!("strain window must be odd and at least 3, got {window}");
        }
        if !regularization_strength.is_finite() || !(0.0..=1.0).contains(&regularization_strength) {
            bail!("regularization strength must be in [0, 1], got {regularization_strength}");
        }
        Ok(Self {
            window,
            regularization_strength,
        })
    }

    /// Re-check the stored values.
    ///
    /// [`Self::new`] already rejects invalid input, but the fields are public,
    /// so a prior can also be assembled by struct literal or deserialized.
    /// Pipeline entry points call this before doing any metric work, so a
    /// malformed stage fails before the expensive part rather than silently
    /// producing a field nobody can interpret.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::new`].
    pub fn validate(&self) -> Result<()> {
        Self::new(self.window, self.regularization_strength).map(|_| ())
    }

    /// Validate both the prior and the field, then regularize.
    ///
    /// [`Self::regularize`] assumes a well-formed field because the matcher
    /// produces one. A field can also be assembled by hand or arrive from
    /// deserialization, where mismatched array lengths would otherwise index
    /// out of bounds or silently drop blocks.
    ///
    /// # Errors
    ///
    /// Returns an error when this prior is invalid, or when the field's three
    /// arrays do not have equal length.
    pub fn try_regularize(&self, field: &DisplacementField) -> Result<DisplacementField> {
        self.validate()?;
        validate_field(field)?;
        Ok(self.regularize(field))
    }

    /// Regularize a displacement field axially, preserving centres and peaks.
    ///
    /// The field's blocks are grouped by lateral `(y, x)` position and ordered
    /// by axial `z`; the fit is computed along each axial line independently.
    #[must_use]
    pub fn regularize(&self, field: &DisplacementField) -> DisplacementField {
        let mut displacements = field.displacements.clone();
        let n = field.len();
        if n == 0 {
            return field.clone();
        }

        // Group centres by (y, x), ordered by z, mirroring
        // `strain_from_displacement` so the two agree on which blocks are axial
        // neighbours.
        let mut indexed: Vec<(usize, usize, usize, usize)> = field
            .centres
            .iter()
            .enumerate()
            .map(|(i, &[z, y, x])| (y, x, z, i))
            .collect();
        indexed.sort_unstable();

        let half = self.window / 2;
        let mut start = 0;
        while start < indexed.len() {
            let (y0, x0, _, _) = indexed[start];
            let mut end = start + 1;
            while end < indexed.len() && indexed[end].0 == y0 && indexed[end].1 == x0 {
                end += 1;
            }
            let line = &indexed[start..end];
            let m = line.len();
            for pos in 0..m {
                let lo = pos.saturating_sub(half);
                let hi = (pos + half).min(m - 1);
                if hi <= lo {
                    continue;
                }
                // Least-squares fit of axial displacement vs axial position
                // over [lo, hi]. Positions are the axial centre coordinates.
                let (mut sum_x, mut sum_y, mut sum_xx, mut sum_xy) =
                    (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
                let count = (hi - lo + 1) as f64;
                for &(_, _, _, i) in &line[lo..=hi] {
                    let x = field.centres[i][0] as f64;
                    let y = field.displacements[i][0];
                    sum_x += x;
                    sum_y += y;
                    sum_xx += x * x;
                    sum_xy += x * y;
                }
                let denominator = count * sum_xx - sum_x * sum_x;
                let (slope, intercept) = if denominator.abs() > f64::EPSILON {
                    (
                        (count * sum_xy - sum_x * sum_y) / denominator,
                        (sum_y - ((count * sum_xy - sum_x * sum_y) / denominator) * sum_x) / count,
                    )
                } else {
                    (0.0, sum_y / count)
                };

                let i = line[pos].3;
                let x = field.centres[i][0] as f64;
                let line_prior = slope * x + intercept;
                let strength = self.regularization_strength;
                let blended = strength * displacements[i][0] + (1.0 - strength) * line_prior;
                displacements[i][0] = blended;
            }
            start = end;
        }

        DisplacementField {
            centres: field.centres.clone(),
            displacements,
            peak_similarities: field.peak_similarities.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn field(displacements: Vec<[f64; 3]>, peaks: Vec<f64>) -> DisplacementField {
        let centres = (0..displacements.len())
            .map(|index| [index * 3, 0, 0]) // axial stride 3
            .collect();
        DisplacementField {
            centres,
            displacements,
            peak_similarities: peaks,
        }
    }

    #[test]
    fn high_confidence_observation_dominates_prior() {
        let prior =
            BayesianDisplacementPrior::new([0.0, 0.0, 0.0], 100.0, 1.0, 0.2).expect("valid prior");
        let result = prior.regularize(&field(vec![[4.0, -2.0, 1.0]], vec![1.0]));
        assert!((result.displacements[0][0] - 3.9604).abs() < 1.0e-3);
        assert!((result.displacements[0][1] + 1.9802).abs() < 1.0e-3);
        assert_eq!(result.centres, vec![[0, 0, 0]]);
        assert_eq!(result.peak_similarities, vec![1.0]);
    }

    #[test]
    fn low_confidence_and_nonfinite_observations_use_the_prior() {
        let prior =
            BayesianDisplacementPrior::new([2.0, -1.0, 0.5], 1.0, 1.0, 0.8).expect("valid prior");
        let result = prior.regularize(&field(
            vec![[100.0, 100.0, 100.0], [3.0, 4.0, 5.0]],
            vec![0.79, f64::NAN],
        ));
        assert_eq!(result.displacements[0], [2.0, -1.0, 0.5]);
        assert_eq!(result.displacements[1], [2.0, -1.0, 0.5]);
    }

    #[test]
    fn intermediate_confidence_is_between_prior_and_observation() {
        let prior =
            BayesianDisplacementPrior::new([0.0, 0.0, 0.0], 1.0, 1.0, 0.0).expect("valid prior");
        let result = prior.regularize(&field(vec![[10.0, 0.0, 0.0]], vec![0.5]));
        assert!((result.displacements[0][0] - 2.0).abs() < 1.0e-12);
    }

    #[test]
    fn rejects_invalid_variances_means_and_thresholds() {
        assert!(BayesianDisplacementPrior::new([0.0; 3], 0.0, 1.0, 0.5).is_err());
        assert!(BayesianDisplacementPrior::new([0.0; 3], 1.0, -1.0, 0.5).is_err());
        assert!(BayesianDisplacementPrior::new([f64::NAN; 3], 1.0, 1.0, 0.5).is_err());
        assert!(BayesianDisplacementPrior::new([0.0; 3], 1.0, 1.0, 1.1).is_err());
    }

    #[test]
    fn least_squares_prior_preserves_a_linear_field_exactly() {
        // A perfectly linear axial displacement field (constant strain 0.01)
        // must be unchanged by a strain-window regularizer, because the local
        // line fit reproduces the field exactly.
        let displacements: Vec<[f64; 3]> =
            (0..10).map(|i| [0.01 * (i * 3) as f64, 0.0, 0.0]).collect();
        let peaks = vec![0.9; 10];
        let regularizer = LeastSquaresDisplacementPrior::new(5, 0.5).expect("valid strain window");
        let result = regularizer.regularize(&field(displacements.clone(), peaks));
        for (before, after) in displacements.iter().zip(&result.displacements) {
            assert!((before[0] - after[0]).abs() < 1.0e-12);
        }
        assert_eq!(result.centres.len(), 10);
    }

    #[test]
    fn least_squares_prior_smooths_a_single_outlier() {
        // One outlier among an otherwise linear field is pulled toward the
        // local line. Least-squares over the window is not robust, so the
        // outlier is not fully suppressed — but it must move substantially
        // toward the surrounding trend, not stay at its raw value.
        let mut displacements: Vec<[f64; 3]> =
            (0..10).map(|i| [0.01 * (i * 3) as f64, 0.0, 0.0]).collect();
        displacements[5] = [5.0, 0.0, 0.0]; // a gross outlier
        let peaks = vec![0.9; 10];
        let regularizer = LeastSquaresDisplacementPrior::new(5, 0.5).expect("valid strain window");
        let result = regularizer.regularize(&field(displacements, peaks));
        // Raw outlier is 5.0; the surrounding trend at that position is ~0.15.
        // The smoothed value must be a clear fraction of the way toward the
        // trend (well below half the outlier magnitude).
        let smoothed = result.displacements[5][0];
        assert!(
            smoothed < 3.5 && smoothed > 0.0,
            "outlier {smoothed} should be pulled toward the local line, not stay at 5.0"
        );
        // The neighbours are only lightly perturbed by the single outlier.
        assert!((result.displacements[7][0] - 0.21).abs() < 0.5);
    }

    #[test]
    fn least_squares_prior_rejects_invalid_configuration() {
        assert!(LeastSquaresDisplacementPrior::new(2, 0.5).is_err());
        assert!(LeastSquaresDisplacementPrior::new(4, 0.5).is_err());
        assert!(LeastSquaresDisplacementPrior::new(5, 1.5).is_err());
        assert!(LeastSquaresDisplacementPrior::new(5, -0.1).is_err());
    }
}
