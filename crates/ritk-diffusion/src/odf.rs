//! Regularized analytical Q-ball orientation distribution functions.
//!
//! RITK fits the normalized diffusion signal in Apollo's orthonormal real,
//! even-degree spherical-harmonic basis. Leto solves the least-squares system
//! augmented with a Laplace-Beltrami penalty. The fitted signal coefficients
//! `c_lm` become Q-ball ODF coefficients through the Funk-Hecke relation
//! `psi_lm = 2*pi*P_l(0)*c_lm`.
//!
//! This is the analytical Q-ball model described by Descoteaux et al. (2007),
//! not constrained spherical deconvolution or a fiber orientation density.
//! <https://doi.org/10.1002/mrm.21277>

use apollo_sht::{RealShError, RealSphericalHarmonicBasis, real_spherical_harmonic};
use leto::{Array1, Array2};
use ritk_diffusion_scheme::{DiffusionWeighting, GradientScheme};

/// Failure while configuring, estimating, or evaluating a Q-ball ODF.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum OdfError {
    /// The signal count and acquisition count differ.
    #[error("signal count {signal_count} does not match acquisition count {acquisition_count}")]
    SignalLengthMismatch {
        /// Number of supplied signals.
        signal_count: usize,
        /// Number of scheme entries.
        acquisition_count: usize,
    },
    /// A signal sample is NaN or infinite.
    #[error("signal at acquisition index {index} is not finite: {value}")]
    NonFiniteSignal {
        /// Acquisition-order index.
        index: usize,
        /// Invalid signal value.
        value: f64,
    },
    /// No samples fall at or below the configured b0 threshold.
    #[error("gradient scheme has no unweighted reference volumes")]
    NoB0Volumes,
    /// No samples exceed the configured b0 threshold.
    #[error("gradient scheme has no diffusion-weighted volumes")]
    NoDwiDirections,
    /// The number of weighted directions cannot identify all coefficients.
    #[error(
        "{direction_count} diffusion-weighted directions cannot identify {coefficient_count} spherical-harmonic coefficients"
    )]
    Underdetermined {
        /// Weighted measurement count.
        direction_count: usize,
        /// Requested coefficient count.
        coefficient_count: usize,
    },
    /// Mean b0 signal is not finite and strictly positive.
    #[error("baseline signal S0 must be finite and positive, got {value}")]
    InvalidBaseline {
        /// Computed baseline value.
        value: f64,
    },
    /// Laplace-Beltrami regularization is negative or non-finite.
    #[error("regularization must be finite and nonnegative, got {value}")]
    InvalidRegularization {
        /// Invalid regularization value.
        value: f64,
    },
    /// Spherical grid dimensions are empty or overflow their element count.
    #[error("invalid spherical grid {theta_samples}x{phi_samples}: {reason}")]
    InvalidGrid {
        /// Polar sample count.
        theta_samples: usize,
        /// Azimuthal sample count.
        phi_samples: usize,
        /// Violated size invariant.
        reason: &'static str,
    },
    /// Evaluation angles or direction violate the finite domain.
    #[error("invalid ODF evaluation direction: {0}")]
    InvalidEvaluation(String),
    /// Apollo rejected the even-degree basis configuration.
    #[error("spherical-harmonic basis error: {0}")]
    Basis(#[from] RealShError),
    /// Leto could not construct or solve the least-squares system.
    #[error("least-squares solve failed: {0}")]
    SolveFailed(String),
}

/// Validated analytical Q-ball configuration.
#[derive(Debug, Clone, Copy)]
pub struct OdfConfig {
    l_max: usize,
    regularization: f64,
    b0_threshold: DiffusionWeighting,
}

impl OdfConfig {
    /// Construct a Q-ball configuration.
    ///
    /// `l_max` must be even and at least two. `regularization` is the
    /// nonnegative Laplace-Beltrami normal-equation weight. `b0_threshold`
    /// classifies reference and weighted volumes.
    ///
    /// # Errors
    ///
    /// Returns a typed error for an invalid degree or regularization value.
    pub fn new(
        l_max: usize,
        regularization: f64,
        b0_threshold: DiffusionWeighting,
    ) -> Result<Self, OdfError> {
        RealSphericalHarmonicBasis::new(l_max)?;
        if !regularization.is_finite() || regularization < 0.0 {
            return Err(OdfError::InvalidRegularization {
                value: regularization,
            });
        }
        Ok(Self {
            l_max,
            regularization,
            b0_threshold,
        })
    }

    /// Maximum even spherical-harmonic degree.
    #[must_use]
    pub const fn l_max(self) -> usize {
        self.l_max
    }

    /// Laplace-Beltrami regularization weight.
    #[must_use]
    pub const fn regularization(self) -> f64 {
        self.regularization
    }

    /// Threshold separating b0 and weighted acquisitions.
    #[must_use]
    pub const fn b0_threshold(self) -> DiffusionWeighting {
        self.b0_threshold
    }
}

impl Default for OdfConfig {
    fn default() -> Self {
        Self {
            l_max: 4,
            regularization: 0.006,
            b0_threshold: DiffusionWeighting::from_seconds_per_square_millimeter(50.0)
                .expect("invariant: default b0 threshold is finite and nonnegative"),
        }
    }
}

/// Contiguous ODF samples on an equiangular spherical grid.
#[derive(Debug, Clone, PartialEq)]
pub struct SphericalOdfGrid {
    shape: [usize; 2],
    values: Box<[f64]>,
}

impl SphericalOdfGrid {
    /// Grid shape `[theta_samples, phi_samples]`.
    #[must_use]
    pub const fn shape(&self) -> [usize; 2] {
        self.shape
    }

    /// Row-major ODF values with phi varying fastest.
    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }
}

/// Estimated Q-ball orientation distribution at one voxel.
#[derive(Debug, Clone)]
pub struct OdField {
    coefficients: Box<[f64]>,
    basis: RealSphericalHarmonicBasis,
    baseline_signal: f64,
    normalized_signal_residual: f64,
}

impl OdField {
    /// Q-ball ODF coefficients in Apollo's degree-major even-order basis.
    #[must_use]
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    /// Maximum spherical-harmonic degree.
    #[must_use]
    pub fn l_max(&self) -> usize {
        self.basis.l_max()
    }

    /// Mean signal over b0 acquisitions.
    #[must_use]
    pub const fn baseline_signal(&self) -> f64 {
        self.baseline_signal
    }

    /// L2 residual of the normalized signal fit before the Funk-Radon transform.
    #[must_use]
    pub const fn normalized_signal_residual(&self) -> f64 {
        self.normalized_signal_residual
    }

    /// Evaluate at polar angle `theta` and azimuth `phi`, in radians.
    ///
    /// # Errors
    ///
    /// Returns [`OdfError::InvalidEvaluation`] unless `theta` is finite in
    /// `[0, pi]` and `phi` is finite.
    pub fn evaluate(&self, theta: f64, phi: f64) -> Result<f64, OdfError> {
        if !theta.is_finite() || !(0.0..=std::f64::consts::PI).contains(&theta) {
            return Err(OdfError::InvalidEvaluation(format!(
                "theta must be in [0, pi], got {theta}"
            )));
        }
        if !phi.is_finite() {
            return Err(OdfError::InvalidEvaluation(format!(
                "phi must be finite, got {phi}"
            )));
        }
        Ok(self.evaluate_unchecked(theta, phi))
    }

    /// Evaluate at a finite unit Cartesian direction.
    ///
    /// # Errors
    ///
    /// Returns [`OdfError::InvalidEvaluation`] when a component is non-finite
    /// or the norm differs from one by more than `1e-6`.
    pub fn evaluate_at_direction(&self, direction: [f64; 3]) -> Result<f64, OdfError> {
        if direction.iter().any(|value| !value.is_finite()) {
            return Err(OdfError::InvalidEvaluation(format!(
                "direction is not finite: {direction:?}"
            )));
        }
        let norm = direction
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        if (norm - 1.0).abs() > 1.0e-6 {
            return Err(OdfError::InvalidEvaluation(format!(
                "direction must be unit length, norm is {norm}"
            )));
        }
        let theta = direction[2].clamp(-1.0, 1.0).acos();
        let phi = direction[1].atan2(direction[0]);
        Ok(self.evaluate_unchecked(theta, phi))
    }

    /// Evaluate a contiguous equiangular spherical grid.
    ///
    /// Polar samples lie at cell centers; azimuthal samples start at zero.
    ///
    /// # Errors
    ///
    /// Returns [`OdfError::InvalidGrid`] for an empty dimension, element-count
    /// overflow, or allocation failure.
    pub fn evaluate_on_grid(
        &self,
        theta_samples: usize,
        phi_samples: usize,
    ) -> Result<SphericalOdfGrid, OdfError> {
        if theta_samples == 0 || phi_samples == 0 {
            return Err(OdfError::InvalidGrid {
                theta_samples,
                phi_samples,
                reason: "both dimensions must be nonzero",
            });
        }
        let count = theta_samples
            .checked_mul(phi_samples)
            .ok_or(OdfError::InvalidGrid {
                theta_samples,
                phi_samples,
                reason: "element count overflows usize",
            })?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|_| OdfError::InvalidGrid {
                theta_samples,
                phi_samples,
                reason: "allocation failed",
            })?;
        for theta_index in 0..theta_samples {
            let theta = std::f64::consts::PI * (theta_index as f64 + 0.5) / theta_samples as f64;
            for phi_index in 0..phi_samples {
                let phi = std::f64::consts::TAU * phi_index as f64 / phi_samples as f64;
                values.push(self.evaluate_unchecked(theta, phi));
            }
        }
        Ok(SphericalOdfGrid {
            shape: [theta_samples, phi_samples],
            values: values.into_boxed_slice(),
        })
    }

    fn evaluate_unchecked(&self, theta: f64, phi: f64) -> f64 {
        self.basis
            .iter_lm()
            .zip(self.coefficients.iter())
            .map(|((_, degree, order), coefficient)| {
                coefficient * real_spherical_harmonic(degree, order, theta, phi)
            })
            .sum()
    }
}

/// Estimate a regularized analytical Q-ball ODF from one voxel's signals.
///
/// # Errors
///
/// Returns a typed error for count mismatch, non-finite signals, missing b0
/// or weighted samples, an underdetermined basis, invalid baseline, or a
/// failed least-squares solve.
pub fn estimate_odf(
    scheme: &GradientScheme,
    signals: &[f64],
    config: OdfConfig,
) -> Result<OdField, OdfError> {
    if signals.len() != scheme.len() {
        return Err(OdfError::SignalLengthMismatch {
            signal_count: signals.len(),
            acquisition_count: scheme.len(),
        });
    }
    if let Some((index, value)) = signals
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(OdfError::NonFiniteSignal { index, value });
    }

    let b0_indices = scheme.b0_indices(config.b0_threshold());
    let dwi_indices = scheme.dwi_indices(config.b0_threshold());
    if b0_indices.is_empty() {
        return Err(OdfError::NoB0Volumes);
    }
    if dwi_indices.is_empty() {
        return Err(OdfError::NoDwiDirections);
    }

    let baseline_signal =
        b0_indices.iter().map(|index| signals[*index]).sum::<f64>() / b0_indices.len() as f64;
    if !baseline_signal.is_finite() || baseline_signal <= 0.0 {
        return Err(OdfError::InvalidBaseline {
            value: baseline_signal,
        });
    }

    let basis = RealSphericalHarmonicBasis::new(config.l_max())?;
    let coefficient_count = basis.num_coefficients();
    if dwi_indices.len() < coefficient_count {
        return Err(OdfError::Underdetermined {
            direction_count: dwi_indices.len(),
            coefficient_count,
        });
    }

    let directions = dwi_indices
        .iter()
        .map(|index| scheme.directions()[*index].direction().to_array())
        .collect::<Vec<_>>();
    let normalized = dwi_indices
        .iter()
        .map(|index| signals[*index] / baseline_signal)
        .collect::<Vec<_>>();
    let design = basis.design_matrix(&directions);
    let signal_coefficients =
        solve_regularized(&design, &normalized, &basis, config.regularization())?;
    let residual = residual_norm(&design, &signal_coefficients, &normalized);

    let coefficients = basis
        .iter_lm()
        .zip(signal_coefficients)
        .map(|((_, degree, _), coefficient)| {
            std::f64::consts::TAU * legendre_at_zero(degree) * coefficient
        })
        .collect::<Vec<_>>()
        .into_boxed_slice();

    Ok(OdField {
        coefficients,
        basis,
        baseline_signal,
        normalized_signal_residual: residual,
    })
}

fn solve_regularized(
    design: &Array2<f64>,
    normalized: &[f64],
    basis: &RealSphericalHarmonicBasis,
    regularization: f64,
) -> Result<Vec<f64>, OdfError> {
    let row_count = design.shape()[0];
    let coefficient_count = basis.num_coefficients();
    if regularization == 0.0 {
        let rhs = Array1::from_vec(row_count, normalized.to_vec())
            .map_err(|error| OdfError::SolveFailed(error.to_string()))?;
        return leto_ops::solve_least_squares(&design.view(), &rhs.view())
            .map(|values| values.iter().copied().collect())
            .map_err(|error| OdfError::SolveFailed(error.to_string()));
    }

    let augmented_rows = row_count
        .checked_add(coefficient_count)
        .ok_or_else(|| OdfError::SolveFailed("augmented row count overflows usize".to_owned()))?;
    let mut augmented = Array2::zeros([augmented_rows, coefficient_count]);
    let mut augmented_rhs = Array1::zeros([augmented_rows]);
    for row in 0..row_count {
        augmented_rhs[row] = normalized[row];
        for column in 0..coefficient_count {
            augmented[[row, column]] = design[[row, column]];
        }
    }
    let root_weight = regularization.sqrt();
    for (index, degree, _) in basis.iter_lm() {
        let laplace_eigenvalue = (degree * (degree + 1)) as f64;
        augmented[[row_count + index, index]] = root_weight * laplace_eigenvalue;
    }
    leto_ops::solve_least_squares(&augmented.view(), &augmented_rhs.view())
        .map(|values| values.iter().copied().collect())
        .map_err(|error| OdfError::SolveFailed(error.to_string()))
}

fn residual_norm(design: &Array2<f64>, coefficients: &[f64], normalized: &[f64]) -> f64 {
    let row_count = design.shape()[0];
    let column_count = design.shape()[1];
    let mut squared_error = 0.0;
    for row in 0..row_count {
        let predicted = (0..column_count)
            .map(|column| design[[row, column]] * coefficients[column])
            .sum::<f64>();
        squared_error += (predicted - normalized[row]).powi(2);
    }
    squared_error.sqrt()
}

fn legendre_at_zero(degree: usize) -> f64 {
    debug_assert_eq!(degree % 2, 0);
    let mut value = 1.0;
    for even_degree in (2..=degree).step_by(2) {
        value *= -(even_degree as f64 - 1.0) / even_degree as f64;
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;
    use ritk_diffusion_scheme::{GradientDirection, GradientFrame};
    use ritk_spatial::Vector;

    fn weighting(value: f64) -> DiffusionWeighting {
        DiffusionWeighting::from_seconds_per_square_millimeter(value).expect("finite weighting")
    }

    fn scheme(direction_count: usize) -> GradientScheme {
        let mut entries = vec![
            GradientDirection::new(weighting(0.0), Vector::new([0.0, 0.0, 0.0])).expect("valid b0"),
        ];
        let golden_angle = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
        for index in 0..direction_count {
            let z = 1.0 - 2.0 * (index as f64 + 0.5) / direction_count as f64;
            let radius = (1.0 - z * z).sqrt();
            let phi = golden_angle * index as f64;
            entries.push(
                GradientDirection::new(
                    weighting(1_000.0),
                    Vector::new([radius * phi.cos(), radius * phi.sin(), z]),
                )
                .expect("unit Fibonacci direction"),
            );
        }
        GradientScheme::new(entries, GradientFrame::Lps).expect("valid scheme")
    }

    fn tensor_signal(scheme: &GradientScheme, axis: [f64; 3]) -> Vec<f64> {
        const PARALLEL_DIFFUSIVITY: f64 = 0.0017;
        const PERPENDICULAR_DIFFUSIVITY: f64 = 0.0003;
        scheme
            .directions()
            .iter()
            .map(|entry| {
                let b = entry.weighting().seconds_per_square_millimeter();
                if b == 0.0 {
                    return 1.0;
                }
                let direction = entry.direction().to_array();
                let projection = direction
                    .iter()
                    .zip(axis)
                    .map(|(left, right)| left * right)
                    .sum::<f64>();
                let apparent = PERPENDICULAR_DIFFUSIVITY
                    + (PARALLEL_DIFFUSIVITY - PERPENDICULAR_DIFFUSIVITY) * projection.powi(2);
                (-b * apparent).exp()
            })
            .collect()
    }

    #[test]
    fn funk_radon_legendre_factors_match_closed_forms() {
        assert_eq!(legendre_at_zero(0), 1.0);
        assert_eq!(legendre_at_zero(2), -0.5);
        assert_eq!(legendre_at_zero(4), 0.375);
        assert_eq!(legendre_at_zero(6), -0.3125);
    }

    #[test]
    fn isotropic_signal_produces_constant_antipodal_odf() -> Result<(), OdfError> {
        let scheme = scheme(30);
        let signals = std::iter::once(1.0)
            .chain(std::iter::repeat_n(0.5, 30))
            .collect::<Vec<_>>();
        let odf = estimate_odf(&scheme, &signals, OdfConfig::default())?;
        let x = odf.evaluate_at_direction([1.0, 0.0, 0.0])?;
        let negative_x = odf.evaluate_at_direction([-1.0, 0.0, 0.0])?;
        let z = odf.evaluate_at_direction([0.0, 0.0, 1.0])?;
        assert!((x - negative_x).abs() < 1.0e-12);
        assert!(
            (x - z).abs() < 2.0e-3,
            "isotropic ODF differs by {}",
            (x - z).abs()
        );
        Ok(())
    }

    #[test]
    fn tensor_phantom_odf_peaks_on_analytical_axis() -> Result<(), OdfError> {
        let scheme = scheme(60);
        let odf = estimate_odf(
            &scheme,
            &tensor_signal(&scheme, [1.0, 0.0, 0.0]),
            OdfConfig::new(6, 0.002, weighting(50.0))?,
        )?;
        let x = odf.evaluate_at_direction([1.0, 0.0, 0.0])?;
        let y = odf.evaluate_at_direction([0.0, 1.0, 0.0])?;
        let z = odf.evaluate_at_direction([0.0, 0.0, 1.0])?;
        assert!(x > y, "x-axis ODF {x} must exceed y-axis ODF {y}");
        assert!(x > z, "x-axis ODF {x} must exceed z-axis ODF {z}");
        assert_eq!(odf.coefficients().len(), 28);
        Ok(())
    }

    #[test]
    fn invalid_configuration_signals_and_grid_are_typed_errors() {
        assert!(matches!(
            OdfConfig::new(3, 0.0, weighting(50.0)),
            Err(OdfError::Basis(_))
        ));
        assert!(matches!(
            OdfConfig::new(4, f64::NAN, weighting(50.0)),
            Err(OdfError::InvalidRegularization { .. })
        ));
        let scheme = scheme(30);
        let mut signals = vec![1.0; 31];
        signals[7] = f64::INFINITY;
        assert!(matches!(
            estimate_odf(&scheme, &signals, OdfConfig::default()),
            Err(OdfError::NonFiniteSignal { index: 7, .. })
        ));
        let odf = estimate_odf(&scheme, &vec![1.0; 31], OdfConfig::default())
            .expect("valid constant signal");
        assert!(matches!(
            odf.evaluate_on_grid(0, 12),
            Err(OdfError::InvalidGrid { .. })
        ));
    }

    #[test]
    fn spherical_grid_is_flat_and_finite() -> Result<(), OdfError> {
        let scheme = scheme(30);
        let odf = estimate_odf(&scheme, &vec![1.0; 31], OdfConfig::default())?;
        let grid = odf.evaluate_on_grid(8, 16)?;
        assert_eq!(grid.shape(), [8, 16]);
        assert_eq!(grid.values().len(), 128);
        assert!(grid.values().iter().all(|value| value.is_finite()));
        Ok(())
    }
}
