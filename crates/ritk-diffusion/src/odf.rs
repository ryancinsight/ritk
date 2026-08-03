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
use ritk_diffusion_scheme::{DiffusionWeighting, GradientFrame, GradientScheme};

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
    /// Weighted acquisitions do not lie on one q-space shell.
    #[error(
        "diffusion weighting at acquisition index {index} is {value} s/mm²; expected {reference} ± {tolerance} s/mm²"
    )]
    MixedShells {
        /// First weighted acquisition's b-value in s/mm².
        reference: f64,
        /// Acquisition index outside the configured shell tolerance.
        index: usize,
        /// Off-shell b-value in s/mm².
        value: f64,
        /// Allowed absolute shell difference in s/mm².
        tolerance: f64,
    },
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
    /// ODF evaluation overflowed to infinity.
    #[error("ODF evaluation produced a non-finite value")]
    NonFiniteEvaluation,
    /// A normalized signal value is non-finite.
    #[error("normalized signal at acquisition index {index} is not finite: {value}")]
    NonFiniteNormalizedSignal {
        /// Acquisition-order index.
        index: usize,
        /// Invalid normalized signal value.
        value: f64,
    },
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
    shell_tolerance: DiffusionWeighting,
}

impl OdfConfig {
    /// Construct a Q-ball configuration.
    ///
    /// `l_max` must be even and at least two. `regularization` is the
    /// nonnegative Laplace-Beltrami normal-equation weight. `b0_threshold`
    /// classifies reference and weighted volumes. `shell_tolerance` is the
    /// maximum absolute b-value difference allowed among weighted samples;
    /// analytical Q-ball fits one q-space shell.
    ///
    /// # Errors
    ///
    /// Returns a typed error for an invalid degree or regularization value.
    pub fn new(
        l_max: usize,
        regularization: f64,
        b0_threshold: DiffusionWeighting,
        shell_tolerance: DiffusionWeighting,
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
            shell_tolerance,
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

    /// Maximum absolute b-value difference within the fitted shell.
    #[must_use]
    pub const fn shell_tolerance(self) -> DiffusionWeighting {
        self.shell_tolerance
    }
}

impl Default for OdfConfig {
    fn default() -> Self {
        Self {
            l_max: 4,
            regularization: 0.006,
            b0_threshold: DiffusionWeighting::from_seconds_per_square_millimeter(50.0)
                .expect("invariant: default b0 threshold is finite and nonnegative"),
            shell_tolerance: DiffusionWeighting::from_seconds_per_square_millimeter(0.0)
                .expect("invariant: default shell tolerance is zero"),
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
    frame: GradientFrame,
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

    /// Coordinate frame used by evaluation directions and ODF peaks.
    #[must_use]
    pub const fn frame(&self) -> GradientFrame {
        self.frame
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
        self.evaluate_unchecked(theta, phi)
    }

    /// Evaluate at a finite unit Cartesian direction in [`Self::frame`].
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
        self.evaluate_unchecked(theta, phi)
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
                values.push(self.evaluate_unchecked(theta, phi)?);
            }
        }
        Ok(SphericalOdfGrid {
            shape: [theta_samples, phi_samples],
            values: values.into_boxed_slice(),
        })
    }

    fn evaluate_unchecked(&self, theta: f64, phi: f64) -> Result<f64, OdfError> {
        let mut result = 0.0;
        for ((_, degree, order), coefficient) in
            self.basis.iter_lm().zip(self.coefficients.iter())
        {
            let basis_value = real_spherical_harmonic(degree, order, theta, phi)
                .expect("invariant: SH evaluation with pre-validated basis");
            result += coefficient * basis_value;
        }
        if result.is_finite() {
            Ok(result)
        } else {
            Err(OdfError::NonFiniteEvaluation)
        }
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

    let reference_weighting = scheme.directions()[dwi_indices[0]]
        .weighting()
        .seconds_per_square_millimeter();
    let shell_tolerance = config.shell_tolerance().seconds_per_square_millimeter();
    if let Some((index, value)) = dwi_indices.iter().copied().find_map(|index| {
        let value = scheme.directions()[index]
            .weighting()
            .seconds_per_square_millimeter();
        ((value - reference_weighting).abs() > shell_tolerance).then_some((index, value))
    }) {
        return Err(OdfError::MixedShells {
            reference: reference_weighting,
            index,
            value,
            tolerance: shell_tolerance,
        });
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
    if let Some((acq_index, &value)) = dwi_indices
        .iter()
        .zip(normalized.iter())
        .find(|&(_, &value)| !value.is_finite())
    {
        return Err(OdfError::NonFiniteNormalizedSignal {
            index: *acq_index,
            value,
        });
    }
    let design = basis.design_matrix(&directions)?;
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
        frame: scheme.frame(),
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
mod tests;
