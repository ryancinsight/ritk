//! Constrained spherical deconvolution (CSD) — non-negative fibre ODF.
//!
//! Tournier et al. (2007) formulated the diffusion signal as the spherical
//! convolution of a fibre orientation distribution (fODF) with an axially
//! symmetric response function.  The deconvolution is ill-posed without a
//! constraint, so the fODF is forced non-negative via Lawson–Hanson NNLS.
//!
//! In the real spherical-harmonic basis the convolution becomes a diagonal
//! rescaling of each degree block:
//!
//! ```text
//! s_lm  =  √(4π/(2l+1)) · r_l · f_lm
//! ```
//!
//! where `r_l` are the rotational harmonics of the single-fibre response.
//! RITK assembles the matrix `B_resp = B · diag(κ_l)` and solves
//!
//! ```text
//! min ‖B_resp · f − S/S₀‖₂   subject to   f ≥ 0
//! ```
//!
//! through [`leto_ops::nnls()`].  The result is a non-negative fODF whose peaks
//! correspond to fibre directions — the input to FOD-based tractography.
//!
//! [Tournier et al. (2007)](https://doi.org/10.1016/j.neuroimage.2007.02.016)
//!
//! # Response function
//!
//! A [`ResponseFunction`] is defined by its rotational harmonics `r_l` for
//! `l = 0, 2, …, l_max`.  The convenience constructor
//! [`ResponseFunction::from_tensor`] computes them from an axially symmetric
//! diffusion tensor (parallel / perpendicular diffusivity) at a given
//! b-value by numerical projection onto the Legendre polynomials.
//!
//! # Relation to analytical Q-ball
//!
//! | Property | [`super::odf`] (analytical Q-ball) | CSD |
//! |----------|-----------------------------------|---------|
//! | Solver | Laplace-Beltrami-regularized least-squares | Lawson–Hanson NNLS |
//! | Output | Q-ball ODF (can be negative) | fODF (guaranteed non-negative) |
//! | Response | Implicit (Funk–Radon of signal SH) | Explicit (rotational harmonics `r_l`) |
//! | Purpose | Orientation distribution | Fibre orientation density |

use apollo_sht::{RealShError, RealSphericalHarmonicBasis, real_spherical_harmonic};
use leto::{Array1, Array2};
use leto_ops::{NnlsConfig, NnlsResult, nnls};
use ritk_diffusion_scheme::{DiffusionWeighting, GradientFrame, GradientScheme};

// ── Error ─────────────────────────────────────────────────────────────────────

/// Failure while configuring, estimating, or evaluating a fibre ODF via CSD.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CsdError {
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
    /// Mean b0 signal is not finite and strictly positive.
    #[error("baseline signal S0 must be finite and positive, got {value}")]
    InvalidBaseline {
        /// Computed baseline value.
        value: f64,
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
    /// The response function degree is too low for the requested basis.
    #[error("response function l_max is {response_l_max}; must be at least {config_l_max}")]
    ResponseDegreeTooLow {
        /// Maximum degree covered by the response.
        response_l_max: usize,
        /// Requested basis degree.
        config_l_max: usize,
    },
    /// The zero-degree rotational harmonic must be unity.
    #[error("r_0 must be 1.0, got {value}")]
    InvalidR0 {
        /// Non-unity `r_0` value.
        value: f64,
    },
    /// Evaluation angles or direction violate the finite domain.
    #[error("invalid fODF evaluation direction: {0}")]
    InvalidEvaluation(String),
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
    /// Apollo rejected the even-degree basis configuration.
    #[error("spherical-harmonic basis error: {0}")]
    Basis(#[from] RealShError),
    /// The NNLS solver failed.
    #[error("NNLS solve failed: {0}")]
    NnlsFailed(String),
    /// Every spatial dimension of a fODF volume must be positive.
    #[error("fODF volume shape [{nx}, {ny}, {nz}] has a zero dimension")]
    VolumeShapeEmpty {
        /// X dimension.
        nx: usize,
        /// Y dimension.
        ny: usize,
        /// Z dimension.
        nz: usize,
    },
    /// The coefficient count does not match `nx × ny × nz × nc`.
    #[error("expected {expected} fODF coefficients for shape {nx}×{ny}×{nz}×{nc}, got {actual}")]
    VolumeCoefficientCountMismatch {
        /// Expected length.
        expected: usize,
        /// Actual length.
        actual: usize,
        /// X dimension.
        nx: usize,
        /// Y dimension.
        ny: usize,
        /// Z dimension.
        nz: usize,
        /// Coefficients per voxel.
        nc: usize,
    },
    /// Voxel spacing must be finite and positive.
    #[error("voxel spacing [{sx}, {sy}, {sz}] must be finite and positive")]
    VolumeSpacingInvalid {
        /// X spacing.
        sx: f64,
        /// Y spacing.
        sy: f64,
        /// Z spacing.
        sz: f64,
    },
    /// Volume origin must be finite.
    #[error("volume origin [{ox}, {oy}, {oz}] must be finite")]
    VolumeOriginInvalid {
        /// X origin.
        ox: f64,
        /// Y origin.
        oy: f64,
        /// Z origin.
        oz: f64,
    },
}

// ── Response Function ─────────────────────────────────────────────────────────

/// Axially symmetric single-fibre response function.
///
/// The rotational harmonics `r_l` are the SH coefficients of the signal
/// produced by a perfectly aligned fibre when projected onto the Legendre
/// polynomials `P_l(cos θ)`.  `r_0` is always 1.0 (the response is
/// normalised to unit baseline signal).
#[derive(Debug, Clone)]
pub struct ResponseFunction {
    /// Rotational harmonics `r_0, r_2, r_4, …, r_{l_max}`.
    harmonics: Box<[f64]>,
    /// Corresponding degrees.
    degrees: Box<[usize]>,
}

impl ResponseFunction {
    /// Construct from precomputed rotational harmonics.
    ///
    /// `harmonics` must be `[r_0, r_2, r_4, …, r_{l_max}]` with `r_0 == 1.0`.
    /// Each entry corresponds to one even SH degree.
    ///
    /// # Errors
    ///
    /// Returns [`CsdError::InvalidR0`] if the first harmonic is not 1.0.
    pub fn new(harmonics: Vec<f64>) -> Result<Self, CsdError> {
        if harmonics.is_empty() {
            return Err(CsdError::InvalidR0 { value: f64::NAN });
        }
        if harmonics.iter().any(|value| !value.is_finite()) {
            return Err(CsdError::InvalidR0 { value: f64::NAN });
        }
        if (harmonics[0] - 1.0).abs() > 1e-12 {
            return Err(CsdError::InvalidR0 {
                value: harmonics[0],
            });
        }
        let degrees: Box<[usize]> = (0..harmonics.len()).map(|i| i * 2).collect();
        Ok(Self {
            harmonics: harmonics.into_boxed_slice(),
            degrees,
        })
    }

    /// Compute rotational harmonics from an axially symmetric diffusion tensor.
    ///
    /// Samples the signal profile `exp(-b · (ad cos²θ + rd sin²θ))` on a dense
    /// polar grid and projects each even degree onto the Legendre polynomial
    /// `P_l(cos θ)` via numerical quadrature.
    ///
    /// `ad` and `rd` are the axial and radial diffusivities in mm²/s.
    /// `b_value` is the shell b-value in s/mm².
    /// `l_max` is the maximum even SH degree.
    ///
    /// The response is normalised so that `r_0 = 1.0`.
    pub fn from_tensor(b_value: f64, ad: f64, rd: f64, l_max: usize) -> Result<Self, CsdError> {
        RealSphericalHarmonicBasis::new(l_max)?;

        const N_THETA: usize = 512;
        let mut harmonics = Vec::with_capacity(l_max / 2 + 1);
        let mut r0_value = 0.0;

        for degree in (0..=l_max).step_by(2) {
            let mut integral = 0.0;
            for i in 0..N_THETA {
                let theta = std::f64::consts::PI * (i as f64 + 0.5) / N_THETA as f64;
                let cos_theta = theta.cos();
                let sin_theta = theta.sin();
                // Apparent diffusion coefficient for a tensor aligned with z.
                let adc = ad * cos_theta * cos_theta + rd * sin_theta * sin_theta;
                let signal = (-b_value * adc).exp();
                let legendre = legendre_p(degree, cos_theta);
                integral += signal * legendre * sin_theta;
            }
            integral *= std::f64::consts::PI / N_THETA as f64;
            // Normalisation factor for the Legendre series of an axially
            // symmetric function: r_l = (2l+1)/2 · ∫ R(θ) P_l(cos θ) sin θ dθ.
            let r_l = (2.0 * degree as f64 + 1.0) / 2.0 * integral;
            harmonics.push(r_l);
            if degree == 0 {
                r0_value = r_l;
            }
        }

        // Normalise so r_0 == 1.0.
        for r in &mut harmonics {
            *r /= r0_value;
        }

        Self::new(harmonics)
    }

    /// Rotational harmonics `r_0, r_2, r_4, …`.
    #[must_use]
    pub fn harmonics(&self) -> &[f64] {
        &self.harmonics
    }

    /// Even degrees corresponding to each harmonic.
    #[must_use]
    pub fn degrees(&self) -> &[usize] {
        &self.degrees
    }

    /// The maximum even degree.
    #[must_use]
    pub fn l_max(&self) -> usize {
        self.degrees.last().copied().unwrap_or(0)
    }

    /// Number of rotational harmonics.
    #[must_use]
    pub fn len(&self) -> usize {
        self.harmonics.len()
    }

    /// True when no harmonics are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.harmonics.is_empty()
    }
}

/// Evaluate the Legendre polynomial `P_n(x)` via the Bonnet recurrence.
fn legendre_p(n: usize, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return x;
    }
    let mut p_prev = 1.0;
    let mut p_curr = x;
    for k in 2..=n {
        let p_next = ((2 * k - 1) as f64 * x * p_curr - (k - 1) as f64 * p_prev) / k as f64;
        p_prev = p_curr;
        p_curr = p_next;
    }
    p_curr
}

// ── Configuration ─────────────────────────────────────────────────────────────

/// Validated CSD configuration.
#[derive(Debug, Clone)]
pub struct CsdConfig {
    l_max: usize,
    b0_threshold: DiffusionWeighting,
    nnls_config: NnlsConfig,
}

impl CsdConfig {
    /// Construct a CSD configuration.
    ///
    /// `l_max` must be even and at least two.  `b0_threshold` classifies
    /// reference and weighted volumes.
    ///
    /// # Errors
    ///
    /// Returns [`CsdError::Basis`] for an invalid degree.
    pub fn new(
        l_max: usize,
        b0_threshold: DiffusionWeighting,
        nnls_config: NnlsConfig,
    ) -> Result<Self, CsdError> {
        RealSphericalHarmonicBasis::new(l_max)?;
        Ok(Self {
            l_max,
            b0_threshold,
            nnls_config,
        })
    }

    /// Maximum even spherical-harmonic degree.
    #[must_use]
    pub const fn l_max(&self) -> usize {
        self.l_max
    }

    /// Threshold separating b0 and weighted acquisitions.
    #[must_use]
    pub const fn b0_threshold(&self) -> DiffusionWeighting {
        self.b0_threshold
    }

    /// NNLS convergence parameters.
    #[must_use]
    pub const fn nnls_config(&self) -> &NnlsConfig {
        &self.nnls_config
    }
}

impl Default for CsdConfig {
    fn default() -> Self {
        Self {
            l_max: 8,
            b0_threshold: DiffusionWeighting::from_seconds_per_square_millimeter(50.0)
                .expect("invariant: default b0 threshold is finite and nonnegative"),
            nnls_config: NnlsConfig::default(),
        }
    }
}

/// A peak direction extracted from an fODF, with its amplitude.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FodPeak {
    /// Unit Cartesian direction of the peak, in the field's coordinate frame.
    pub direction: [f64; 3],
    /// fODF amplitude at this peak.
    pub amplitude: f64,
}

impl FodPeak {
    /// Azimuthal angle `φ ∈ [0, 2π)` from the +x axis.
    #[must_use]
    pub fn phi(&self) -> f64 {
        self.direction[1].atan2(self.direction[0])
    }

    /// Polar angle `θ ∈ [0, π]` from the +z axis.
    #[must_use]
    pub fn theta(&self) -> f64 {
        self.direction[2].clamp(-1.0, 1.0).acos()
    }
}

// ── Contiguous fODF Grid ──────────────────────────────────────────────────────

/// Contiguous fODF samples on an equiangular spherical grid.
#[derive(Debug, Clone, PartialEq)]
pub struct SphericalFodGrid {
    shape: [usize; 2],
    values: Box<[f64]>,
}

impl SphericalFodGrid {
    /// Grid shape `[theta_samples, phi_samples]`.
    #[must_use]
    pub const fn shape(&self) -> [usize; 2] {
        self.shape
    }

    /// Row-major fODF values with phi varying fastest.
    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }
}

// ── Fibre ODF Field ───────────────────────────────────────────────────────────

/// Estimated fibre orientation distribution at one voxel via CSD.
///
/// All coefficients are guaranteed non-negative by the NNLS solver.
#[derive(Debug, Clone)]
pub struct FodField {
    coefficients: Box<[f64]>,
    basis: RealSphericalHarmonicBasis,
    baseline_signal: f64,
    residual_norm: f64,
    nnls_iterations: usize,
    nnls_converged: bool,
    frame: GradientFrame,
}

impl FodField {
    /// Non-negative fODF coefficients in Apollo's degree-major even-order basis.
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

    /// ‖B_resp · f − S/S₀‖₂ after NNLS convergence or iteration limit.
    #[must_use]
    pub const fn residual_norm(&self) -> f64 {
        self.residual_norm
    }

    /// Number of NNLS active-set iterations.
    #[must_use]
    pub const fn nnls_iterations(&self) -> usize {
        self.nnls_iterations
    }

    /// True if the NNLS convergence tolerance was met.
    #[must_use]
    pub const fn nnls_converged(&self) -> bool {
        self.nnls_converged
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
    /// Returns [`CsdError::InvalidEvaluation`] unless `theta` is finite in
    /// `[0, pi]` and `phi` is finite.
    pub fn evaluate(&self, theta: f64, phi: f64) -> Result<f64, CsdError> {
        if !theta.is_finite() || !(0.0..=std::f64::consts::PI).contains(&theta) {
            return Err(CsdError::InvalidEvaluation(format!(
                "theta must be in [0, pi], got {theta}"
            )));
        }
        if !phi.is_finite() {
            return Err(CsdError::InvalidEvaluation(format!(
                "phi must be finite, got {phi}"
            )));
        }
        Ok(self.evaluate_unchecked(theta, phi))
    }

    /// Evaluate at a finite unit Cartesian direction in [`Self::frame`].
    ///
    /// # Errors
    ///
    /// Returns [`CsdError::InvalidEvaluation`] when a component is non-finite
    /// or the norm differs from one by more than `1e-6`.
    pub fn evaluate_at_direction(&self, direction: [f64; 3]) -> Result<f64, CsdError> {
        if direction.iter().any(|value| !value.is_finite()) {
            return Err(CsdError::InvalidEvaluation(format!(
                "direction is not finite: {direction:?}"
            )));
        }
        let norm = direction
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        if (norm - 1.0).abs() > 1.0e-6 {
            return Err(CsdError::InvalidEvaluation(format!(
                "direction must be unit length, norm is {norm}"
            )));
        }
        let theta = direction[2].clamp(-1.0, 1.0).acos();
        let phi = direction[1].atan2(direction[0]);
        Ok(self.evaluate_unchecked(theta, phi))
    }

    /// Evaluate a contiguous equiangular spherical grid.
    ///
    /// # Errors
    ///
    /// Returns [`CsdError::InvalidGrid`] for an empty dimension, element-count
    /// overflow, or allocation failure.
    pub fn evaluate_on_grid(
        &self,
        theta_samples: usize,
        phi_samples: usize,
    ) -> Result<SphericalFodGrid, CsdError> {
        if theta_samples == 0 || phi_samples == 0 {
            return Err(CsdError::InvalidGrid {
                theta_samples,
                phi_samples,
                reason: "both dimensions must be nonzero",
            });
        }
        let count = theta_samples
            .checked_mul(phi_samples)
            .ok_or(CsdError::InvalidGrid {
                theta_samples,
                phi_samples,
                reason: "element count overflows usize",
            })?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|_| CsdError::InvalidGrid {
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
        Ok(SphericalFodGrid {
            shape: [theta_samples, phi_samples],
            values: values.into_boxed_slice(),
        })
    }

    fn evaluate_unchecked(&self, theta: f64, phi: f64) -> f64 {
        self.basis
            .iter_lm()
            .zip(self.coefficients.iter())
            .map(|((_, degree, order), coefficient)| {
                coefficient
                    * real_spherical_harmonic(degree, order, theta, phi)
                        .expect("invariant: SH evaluation with pre-validated basis")
            })
            .sum()
    }

    /// Extract local maxima (peaks) from the fODF via a spherical grid search.
    ///
    /// Samples the fODF on a dense equiangular `theta×phi` grid, then returns
    /// every sample that exceeds all eight neighbours and a configurable
    /// relative-amplitude floor.  Peaks are sorted by descending amplitude.
    ///
    /// `grid_theta` and `grid_phi` control the search resolution.
    /// `relative_threshold` discards peaks whose amplitude is below
    /// `relative_threshold × max(fODF)`; 0.1 is a reasonable default.
    ///
    /// # Errors
    ///
    /// Returns [`CsdError::InvalidGrid`] when the grid dimensions are invalid
    /// or the allocation fails.
    pub fn find_peaks(
        &self,
        grid_theta: usize,
        grid_phi: usize,
        relative_threshold: f64,
    ) -> Result<Vec<FodPeak>, CsdError> {
        let grid = self.evaluate_on_grid(grid_theta, grid_phi)?;
        let values = grid.values();
        let [n_theta, n_phi] = grid.shape();
        let max_value = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let floor = max_value * relative_threshold;

        let index = |ti: usize, pi: usize| -> usize { ti * n_phi + pi };
        let mut peaks = Vec::new();

        for ti in 0..n_theta {
            for pi in 0..n_phi {
                let value = values[index(ti, pi)];
                if value < floor {
                    continue;
                }
                // Eight-neighbour check with toroidal wrap in phi.
                // Pole rows (ti == 0 or ti == n_theta-1) are not excluded:
                // the deduplication pass below merges physically identical
                // pole peaks that the ϕ-periodic grid produces.
                let is_maximum = [
                    (0, -1),
                    (0, 1),
                    (-1, -1),
                    (-1, 0),
                    (-1, 1),
                    (1, -1),
                    (1, 0),
                    (1, 1),
                ]
                .into_iter()
                .filter_map(|(dt, dp)| {
                    let nt = ti.wrapping_add_signed(dt);
                    if nt >= n_theta {
                        return None;
                    }
                    let np = (pi.wrapping_add_signed(dp)) % n_phi;
                    Some(values[index(nt, np)])
                })
                .all(|neighbour| value > neighbour);
                if !is_maximum {
                    continue;
                }
                let theta = std::f64::consts::PI * (ti as f64 + 0.5) / n_theta as f64;
                let phi = std::f64::consts::TAU * pi as f64 / n_phi as f64;
                let sin_theta = theta.sin();
                peaks.push(FodPeak {
                    direction: [sin_theta * phi.cos(), sin_theta * phi.sin(), theta.cos()],
                    amplitude: value,
                });
            }
        }

        peaks.sort_by(|a, b| {
            b.amplitude
                .partial_cmp(&a.amplitude)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Deduplicate: keep the highest-amplitude peak when multiple grid
        // points represent the same physical direction (common at poles
        // where the φ-periodic grid maps one direction to many φ values).
        const DEDUP_DOT: f64 = 0.996; // cos(≈5°)
        let mut kept: Vec<FodPeak> = Vec::with_capacity(peaks.len());
        for peak in peaks {
            if kept.iter().any(|existing| {
                existing.direction[0] * peak.direction[0]
                    + existing.direction[1] * peak.direction[1]
                    + existing.direction[2] * peak.direction[2]
                    > DEDUP_DOT
            }) {
                continue;
            }
            kept.push(peak);
        }
        Ok(kept)
    }

    /// Construct an fODF field from externally computed coefficients.
    ///
    /// This is used for spatial interpolation in whole-brain tractography:
    /// trilinear interpolation produces a new coefficient vector at an
    /// off-grid position, and this constructor wraps it for peak finding.
    ///
    /// The diagnostics fields (`baseline_signal`, `residual_norm`,
    /// `nnls_iterations`, `nnls_converged`) are not used by peak extraction
    /// or evaluation and may carry placeholder values.
    pub fn from_coefficients(
        coefficients: Box<[f64]>,
        basis: RealSphericalHarmonicBasis,
        baseline_signal: f64,
        residual_norm: f64,
        nnls_iterations: usize,
        nnls_converged: bool,
        frame: GradientFrame,
    ) -> Self {
        Self {
            coefficients,
            basis,
            baseline_signal,
            residual_norm,
            nnls_iterations,
            nnls_converged,
            frame,
        }
    }
}

// ── Estimation ────────────────────────────────────────────────────────────────

/// Estimate a non-negative fibre ODF from one voxel's signals via CSD.
///
/// The response function's rotational harmonics `r_l` are combined with the
/// Apollo real SH design matrix to form the deconvolution matrix
/// `B_resp = B · diag(κ_l)`, then [`leto_ops::nnls()`] enforces `f ≥ 0`.
///
/// # Errors
///
/// Returns a typed error for count mismatch, non-finite signals, missing b0
/// or weighted samples, response/basis degree mismatch, invalid baseline,
/// underdetermined system, or a failed NNLS solve.
pub fn estimate_fod(
    scheme: &GradientScheme,
    signals: &[f64],
    response: &ResponseFunction,
    config: &CsdConfig,
) -> Result<FodField, CsdError> {
    // ── Validation ────────────────────────────────────────────────────────
    if signals.len() != scheme.len() {
        return Err(CsdError::SignalLengthMismatch {
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
        return Err(CsdError::NonFiniteSignal { index, value });
    }

    let b0_indices = scheme.b0_indices(config.b0_threshold());
    let dwi_indices = scheme.dwi_indices(config.b0_threshold());
    if b0_indices.is_empty() {
        return Err(CsdError::NoB0Volumes);
    }
    if dwi_indices.is_empty() {
        return Err(CsdError::NoDwiDirections);
    }

    let baseline_signal =
        b0_indices.iter().map(|index| signals[*index]).sum::<f64>() / b0_indices.len() as f64;
    if !baseline_signal.is_finite() || baseline_signal <= 0.0 {
        return Err(CsdError::InvalidBaseline {
            value: baseline_signal,
        });
    }

    let response_l_max = response.l_max();
    if response_l_max < config.l_max() {
        return Err(CsdError::ResponseDegreeTooLow {
            response_l_max,
            config_l_max: config.l_max(),
        });
    }

    let basis = RealSphericalHarmonicBasis::new(config.l_max())?;
    let coefficient_count = basis.num_coefficients();
    if dwi_indices.len() < coefficient_count {
        return Err(CsdError::Underdetermined {
            direction_count: dwi_indices.len(),
            coefficient_count,
        });
    }

    // ── Build deconvolution matrix ────────────────────────────────────────
    let directions = dwi_indices
        .iter()
        .map(|index| scheme.directions()[*index].direction().to_array())
        .collect::<Vec<_>>();
    let normalized = dwi_indices
        .iter()
        .map(|index| signals[*index] / baseline_signal)
        .collect::<Vec<_>>();
    let design = basis.design_matrix(&directions)?;

    let deconv = build_deconvolution_matrix(&design, &basis, response);

    // ── NNLS solve ────────────────────────────────────────────────────────
    let rhs = Array1::from_vec(dwi_indices.len(), normalized.clone())
        .map_err(|error| CsdError::NnlsFailed(error.to_string()))?;
    let nnls_result: NnlsResult = nnls(&deconv.view(), &rhs.view(), config.nnls_config)
        .map_err(|error| CsdError::NnlsFailed(error.to_string()))?;

    let coefficients = nnls_result
        .solution
        .iter()
        .copied()
        .collect::<Vec<_>>()
        .into_boxed_slice();

    Ok(FodField {
        coefficients,
        basis,
        baseline_signal,
        residual_norm: nnls_result.residual_norm,
        nnls_iterations: nnls_result.iterations,
        nnls_converged: nnls_result.converged,
        frame: scheme.frame(),
    })
}

/// Build `B_resp = B · diag(κ_l)` where `κ_{l} = 4π/(2l+1) · r_l`.
///
/// The rotational harmonics `r_l` are Legendre coefficients
/// `c_l = (2l+1)/2 ∫ R(θ) P_l(cos θ) sin θ dθ`, normalised so `c_0 = 1`.
/// In the spherical convolution `s_lm = √(4π/(2l+1)) · R_l⁰ · f_lm` expressed
/// with Legendre instead of SH coefficients, the factor becomes
/// `4π/(2l+1)`.  The same `κ_l` is applied to every coefficient of degree
/// `l` (all allowed orders `m = -l, …, l`).
fn build_deconvolution_matrix(
    design: &Array2<f64>,
    basis: &RealSphericalHarmonicBasis,
    response: &ResponseFunction,
) -> Array2<f64> {
    let [n_measurements, n_coeffs] = design.shape();
    let response_lookup: Vec<f64> = (0..n_coeffs)
        .map(|idx| {
            let (degree, _) = basis.index_to_lm(idx).expect("index within basis range");
            let response_index = degree / 2;
            let r_l = response.harmonics()[response_index];
            // κ_l = 4π/(2l+1) · r_l  (Legendre-coefficient convention).
            4.0 * std::f64::consts::PI / (2.0 * degree as f64 + 1.0) * r_l
        })
        .collect();

    let mut deconv = Array2::zeros([n_measurements, n_coeffs]);
    for i in 0..n_measurements {
        for j in 0..n_coeffs {
            deconv[[i, j]] = design[[i, j]] * response_lookup[j];
        }
    }
    deconv
}

// ── FOD Volume (whole-brain tractography) ─────────────────────────────────────

/// A 3-D volume of fODF coefficients on a regular grid.
///
/// Stores one coefficient vector per voxel in z-major (slice-first) order.
/// Supports trilinear interpolation for sub-voxel direction queries during
/// whole-brain tractography via [`FodVolume::direction_at`].
#[derive(Debug, Clone)]
pub struct FodVolume {
    /// Flat coefficient array: `[z][y][x][coefficient_index]`.
    coefficients: Box<[f64]>,
    /// Grid dimensions `[nx, ny, nz]`.
    shape: [usize; 3],
    /// Voxel size in physical units (mm), `[sx, sy, sz]`.
    spacing: [f64; 3],
    /// Physical position of the first voxel centre `[ox, oy, oz]`.
    origin: [f64; 3],
    /// Shared even-order real SH basis for peak extraction.
    basis: RealSphericalHarmonicBasis,
    /// Coordinate frame for direction queries.
    frame: GradientFrame,
}

impl FodVolume {
    /// Construct a volume from a flat coefficient array.
    ///
    /// `coefficients` must have exactly `nx × ny × nz × nc` elements where
    /// `nc` is the number of coefficients in `basis`.  Spacing must be finite
    /// and positive; origin must be finite.
    ///
    /// # Errors
    ///
    /// Returns a typed [`CsdError`] for a mismatched coefficient count, zero
    /// dimension, or invalid spacing / origin.
    pub fn new(
        coefficients: Box<[f64]>,
        shape: [usize; 3],
        spacing: [f64; 3],
        origin: [f64; 3],
        basis: RealSphericalHarmonicBasis,
        frame: GradientFrame,
    ) -> Result<Self, CsdError> {
        let [nx, ny, nz] = shape;
        if nx == 0 || ny == 0 || nz == 0 {
            return Err(CsdError::VolumeShapeEmpty { nx, ny, nz });
        }
        let nc = basis.num_coefficients();
        let expected = nx
            .checked_mul(ny)
            .and_then(|v| v.checked_mul(nz))
            .and_then(|v| v.checked_mul(nc))
            .ok_or(CsdError::VolumeCoefficientCountMismatch {
                expected: 0,
                actual: coefficients.len(),
                nx,
                ny,
                nz,
                nc,
            })?;
        if coefficients.len() != expected {
            return Err(CsdError::VolumeCoefficientCountMismatch {
                expected,
                actual: coefficients.len(),
                nx,
                ny,
                nz,
                nc,
            });
        }
        let [sx, sy, sz] = spacing;
        if !sx.is_finite()
            || sx <= 0.0
            || !sy.is_finite()
            || sy <= 0.0
            || !sz.is_finite()
            || sz <= 0.0
        {
            return Err(CsdError::VolumeSpacingInvalid { sx, sy, sz });
        }
        let [ox, oy, oz] = origin;
        if !ox.is_finite() || !oy.is_finite() || !oz.is_finite() {
            return Err(CsdError::VolumeOriginInvalid { ox, oy, oz });
        }
        Ok(Self {
            coefficients,
            shape,
            spacing,
            origin,
            basis,
            frame,
        })
    }

    /// Number of SH coefficients per voxel.
    #[must_use]
    pub fn coefficient_count(&self) -> usize {
        self.basis.num_coefficients()
    }

    /// Grid dimensions `[nx, ny, nz]`.
    #[must_use]
    pub const fn shape(&self) -> [usize; 3] {
        self.shape
    }

    /// Voxel spacing in physical units `[sx, sy, sz]`.
    #[must_use]
    pub const fn spacing(&self) -> [f64; 3] {
        self.spacing
    }

    /// Origin of the first voxel centre `[ox, oy, oz]`.
    #[must_use]
    pub const fn origin(&self) -> [f64; 3] {
        self.origin
    }

    /// Coordinate frame for direction queries.
    #[must_use]
    pub const fn frame(&self) -> GradientFrame {
        self.frame
    }

    /// Convert physical coordinates to continuous voxel indices.
    ///
    /// Returns `None` when any component of `point` is non-finite.
    fn world_to_voxel(&self, point: &ritk_spatial::Point<3>) -> Option<[f64; 3]> {
        let [px, py, pz] = point.to_array();
        if !px.is_finite() || !py.is_finite() || !pz.is_finite() {
            return None;
        }
        let [ox, oy, oz] = self.origin;
        let [sx, sy, sz] = self.spacing;
        Some([(px - ox) / sx, (py - oy) / sy, (pz - oz) / sz])
    }

    /// Trilinear interpolation of fODF coefficients at a physical point.
    ///
    /// Returns `None` when the point maps to a continuous voxel index outside
    /// `[-0.5, shape[i] - 0.5)` for any axis — i.e., when the interpolation
    /// stencil extends beyond the grid.
    pub fn interpolate_coefficients_at(&self, point: &ritk_spatial::Point<3>) -> Option<Vec<f64>> {
        let [fx, fy, fz] = self.world_to_voxel(point)?;
        let [nx, ny, nz] = self.shape;

        // Reject points whose stencil would reach outside the grid.
        if fx < -0.5 || fx >= nx as f64 - 0.5 {
            return None;
        }
        if fy < -0.5 || fy >= ny as f64 - 0.5 {
            return None;
        }
        if fz < -0.5 || fz >= nz as f64 - 0.5 {
            return None;
        }

        // Clamp continuous voxel coordinates so that floor and floor+1 are
        // valid indices for the interpolation stencil.  Near-boundary points
        // within the half-voxel margin degenerate to nearest-neighbour
        // (wx = 0.0) rather than extrapolating outside the grid.
        let fx = fx.clamp(0.0, (nx - 1) as f64);
        let fy = fy.clamp(0.0, (ny - 1) as f64);
        let fz = fz.clamp(0.0, (nz - 1) as f64);
        let ix = fx.floor() as usize;
        let iy = fy.floor() as usize;
        let iz = fz.floor() as usize;
        let ix1 = (ix + 1).min(nx - 1);
        let iy1 = (iy + 1).min(ny - 1);
        let iz1 = (iz + 1).min(nz - 1);

        let wx = fx - ix as f64;
        let wy = fy - iy as f64;
        let wz = fz - iz as f64;

        let nc = self.coefficient_count();
        let nxy = nx * ny * nc;
        let nxnc = nx * nc;

        let idx = |z: usize, y: usize, x: usize| -> usize { z * nxy + y * nxnc + x * nc };

        let mut result = vec![0.0; nc];
        for (c, value) in result.iter_mut().enumerate() {
            let v000 = self.coefficients[idx(iz, iy, ix) + c];
            let v100 = self.coefficients[idx(iz, iy, ix1) + c];
            let v010 = self.coefficients[idx(iz, iy1, ix) + c];
            let v110 = self.coefficients[idx(iz, iy1, ix1) + c];
            let v001 = self.coefficients[idx(iz1, iy, ix) + c];
            let v101 = self.coefficients[idx(iz1, iy, ix1) + c];
            let v011 = self.coefficients[idx(iz1, iy1, ix) + c];
            let v111 = self.coefficients[idx(iz1, iy1, ix1) + c];

            *value = (1.0 - wx) * (1.0 - wy) * (1.0 - wz) * v000
                + wx * (1.0 - wy) * (1.0 - wz) * v100
                + (1.0 - wx) * wy * (1.0 - wz) * v010
                + wx * wy * (1.0 - wz) * v110
                + (1.0 - wx) * (1.0 - wy) * wz * v001
                + wx * (1.0 - wy) * wz * v101
                + (1.0 - wx) * wy * wz * v011
                + wx * wy * wz * v111;
        }
        Some(result)
    }

    /// Interpolate the fODF at a physical point and extract the strongest
    /// peak direction.
    ///
    /// Performs trilinear coefficient interpolation followed by a
    /// spherical-grid peak search on the interpolated fODF.  Returns `None`
    /// outside the volume or when no peak meets the relative-amplitude
    /// threshold.
    ///
    /// # Panics
    ///
    /// Panics internally if the interpolated coefficient count does not match
    /// the basis — this is an invariant violation, not a recoverable error.
    pub fn direction_at(
        &self,
        point: &ritk_spatial::Point<3>,
        grid_theta: usize,
        grid_phi: usize,
        relative_threshold: f64,
    ) -> Option<ritk_spatial::Vector<3>> {
        let coefficients = self.interpolate_coefficients_at(point)?;
        let coefficients: Box<[f64]> = coefficients.into_boxed_slice();
        assert_eq!(
            coefficients.len(),
            self.basis.num_coefficients(),
            "interpolated coefficient count must match basis"
        );
        let field = FodField::from_coefficients(
            coefficients,
            self.basis.clone(),
            1.0,   // baseline_signal — not consumed by find_peaks
            0.0,   // residual_norm
            0,     // nnls_iterations
            false, // nnls_converged
            self.frame,
        );
        let peaks = field
            .find_peaks(grid_theta, grid_phi, relative_threshold)
            .ok()?;
        peaks
            .first()
            .map(|peak| ritk_spatial::Vector::new(peak.direction))
    }
}

#[cfg(test)]
mod tests;
