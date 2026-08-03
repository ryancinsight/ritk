//! Neurite Orientation Dispersion and Density Imaging (NODDI).
//!
//! Zhang et al. (2012) model: a 3-compartment tissue model that separates
//! water diffusion into intra-neurite (restricted), extra-neurite (hindered),
//! and CSF (free) pools.  The intra-cellular compartment is a Watson-
//! distributed population of zero-radius cylinders ("sticks"); the
//! extra-cellular compartment is an isotropic ball; CSF is free water.
//!
//! # Signal model
//!
//! ```text
//! S/S₀ = (1 − f_iso)·[f_intra·A_ic + (1 − f_intra)·A_ec] + f_iso·A_iso
//! ```
//!
//! where `A_ic` is the Watson-averaged stick signal evaluated by numerical
//! quadrature over the sphere, `A_ec = exp(−b·d_ec)` and
//! `A_iso = exp(−b·d_iso)`.  Fitting is by damped Gauss-Newton
//! (Levenberg-Marquardt) through [`coeus_optim::levenberg_marquardt`].
//!
//! # Derived metrics
//!
//! | Metric | Symbol | Formula | Interpretation |
//! |--------|--------|---------|----------------|
//! | Neurite density | NDI | `f_intra` | Fraction of non-CSF water in neurites |
//! | Orientation dispersion | ODI | `f_odi` | `(2/π)·arctan(1/κ)` ∈ [0, 1] |
//! | CSF fraction | f_ISO | `f_iso` | Free-water volume fraction |
//! | Extra-cellular fraction | f_EC | `(1−f_iso)·(1−f_intra)` | Hindered compartment |
//!
//! [Zhang et al. (2012)](https://doi.org/10.1016/j.neuroimage.2012.03.072)

use coeus_optim::{
    LeastSquaresProblem, LeastSquaresReport, LevenbergMarquardtConfig, ProblemError,
    levenberg_marquardt,
};
use ritk_diffusion_scheme::{DiffusionWeighting, GradientFrame, GradientScheme};
use std::sync::OnceLock;

// ── Biophysical constants ─────────────────────────────────────────────────────

/// Intrinsic intra-neurite parallel diffusivity (mm²/s).
const D_PARALLEL: f64 = 1.7e-3;

/// Extra-cellular mean diffusivity (mm²/s).
const D_EXTRA: f64 = 0.8e-3;

/// CSF free-water isotropic diffusivity (mm²/s).
const D_ISO: f64 = 3.0e-3;

/// Number of free parameters: `f_intra, f_iso, ODI, θ, φ`.
const PARAM_COUNT: usize = 5;

/// Parameter vector offsets.
const F_INTRA: usize = 0;
const F_ISO: usize = 1;
const ODI: usize = 2;
const THETA: usize = 3;
const PHI: usize = 4;

/// Number of quasi-uniform directions for Watson quadrature.
const N_QUAD: usize = 300;

/// Machine-epsilon-derived finite-difference step scale.
const FD_EPS: f64 = 1.4901161193847656e-8; // √(f64::EPSILON)

// ── Error ─────────────────────────────────────────────────────────────────────

/// Failure while estimating a NODDI model.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum NoddiError {
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
    /// The Levenberg-Marquardt solver could not proceed.
    #[error("Levenberg-Marquardt solver error: {0}")]
    SolverFailed(String),
    /// NODDI volume construction failed validation.
    #[error("NODDI volume validation error: {0}")]
    VolumeValidation(String),
}

// ── Configuration ─────────────────────────────────────────────────────────────

/// Validated NODDI configuration.
#[derive(Debug, Clone)]
pub struct NoddiConfig {
    b0_threshold: DiffusionWeighting,
    lm_config: LevenbergMarquardtConfig<f64>,
}

impl NoddiConfig {
    /// Construct a NODDI configuration.
    pub const fn new(
        b0_threshold: DiffusionWeighting,
        lm_config: LevenbergMarquardtConfig<f64>,
    ) -> Self {
        Self {
            b0_threshold,
            lm_config,
        }
    }

    /// Threshold separating b0 and weighted acquisitions.
    #[must_use]
    pub const fn b0_threshold(&self) -> DiffusionWeighting {
        self.b0_threshold
    }

    /// Levenberg-Marquardt tuning parameters.
    #[must_use]
    pub const fn lm_config(&self) -> &LevenbergMarquardtConfig<f64> {
        &self.lm_config
    }
}

impl Default for NoddiConfig {
    fn default() -> Self {
        Self {
            b0_threshold: DiffusionWeighting::from_seconds_per_square_millimeter(50.0)
                .expect("invariant: default b0 threshold is finite and nonnegative"),
            lm_config: LevenbergMarquardtConfig::default(),
        }
    }
}

// ── NODDI Fit ─────────────────────────────────────────────────────────────────

/// Estimated NODDI model parameters at one voxel.
#[derive(Debug, Clone)]
pub struct NoddiFit {
    f_intra: f64,
    f_iso: f64,
    odi: f64,
    principal_direction: [f64; 3],
    baseline_signal: f64,
    residual_norm: f64,
    converged: bool,
    iterations: usize,
    gradient_norm: f64,
    frame: GradientFrame,
}

impl NoddiFit {
    /// Neurite density index — fraction of the non-CSF signal attributed
    /// to intra-neurite water.  Range [0, 1].
    #[must_use]
    pub const fn ndi(&self) -> f64 {
        self.f_intra
    }

    /// Intra-cellular volume fraction (synonym for [`Self::ndi`]).
    #[must_use]
    pub const fn f_intra(&self) -> f64 {
        self.f_intra
    }

    /// CSF (isotropic free water) volume fraction.  Range [0, 1].
    #[must_use]
    pub const fn f_iso(&self) -> f64 {
        self.f_iso
    }

    /// Orientation Dispersion Index — `(2/π)·arctan(1/κ)`.
    /// 0 = perfectly aligned sticks, 1 = isotropic dispersion.
    #[must_use]
    pub const fn odi(&self) -> f64 {
        self.odi
    }

    /// Extra-cellular (hindered) volume fraction.
    /// `(1 − f_iso)·(1 − f_intra)`.  Range [0, 1].
    #[must_use]
    pub fn f_extra(&self) -> f64 {
        (1.0 - self.f_iso) * (1.0 - self.f_intra)
    }

    /// Principal fibre direction (unit vector) — the Watson mean direction.
    #[must_use]
    pub fn principal_direction(&self) -> [f64; 3] {
        self.principal_direction
    }

    /// Mean signal over b0 acquisitions.
    #[must_use]
    pub const fn baseline_signal(&self) -> f64 {
        self.baseline_signal
    }

    /// `‖S_measured − S_model‖₂` at the solution.
    #[must_use]
    pub const fn residual_norm(&self) -> f64 {
        self.residual_norm
    }

    /// Whether the Levenberg-Marquardt convergence criterion was met.
    #[must_use]
    pub const fn converged(&self) -> bool {
        self.converged
    }

    /// Levenberg-Marquardt iterations executed.
    #[must_use]
    pub const fn iterations(&self) -> usize {
        self.iterations
    }

    /// `‖Jᵀr‖_∞` at the returned parameters.
    #[must_use]
    pub const fn gradient_norm(&self) -> f64 {
        self.gradient_norm
    }

    /// Coordinate frame the direction is expressed in.
    #[must_use]
    pub const fn frame(&self) -> GradientFrame {
        self.frame
    }

    /// Predicted signal at a unit gradient direction for a given b-value
    /// using the fitted Watson NODDI model.
    #[must_use]
    pub fn predict_signal(&self, direction: [f64; 3], b_value: f64) -> f64 {
        if b_value == 0.0 {
            return self.baseline_signal;
        }
        let kappa = odi_to_kappa(self.odi);
        let a_ic = watson_stick(b_value, direction, self.principal_direction, kappa, &quadrature_sphere());
        let a_ec = (-b_value * D_EXTRA).exp();
        let a_iso = (-b_value * D_ISO).exp();
        self.baseline_signal
            * ((1.0 - self.f_iso) * (self.f_intra * a_ic + (1.0 - self.f_intra) * a_ec)
                + self.f_iso * a_iso)
    }
}

// ── Helper utilities ──────────────────────────────────────────────────────────

/// Dot product of two 3-vectors.
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Compute the fibre direction unit vector from spherical coordinates.
fn sph_to_dir(theta: f64, phi: f64) -> [f64; 3] {
    let sin_theta = theta.sin();
    [sin_theta * phi.cos(), sin_theta * phi.sin(), theta.cos()]
}

/// Convert ODI ∈ [0, 1] to Watson concentration κ.
///
/// The domain is clamped to `[ODI_MIN, ODI_MAX]` to keep κ in a range
/// where `exp(κ)` does not overflow f64 (κ ≲ 700).  At κ → ∞ the
/// Watson distribution collapses to a delta at μ, which the
/// single-stick fallback in [`watson_stick`] handles exactly.
fn odi_to_kappa(odi: f64) -> f64 {
    const ODI_MIN: f64 = 0.005;
    const ODI_MAX: f64 = 0.995;
    let clamped = odi.clamp(ODI_MIN, ODI_MAX);
    1.0 / (std::f64::consts::FRAC_PI_2 * clamped).tan()
}

/// Generate N quasi-uniform directions on the sphere (Fibonacci lattice).
fn fibonacci_sphere(n: usize) -> Vec<[f64; 3]> {
    let phi_golden = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    let mut points = Vec::with_capacity(n);
    for i in 0..n {
        let z = 1.0 - (2.0 * i as f64 + 1.0) / n as f64;
        let radius = (1.0 - z * z).sqrt();
        let theta = phi_golden * i as f64;
        points.push([radius * theta.cos(), radius * theta.sin(), z]);
    }
    points
}

/// Lazy-initialized Fibonacci sphere quadrature points.
fn quadrature_sphere() -> &'static [[f64; 3]] {
    static QUAD: OnceLock<Box<[[f64; 3]]>> = OnceLock::new();
    QUAD.get_or_init(|| {
        let pts = fibonacci_sphere(N_QUAD);
        pts.into_boxed_slice()
    })
}

/// Evaluate the Watson-averaged stick signal at gradient direction `g`
/// by self-normalized Monte Carlo quadrature over the sphere.
///
/// Returns `(1/Z) · Σᵢ exp(κ·(μ·vᵢ)²) · exp(−b·d_‖·(g·vᵢ)²)` where
/// `Z = Σᵢ exp(κ·(μ·vᵢ)²)` is the empirical partition function.
fn watson_stick(
    b: f64,
    g: [f64; 3],
    mu: [f64; 3],
    kappa: f64,
    quad_points: &[[f64; 3]],
) -> f64 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;

    for &v in quad_points {
        let w = (kappa * dot3(mu, v).powi(2)).exp();
        den += w;
        num += w * (-b * D_PARALLEL * dot3(g, v).powi(2)).exp();
    }

    if den < 1e-30 {
        // κ → ∞: all weight on v ≈ μ, so stick → exp(−b·d_‖·(g·μ)²).
        return (-b * D_PARALLEL * dot3(g, mu).powi(2)).exp();
    }
    num / den
}

// ── Finite-difference Jacobian helpers ────────────────────────────────────────

/// Compute a sensible finite-difference step for parameter `p`.
fn fd_step(p: f64) -> f64 {
    p.abs().max(1.0) * FD_EPS
}

// ── Watson NODDI Problem for Levenberg-Marquardt ──────────────────────────────

/// The Watson NODDI model wrapped as a [`LeastSquaresProblem<f64>`].
///
/// Parameters: `[f_intra, f_iso, ODI, θ, φ]`.  Each residual is
/// `S_i − S_model(b_i, g_i; p)`.
struct WatsonNoddiProblem {
    b_values: Vec<f64>,
    directions: Vec<[f64; 3]>,
    signals: Vec<f64>,
    baseline: f64,
}

impl LeastSquaresProblem<f64> for WatsonNoddiProblem {
    fn residual_count(&self) -> usize {
        self.signals.len()
    }

    fn parameter_count(&self) -> usize {
        PARAM_COUNT
    }

    fn residuals(
        &self,
        parameters: &[f64],
        residuals: &mut [f64],
    ) -> Result<(), ProblemError> {
        let f_intra = parameters[F_INTRA];
        let f_iso = parameters[F_ISO];
        let odi = parameters[ODI];
        let dir = sph_to_dir(parameters[THETA], parameters[PHI]);
        let kappa = odi_to_kappa(odi);
        let quad = quadrature_sphere();

        for (i, slot) in residuals.iter_mut().enumerate() {
            let b = self.b_values[i];
            let a_ic = watson_stick(b, self.directions[i], dir, kappa, quad);
            let a_ec = (-b * D_EXTRA).exp();
            let a_iso = (-b * D_ISO).exp();

            let predicted = self.baseline
                * ((1.0 - f_iso) * (f_intra * a_ic + (1.0 - f_intra) * a_ec)
                    + f_iso * a_iso);
            *slot = self.signals[i] - predicted;
        }
        Ok(())
    }

    fn jacobian(
        &self,
        parameters: &[f64],
        jacobian: &mut [f64],
    ) -> Result<(), ProblemError> {
        let f_intra = parameters[F_INTRA];
        let f_iso = parameters[F_ISO];
        let odi = parameters[ODI];
        let theta = parameters[THETA];
        let phi = parameters[PHI];
        let dir = sph_to_dir(theta, phi);
        let kappa = odi_to_kappa(odi);
        let quad = quadrature_sphere();

        // ── Analytic columns: f_intra, f_iso ──────────────────────────────
        for (i, (&b, &g)) in self.b_values.iter().zip(self.directions.iter()).enumerate() {
            let a_ic = watson_stick(b, g, dir, kappa, quad);
            let a_ec = (-b * D_EXTRA).exp();
            let a_iso = (-b * D_ISO).exp();
            let s0 = self.baseline;
            let base = i * PARAM_COUNT;

            // ∂r/∂f_intra = −S₀·(1 − f_iso)·(A_ic − A_ec)
            jacobian[base + F_INTRA] = -s0 * (1.0 - f_iso) * (a_ic - a_ec);

            // ∂r/∂f_iso = S₀·(f_intra·A_ic + (1−f_intra)·A_ec − A_iso)
            jacobian[base + F_ISO] =
                s0 * (f_intra * a_ic + (1.0 - f_intra) * a_ec - a_iso);
        }

        // ── Finite-difference columns: ODI, θ, φ ──────────────────────────
        let mut r_plus = vec![0.0; self.signals.len()];
        let mut r_minus = vec![0.0; self.signals.len()];

        // ODI
        let h = fd_step(odi);
        let mut p_plus = parameters.to_vec();
        p_plus[ODI] = (odi + h).clamp(0.0, 1.0);
        let mut p_minus = parameters.to_vec();
        p_minus[ODI] = (odi - h).clamp(0.0, 1.0);
        self.residuals(&p_plus, &mut r_plus)?;
        self.residuals(&p_minus, &mut r_minus)?;
        let h_eff = p_plus[ODI] - p_minus[ODI];
        for i in 0..self.signals.len() {
            jacobian[i * PARAM_COUNT + ODI] =
                if h_eff > 1e-30 { (r_plus[i] - r_minus[i]) / h_eff } else { 0.0 };
        }

        // θ
        let h = fd_step(theta);
        let mut p_plus = parameters.to_vec();
        p_plus[THETA] = theta + h;
        let mut p_minus = parameters.to_vec();
        p_minus[THETA] = theta - h;
        self.residuals(&p_plus, &mut r_plus)?;
        self.residuals(&p_minus, &mut r_minus)?;
        for i in 0..self.signals.len() {
            jacobian[i * PARAM_COUNT + THETA] = (r_plus[i] - r_minus[i]) / (2.0 * h);
        }

        // φ
        let h = fd_step(phi);
        let mut p_plus = parameters.to_vec();
        p_plus[PHI] = phi + h;
        let mut p_minus = parameters.to_vec();
        p_minus[PHI] = phi - h;
        self.residuals(&p_plus, &mut r_plus)?;
        self.residuals(&p_minus, &mut r_minus)?;
        for i in 0..self.signals.len() {
            jacobian[i * PARAM_COUNT + PHI] = (r_plus[i] - r_minus[i]) / (2.0 * h);
        }

        Ok(())
    }
}

// ── Estimation ────────────────────────────────────────────────────────────────

/// Estimate NODDI parameters from one voxel's signals via
/// Levenberg-Marquardt nonlinear fitting.
///
/// The intra-cellular signal is evaluated by Monte Carlo quadrature over
/// 300 quasi-uniform directions on the sphere, with self-normalizing
/// Watson weights that avoid evaluating the confluent hypergeometric
/// function ₁F₁.
///
/// Initial guess: `f_intra = 0.5`, `f_iso = 0.0`, `ODI = 0.03`
/// (near-perfect alignment), and the principal direction from the DTI
/// PEV of a preliminary fit.
///
/// # Errors
///
/// Returns a typed error for count mismatch, non-finite signals, missing
/// b0 or weighted samples, an invalid baseline, or a solver failure.
pub fn estimate_noddi(
    scheme: &GradientScheme,
    signals: &[f64],
    config: &NoddiConfig,
) -> Result<NoddiFit, NoddiError> {
    // ── Validation ────────────────────────────────────────────────────────
    if signals.len() != scheme.len() {
        return Err(NoddiError::SignalLengthMismatch {
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
        return Err(NoddiError::NonFiniteSignal { index, value });
    }

    let b0_indices = scheme.b0_indices(config.b0_threshold());
    let dwi_indices = scheme.dwi_indices(config.b0_threshold());
    if b0_indices.is_empty() {
        return Err(NoddiError::NoB0Volumes);
    }
    if dwi_indices.is_empty() {
        return Err(NoddiError::NoDwiDirections);
    }

    let baseline_signal =
        b0_indices.iter().map(|index| signals[*index]).sum::<f64>() / b0_indices.len() as f64;
    if !baseline_signal.is_finite() || baseline_signal <= 0.0 {
        return Err(NoddiError::InvalidBaseline {
            value: baseline_signal,
        });
    }

    // ── Initial guess from DTI ────────────────────────────────────────────
    let dti_tensor =
        crate::dti::estimate_dti(scheme, signals, crate::dti::DtiConfig::new(config.b0_threshold()))
            .map_err(|e| NoddiError::SolverFailed(format!("DTI initial fit failed: {e}")))?;
    let pev = dti_tensor.principal_eigenvector();
    let theta_init = pev[2].clamp(-1.0, 1.0).acos();
    let phi_init = pev[1].atan2(pev[0]);

    // ── Collect DWI acquisitions ──────────────────────────────────────────
    let n_dwi = dwi_indices.len();
    let mut b_values = Vec::with_capacity(n_dwi);
    let mut directions = Vec::with_capacity(n_dwi);
    let mut dwi_signals = Vec::with_capacity(n_dwi);

    for &global_index in &dwi_indices {
        let entry = &scheme.directions()[global_index];
        let b = entry.weighting().seconds_per_square_millimeter();
        b_values.push(b);
        directions.push(entry.direction().to_array());
        dwi_signals.push(signals[global_index]);
    }

    // ── Levenberg-Marquardt ───────────────────────────────────────────────
    // [f_intra, f_iso, ODI, θ, φ]
    let initial = vec![0.5, 0.0, 0.03, theta_init, phi_init];

    let problem = WatsonNoddiProblem {
        b_values,
        directions,
        signals: dwi_signals,
        baseline: baseline_signal,
    };

    let report: LeastSquaresReport<f64> = levenberg_marquardt(
        &problem,
        &initial,
        config.lm_config(),
    )
    .map_err(|error| NoddiError::SolverFailed(error.to_string()))?;

    // ── Post-process ─────────────────────────────────────────────────────
    let f_intra = report.parameters[F_INTRA].clamp(0.0, 1.0);
    let f_iso = report.parameters[F_ISO].clamp(0.0, 1.0);
    let odi = report.parameters[ODI].clamp(0.0, 1.0);
    let theta = report.parameters[THETA];
    let phi = report.parameters[PHI];
    let principal_direction = sph_to_dir(theta, phi);

    Ok(NoddiFit {
        f_intra,
        f_iso,
        odi,
        principal_direction,
        baseline_signal,
        residual_norm: (2.0 * report.cost).sqrt(),
        converged: report.termination.is_converged(),
        iterations: report.iterations,
        gradient_norm: report.gradient_norm,
        frame: scheme.frame(),
    })
}

#[cfg(test)]
mod tests;

// ── NODDI Volume (whole-brain tractography) ───────────────────────────────────

/// A 3-D volume of NODDI principal directions on a regular grid.
///
/// Stores one unit direction vector per voxel in z-major (slice-first)
/// order.  Supports nearest-neighbour spatial lookup for sub-voxel
/// direction queries during whole-brain NODDI-based tractography via
/// [`NoddiVolume::direction_at`].
///
/// Unlike [`super::csd::FodVolume`], which stores raw coefficients and
/// performs expensive peak-extraction at query time, the NODDI volume
/// stores the fitted principal direction directly — the NODDI model
/// intrinsically yields a single fibre orientation per voxel.
#[derive(Debug, Clone)]
pub struct NoddiVolume {
    /// Flat direction array: `[z][y][x][component]` where component ∈ {0,1,2}.
    directions: Box<[f64]>,
    /// Grid dimensions `[nx, ny, nz]`.
    shape: [usize; 3],
    /// Voxel size in physical units (mm), `[sx, sy, sz]`.
    spacing: [f64; 3],
    /// Physical position of the first voxel centre `[ox, oy, oz]`.
    origin: [f64; 3],
    /// Coordinate frame for direction queries.
    frame: GradientFrame,
}

impl NoddiVolume {
    /// Construct a volume from a flat direction array.
    ///
    /// `directions` must have exactly `nx × ny × nz × 3` elements.
    /// Spacing must be finite and positive; origin must be finite.
    ///
    /// # Errors
    ///
    /// Returns [`NoddiError::VolumeValidation`] for validation failures.
    pub fn new(
        directions: Box<[f64]>,
        shape: [usize; 3],
        spacing: [f64; 3],
        origin: [f64; 3],
        frame: GradientFrame,
    ) -> Result<Self, NoddiError> {
        let [nx, ny, nz] = shape;
        if nx == 0 || ny == 0 || nz == 0 {
            return Err(NoddiError::VolumeValidation(format!(
                "shape must be nonzero, got [{nx}, {ny}, {nz}]"
            )));
        }
        let expected = nx
            .checked_mul(ny)
            .and_then(|v| v.checked_mul(nz))
            .and_then(|v| v.checked_mul(3))
            .ok_or_else(|| NoddiError::VolumeValidation("element count overflow".into()))?;
        if directions.len() != expected {
            return Err(NoddiError::VolumeValidation(format!(
                "expected {} direction components for shape {nx}×{ny}×{nz}, got {}",
                expected,
                directions.len()
            )));
        }
        let [sx, sy, sz] = spacing;
        if !sx.is_finite() || sx <= 0.0 || !sy.is_finite() || sy <= 0.0 || !sz.is_finite() || sz <= 0.0 {
            return Err(NoddiError::VolumeValidation(format!(
                "spacing must be finite and positive, got [{sx}, {sy}, {sz}]"
            )));
        }
        let [ox, oy, oz] = origin;
        if !ox.is_finite() || !oy.is_finite() || !oz.is_finite() {
            return Err(NoddiError::VolumeValidation(format!(
                "origin must be finite, got [{ox}, {oy}, {oz}]"
            )));
        }
        Ok(Self {
            directions,
            shape,
            spacing,
            origin,
            frame,
        })
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

    /// Nearest-neighbour lookup of the NODDI principal direction at a
    /// physical point.
    ///
    /// Returns `None` when the point maps to a voxel index outside the
    /// grid or when any coordinate is non-finite.
    pub fn direction_at(&self, point: &ritk_spatial::Point<3>) -> Option<ritk_spatial::Vector<3>> {
        let [px, py, pz] = point.to_array();
        if !px.is_finite() || !py.is_finite() || !pz.is_finite() {
            return None;
        }
        let [ox, oy, oz] = self.origin;
        let [sx, sy, sz] = self.spacing;
        let [nx, ny, nz] = self.shape;

        let ix = ((px - ox) / sx).round() as isize;
        let iy = ((py - oy) / sy).round() as isize;
        let iz = ((pz - oz) / sz).round() as isize;

        if ix < 0 || ix >= nx as isize || iy < 0 || iy >= ny as isize || iz < 0 || iz >= nz as isize {
            return None;
        }

        let base = (iz as usize * ny * nx + iy as usize * nx + ix as usize) * 3;
        let dir = [
            self.directions[base],
            self.directions[base + 1],
            self.directions[base + 2],
        ];
        // The direction should be unit from the fit, but guard against
        // degenerate (zero-length) vectors from failed fits.
        let norm_sq = dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2];
        if norm_sq < 1e-30 {
            return None;
        }
        let inv_norm = 1.0 / norm_sq.sqrt();
        Some(ritk_spatial::Vector::new([
            dir[0] * inv_norm,
            dir[1] * inv_norm,
            dir[2] * inv_norm,
        ]))
    }
}
