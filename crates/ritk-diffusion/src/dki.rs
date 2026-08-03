//! Diffusion kurtosis imaging (DKI) — nonlinear kurtosis tensor estimation.
//!
//! DKI extends DTI by adding the kurtosis tensor `W`, a fourth-order symmetric
//! tensor that captures non-Gaussian water diffusion (Jensen et al., 2005).
//! The normalised signal `S/S₀` at gradient direction `g` and b-value `b` is
//!
//! ```text
//! S/S₀ = exp(−b · D(g) + (b²/6) · MD² · W(g))
//! ```
//!
//! where `D(g) = gᵀDg` is the apparent diffusion coefficient,
//! `MD = tr(D)/3` is the mean diffusivity, and `W(g)` is the full contraction
//! of the kurtosis tensor with the gradient direction:
//!
//! ```text
//! W(g) = Σ W_{ijkl} · g_i · g_j · g_k · g_l
//! ```
//!
//! Taking logs gives a nonlinear model that separates D and W: the Gaussian
//! term scales with `b` while the kurtosis term scales with `b²`.  RITK fits
//! this model via Levenberg-Marquardt (damped Gauss-Newton) through
//! [`coeus_optim::levenberg_marquardt`], with an initial D guess supplied by
//! the DTI log-linear fit.
//!
//! # Kurtosis metrics
//!
//! | Metric | Notation | Meaning |
//! |--------|----------|---------|
//! | Mean kurtosis | MK | `⟨K(g)⟩` over the sphere — overall deviation from Gaussian |
//! | Axial kurtosis | AK | `K(e₁)` along the principal diffusion direction |
//! | Radial kurtosis | RK | `⟨K(g)⟩` over directions perpendicular to `e₁` |
//!
//! # Relation to DTI
//!
//! DTI is the `W = 0` special case of DKI.  The kurtosis term becomes
//! significant at b-values above ≈ 1500 s/mm², where the Gaussian
//! approximation breaks down.
//!
//! [Jensen et al. (2005)](https://doi.org/10.1002/mrm.20508)

use coeus_optim::{
    LeastSquaresProblem, LeastSquaresReport, LevenbergMarquardtConfig, ProblemError,
    levenberg_marquardt,
};
use ritk_diffusion_scheme::{DiffusionWeighting, GradientFrame, GradientScheme};

use crate::dti::{self, DtiConfig, DtiError};

// ── Error ─────────────────────────────────────────────────────────────────────

/// Failure while estimating a diffusion kurtosis tensor.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum KtiError {
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
    /// A normalised DWI signal is not finite and positive for the log domain.
    #[error(
        "normalised signal at acquisition index {index} must be finite and positive, got {value}"
    )]
    InvalidNormalisedSignal {
        /// Acquisition-order index.
        index: usize,
        /// Invalid normalised value.
        value: f64,
    },
    /// The initial DTI fit failed.
    #[error("initial DTI fit failed: {0}")]
    DtiFailed(#[from] DtiError),
    /// The Levenberg-Marquardt solver could not proceed.
    #[error("Levenberg-Marquardt solver error: {0}")]
    SolverFailed(String),
}

// ── Multiplicity table for the 15-element kurtosis tensor ─────────────────────

/// The 15 independent elements of a fully symmetric 4th-order 3-D tensor,
/// in the order they appear in the parameter vector.
const W_ELEMENTS: [WElement; 15] = [
    // Type-iiii (multiplicity 1): W_iiii · g_i⁴
    WElement { multiplicity: 1, powers: [4, 0, 0] },
    WElement { multiplicity: 1, powers: [0, 4, 0] },
    WElement { multiplicity: 1, powers: [0, 0, 4] },
    // Type-iiij (multiplicity 4): W_iiij · 4·g_i³·g_j
    WElement { multiplicity: 4, powers: [3, 1, 0] },
    WElement { multiplicity: 4, powers: [3, 0, 1] },
    WElement { multiplicity: 4, powers: [1, 3, 0] },
    WElement { multiplicity: 4, powers: [0, 3, 1] },
    WElement { multiplicity: 4, powers: [1, 0, 3] },
    WElement { multiplicity: 4, powers: [0, 1, 3] },
    // Type-iijj (multiplicity 6): W_iijj · 6·g_i²·g_j²
    WElement { multiplicity: 6, powers: [2, 2, 0] },
    WElement { multiplicity: 6, powers: [2, 0, 2] },
    WElement { multiplicity: 6, powers: [0, 2, 2] },
    // Type-iijk (multiplicity 12): W_iijk · 12·g_i²·g_j·g_k
    WElement { multiplicity: 12, powers: [2, 1, 1] },
    WElement { multiplicity: 12, powers: [1, 2, 1] },
    WElement { multiplicity: 12, powers: [1, 1, 2] },
];

/// One element of the kurtosis tensor, with its multiplicity and monomial powers.
#[derive(Debug, Clone, Copy)]
struct WElement {
    /// Multiplicity of this element in the sum Σ W_ijkl g_i g_j g_k g_l.
    multiplicity: u32,
    /// Powers [px, py, pz] such that the monomial is g_x^px · g_y^py · g_z^pz.
    powers: [u32; 3],
}

impl WElement {
    /// Evaluate the monomial `m · g_x^px · g_y^py · g_z^pz`.
    fn monomial(&self, gx: f64, gy: f64, gz: f64) -> f64 {
        self.multiplicity as f64
            * gx.powi(self.powers[0] as i32)
            * gy.powi(self.powers[1] as i32)
            * gz.powi(self.powers[2] as i32)
    }
}

/// Number of free parameters: 6 D elements + 15 W elements.
const PARAMETER_COUNT: usize = 21;

/// Parameter indices.
const D_OFFSET: usize = 0;
const W_OFFSET: usize = 6;

// ── Configuration ─────────────────────────────────────────────────────────────

/// Validated DKI configuration.
#[derive(Debug, Clone)]
pub struct KtiConfig {
    b0_threshold: DiffusionWeighting,
    lm_config: LevenbergMarquardtConfig<f64>,
}

impl KtiConfig {
    /// Construct a DKI configuration.
    ///
    /// `b0_threshold` classifies reference (≤ threshold) and weighted
    /// volumes.  `lm_config` tunes the Levenberg-Marquardt solver; the
    /// default tolerances (`√ε`) are appropriate for noise-free synthetic
    /// data and should be relaxed for in-vivo fitting.
    pub const fn new(b0_threshold: DiffusionWeighting, lm_config: LevenbergMarquardtConfig<f64>) -> Self {
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

impl Default for KtiConfig {
    fn default() -> Self {
        Self {
            b0_threshold: DiffusionWeighting::from_seconds_per_square_millimeter(50.0)
                .expect("invariant: default b0 threshold is finite and nonnegative"),
            lm_config: LevenbergMarquardtConfig::default(),
        }
    }
}

// ── Diffusion Kurtosis Tensor ─────────────────────────────────────────────────

/// Estimated diffusion and kurtosis tensors at one voxel.
///
/// The six D elements `[Dₓₓ, D_yy, D_zz, Dₓy, Dₓz, D_yz]` are in mm²/s
/// (Voigt notation).  The fifteen W elements are dimensionless when
/// multiplied by MD², and are stored in the canonical order:
///
/// `[W_xxxx, W_yyyy, W_zzzz, W_xxxy, W_xxxz, W_xyyy, W_yyyz, W_xzzz,
///   W_yzzz, W_xxyy, W_xxzz, W_yyzz, W_xxyz, W_xyyz, W_xyxz]`
#[derive(Debug, Clone)]
pub struct DiffusionKurtosisTensor {
    elements_d: [f64; 6],
    elements_w: [f64; 15],
    eigenvalues: [f64; 3],
    principal_eigenvector: [f64; 3],
    mk: f64,
    ak: f64,
    rk: f64,
    baseline_signal: f64,
    residual_norm: f64,
    converged: bool,
    iterations: usize,
    gradient_norm: f64,
    frame: GradientFrame,
}

impl DiffusionKurtosisTensor {
    /// Six unique D elements `[Dₓₓ, D_yy, D_zz, Dₓy, Dₓz, D_yz]` in mm²/s.
    #[must_use]
    pub fn elements_d(&self) -> &[f64; 6] {
        &self.elements_d
    }

    /// Fifteen independent kurtosis tensor elements.
    #[must_use]
    pub fn elements_w(&self) -> &[f64; 15] {
        &self.elements_w
    }

    /// Three eigenvalues `λ₀ ≥ λ₁ ≥ λ₂` of D in mm²/s.
    #[must_use]
    pub fn eigenvalues(&self) -> &[f64; 3] {
        &self.eigenvalues
    }

    /// Fractional anisotropy ∈ [0, 1] of the diffusion tensor.
    #[must_use]
    pub fn fa(&self) -> f64 {
        let [l0, l1, l2] = self.eigenvalues;
        let md = (l0 + l1 + l2) / 3.0;
        let numerator = ((l0 - md).powi(2) + (l1 - md).powi(2) + (l2 - md).powi(2)).sqrt();
        let denominator = (l0.powi(2) + l1.powi(2) + l2.powi(2)).sqrt();
        if denominator < 1e-15 {
            return 0.0;
        }
        (1.5_f64).sqrt() * numerator / denominator
    }

    /// Mean diffusivity in mm²/s.
    #[must_use]
    pub fn md(&self) -> f64 {
        (self.elements_d[0] + self.elements_d[1] + self.elements_d[2]) / 3.0
    }

    /// Mean kurtosis — the average of `K(g)` over the sphere.
    ///
    /// Computed by numerical integration over 200 quasi-uniform directions.
    /// Zero for purely Gaussian diffusion; positive for platykurtic
    /// (restricted) diffusion.
    #[must_use]
    pub fn mk(&self) -> f64 {
        self.mk
    }

    /// Axial kurtosis — `K(e₁)` along the principal diffusion direction.
    #[must_use]
    pub fn ak(&self) -> f64 {
        self.ak
    }

    /// Radial kurtosis — average of `K(g)` over directions perpendicular
    /// to the principal eigenvector.
    #[must_use]
    pub fn rk(&self) -> f64 {
        self.rk
    }

    /// Principal eigenvector `∥PEV∥ = 1` of D.
    #[must_use]
    pub fn principal_eigenvector(&self) -> [f64; 3] {
        self.principal_eigenvector
    }

    /// Mean signal over b0 acquisitions.
    #[must_use]
    pub const fn baseline_signal(&self) -> f64 {
        self.baseline_signal
    }

    /// `0.5‖r‖²` after the Levenberg-Marquardt solve.
    #[must_use]
    pub const fn residual_norm(&self) -> f64 {
        self.residual_norm
    }

    /// Whether a derived convergence criterion (gradient, step, or cost)
    /// was met.
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

    /// Coordinate frame the tensor axes are expressed in.
    #[must_use]
    pub const fn frame(&self) -> GradientFrame {
        self.frame
    }

    /// Predicted signal at a unit gradient direction for a given b-value
    /// using the full DKI model.
    #[must_use]
    pub fn predict_signal(&self, direction: [f64; 3], b_value: f64) -> f64 {
        let d_app = self.quadratic_form(direction);
        let md = self.md();
        let w_app = compute_w_contraction(&self.elements_w, direction);
        self.baseline_signal * (-b_value * d_app + (b_value.powi(2) / 6.0) * md.powi(2) * w_app).exp()
    }

    /// Quadratic form `gᵀ D g` for a unit direction `g`.
    #[must_use]
    pub fn quadratic_form(&self, direction: [f64; 3]) -> f64 {
        let [dxx, dyy, dzz, dxy, dxz, dyz] = self.elements_d;
        let [gx, gy, gz] = direction;
        dxx * gx * gx
            + dyy * gy * gy
            + dzz * gz * gz
            + 2.0 * dxy * gx * gy
            + 2.0 * dxz * gx * gz
            + 2.0 * dyz * gy * gz
    }

    /// Apparent kurtosis coefficient `K(g)` for a unit direction `g`.
    ///
    /// ```text
    /// K(g) = (MD / D(g))² · W(g)
    /// ```
    #[must_use]
    pub fn kurtosis_at_direction(&self, direction: [f64; 3]) -> f64 {
        let d_app = self.quadratic_form(direction);
        let md = self.md();
        if d_app < 1e-15 {
            return 0.0;
        }
        let w_app = compute_w_contraction(&self.elements_w, direction);
        (md / d_app).powi(2) * w_app
    }
}

// ── Full contraction of the kurtosis tensor ───────────────────────────────────

/// `W(g) = Σ W_{ijkl} · g_i · g_j · g_k · g_l` for a unit direction.
fn compute_w_contraction(w: &[f64; 15], g: [f64; 3]) -> f64 {
    let [gx, gy, gz] = g;
    let mut sum = 0.0;
    for (idx, element) in W_ELEMENTS.iter().enumerate() {
        sum += w[idx] * element.monomial(gx, gy, gz);
    }
    sum
}

// ── DKI Problem for Levenberg-Marquardt ───────────────────────────────────────

/// The DKI model wrapped as a [`LeastSquaresProblem<f64>`].
///
/// Parameters are `[Dxx, ..., Dyz, W_xxxx, ..., W_xyxz]` (21 elements).
/// Each residual is `ln(S_i/S₀) + b_i·D(g_i) − (b_i²/6)·MD²·W(g_i)`.
struct DkiProblem {
    /// Cached b-values for each DWI acquisition.
    b_values: Vec<f64>,
    /// Cached gradient directions [gx, gy, gz] for each DWI acquisition.
    directions: Vec<[f64; 3]>,
    /// Normalised log-signals `ln(S_i/S₀)` for each DWI acquisition.
    log_signals: Vec<f64>,
}

impl LeastSquaresProblem<f64> for DkiProblem {
    fn residual_count(&self) -> usize {
        self.log_signals.len()
    }

    fn parameter_count(&self) -> usize {
        PARAMETER_COUNT
    }

    fn residuals(
        &self,
        parameters: &[f64],
        residuals: &mut [f64],
    ) -> Result<(), ProblemError> {
        let d = &parameters[D_OFFSET..D_OFFSET + 6];
        let w = &parameters[W_OFFSET..W_OFFSET + 15];
        let md = (d[0] + d[1] + d[2]) / 3.0;

        for (i, slot) in residuals.iter_mut().enumerate() {
            let b = self.b_values[i];
            let [gx, gy, gz] = self.directions[i];

            let d_app = d[0] * gx * gx
                + d[1] * gy * gy
                + d[2] * gz * gz
                + 2.0 * d[3] * gx * gy
                + 2.0 * d[4] * gx * gz
                + 2.0 * d[5] * gy * gz;

            let w_app = compute_w_contraction_slice(w, gx, gy, gz);

            let predicted = -b * d_app + (b.powi(2) / 6.0) * md.powi(2) * w_app;
            *slot = self.log_signals[i] - predicted;
        }
        Ok(())
    }

    fn jacobian(
        &self,
        parameters: &[f64],
        jacobian: &mut [f64],
    ) -> Result<(), ProblemError> {
        let d = &parameters[D_OFFSET..D_OFFSET + 6];
        let w = &parameters[W_OFFSET..W_OFFSET + 15];
        let md = (d[0] + d[1] + d[2]) / 3.0;
        let md_sq = md * md;

        for (i, (b, [gx, gy, gz])) in self
            .b_values
            .iter()
            .copied()
            .zip(self.directions.iter().copied())
            .enumerate()
        {            let w_app = compute_w_contraction_slice(w, gx, gy, gz);

            let base = i * PARAMETER_COUNT;

            // ── D derivatives ──────────────────────────────────────────────
            // ∂r/∂Dkk = b·g_k² − (b²/9)·MD·W(g)   (MD derivative via chain rule)
            // ∂r/∂Dkl = 2·b·g_k·g_l                (no MD dependence for off-diag)
            let md_term = -(b.powi(2) / 9.0) * md * w_app;

            jacobian[base] = b * gx * gx + md_term; // Dxx
            jacobian[base + 1] = b * gy * gy + md_term; // Dyy
            jacobian[base + 2] = b * gz * gz + md_term; // Dzz
            jacobian[base + 3] = 2.0 * b * gx * gy; // Dxy
            jacobian[base + 4] = 2.0 * b * gx * gz; // Dxz
            jacobian[base + 5] = 2.0 * b * gy * gz; // Dyz

            // ── W derivatives ──────────────────────────────────────────────
            // ∂r/∂W_j = −(b²/6)·MD²·m_j·(g-power monomial)
            let factor = -(b.powi(2) / 6.0) * md_sq;
            for (j, element) in W_ELEMENTS.iter().enumerate() {
                jacobian[base + W_OFFSET + j] =
                    factor * element.monomial(gx, gy, gz);
            }
        }
        Ok(())
    }
}

/// Slice variant of [`compute_w_contraction`] for use inside the trait impl.
fn compute_w_contraction_slice(w: &[f64], gx: f64, gy: f64, gz: f64) -> f64 {
    let mut sum = 0.0;
    for (idx, element) in W_ELEMENTS.iter().enumerate() {
        sum += w[idx] * element.monomial(gx, gy, gz);
    }
    sum
}

// ── Estimation ────────────────────────────────────────────────────────────────

/// Estimate a diffusion kurtosis tensor from one voxel's signals via
/// Levenberg-Marquardt nonlinear fitting.
///
/// The estimation proceeds in two stages:
///
/// 1. A log-linear DTI fit supplies the initial D guess and baseline
///    signal `S₀`.
/// 2. Levenberg-Marquardt refines D and fits the kurtosis tensor W
///    simultaneously.
///
/// The solver requires at least 21 DWI directions with non-zero b-values
/// distributed across shells (a single-shell scheme cannot disambiguate
/// the quadratic D term from the quartic W term).
///
/// # Errors
///
/// Returns a typed error for count mismatch, non-finite signals, missing
/// b0 or weighted samples, an invalid baseline, a failed DTI initial fit,
/// or a solver failure (underdetermined, singular, non-finite, or
/// unrecoverable).
pub fn estimate_dki(
    scheme: &GradientScheme,
    signals: &[f64],
    config: &KtiConfig,
) -> Result<DiffusionKurtosisTensor, KtiError> {
    // ── Validation ────────────────────────────────────────────────────────
    if signals.len() != scheme.len() {
        return Err(KtiError::SignalLengthMismatch {
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
        return Err(KtiError::NonFiniteSignal { index, value });
    }

    let b0_indices = scheme.b0_indices(config.b0_threshold());
    let dwi_indices = scheme.dwi_indices(config.b0_threshold());
    if b0_indices.is_empty() {
        return Err(KtiError::NoB0Volumes);
    }
    if dwi_indices.is_empty() {
        return Err(KtiError::NoDwiDirections);
    }

    let baseline_signal =
        b0_indices.iter().map(|index| signals[*index]).sum::<f64>() / b0_indices.len() as f64;
    if !baseline_signal.is_finite() || baseline_signal <= 0.0 {
        return Err(KtiError::InvalidBaseline {
            value: baseline_signal,
        });
    }

    // ── Stage 1: DTI initial guess ────────────────────────────────────────
    let dti_tensor = dti::estimate_dti(scheme, signals, DtiConfig::new(config.b0_threshold()))?;

    // ── Stage 2: Levenberg-Marquardt ──────────────────────────────────────
    // Collect DWI acquisitions: b-values, directions, and log-normalised signals.
    let n_dwi = dwi_indices.len();
    let mut b_values = Vec::with_capacity(n_dwi);
    let mut directions = Vec::with_capacity(n_dwi);
    let mut log_signals = Vec::with_capacity(n_dwi);

    for &global_index in &dwi_indices {
        let entry = &scheme.directions()[global_index];
        let b = entry.weighting().seconds_per_square_millimeter();
        let signal = signals[global_index];
        let normalised = signal / baseline_signal;
        if !normalised.is_finite() || normalised <= 0.0 {
            return Err(KtiError::InvalidNormalisedSignal {
                index: global_index,
                value: normalised,
            });
        }
        b_values.push(b);
        directions.push(entry.direction().to_array());
        log_signals.push(normalised.ln());
    }

    // Initial parameters: D from DTI, W = 0.
    let mut initial = vec![0.0; PARAMETER_COUNT];
    initial[..6].copy_from_slice(dti_tensor.elements());

    let problem = DkiProblem {
        b_values,
        directions,
        log_signals,
    };

    let report: LeastSquaresReport<f64> = levenberg_marquardt(
        &problem,
        &initial,
        config.lm_config(),
    )
    .map_err(|error| KtiError::SolverFailed(error.to_string()))?;

    // ── Post-process ─────────────────────────────────────────────────────
    let elements_d = [
        report.parameters[0],
        report.parameters[1],
        report.parameters[2],
        report.parameters[3],
        report.parameters[4],
        report.parameters[5],
    ];
    let elements_w = [
        report.parameters[6],  report.parameters[7],  report.parameters[8],
        report.parameters[9],  report.parameters[10], report.parameters[11],
        report.parameters[12], report.parameters[13], report.parameters[14],
        report.parameters[15], report.parameters[16], report.parameters[17],
        report.parameters[18], report.parameters[19], report.parameters[20],
    ];

    // Use DTI's eigendecomposition for the final D tensor (the LM refinement
    // is typically small and using the DTI decomposition is safe).
    let (eigenvalues, principal_eigenvector) =
        dti::decompose_3x3_symmetric_infallible(elements_d);

    let md = (elements_d[0] + elements_d[1] + elements_d[2]) / 3.0;
    let mk = compute_mk(&elements_d, &elements_w, md);
    let ak = compute_ak(&elements_w, md, principal_eigenvector, eigenvalues[0]);
    let rk = compute_rk(&elements_d, &elements_w, md, principal_eigenvector);

    Ok(DiffusionKurtosisTensor {
        elements_d,
        elements_w,
        eigenvalues,
        principal_eigenvector,
        mk,
        ak,
        rk,
        baseline_signal,
        residual_norm: (2.0 * report.cost).sqrt(),
        converged: report.termination.is_converged(),
        iterations: report.iterations,
        gradient_norm: report.gradient_norm,
        frame: scheme.frame(),
    })
}

// ── Kurtosis metrics ──────────────────────────────────────────────────────────

/// Number of quasi-uniform directions for numerical sphere integration.
const N_SPHERE_DIRS: usize = 200;

/// Compute mean kurtosis `MK = ⟨K(g)⟩` over the sphere via numerical
/// integration over `N_SPHERE_DIRS` quasi-uniform directions.
///
/// `K(g) = (MD / D(g))² · W(g)` where `D(g) = gᵀDg`.
fn compute_mk(d: &[f64; 6], w: &[f64; 15], md: f64) -> f64 {
    if md < 1e-15 {
        return 0.0;
    }
    let mut sum = 0.0;
    for (theta, phi) in sphere_directions(N_SPHERE_DIRS) {
        let g = sph_to_cart(theta, phi);
        let d_app = quadratic_form_from_elements(d, g);
        if d_app < 1e-15 {
            continue;
        }
        let w_app = compute_w_contraction(w, g);
        let k = (md / d_app).powi(2) * w_app;
        sum += k;
    }
    sum / N_SPHERE_DIRS as f64
}

/// Compute axial kurtosis `AK = K(e₁)` along the principal eigenvector.
///
/// `AK = (MD / λ₁)² · W(e₁)` where `λ₁` is the largest eigenvalue.
fn compute_ak(
    w: &[f64; 15],
    md: f64,
    pev: [f64; 3],
    lambda1: f64,
) -> f64 {
    if md < 1e-15 || lambda1 < 1e-15 {
        return 0.0;
    }
    let w_pev = compute_w_contraction(w, pev);
    (md / lambda1).powi(2) * w_pev
}

/// Compute radial kurtosis `RK = ⟨K(g)⟩_{g⟂e₁}`, the mean kurtosis over
/// directions perpendicular to the principal eigenvector.
fn compute_rk(d: &[f64; 6], w: &[f64; 15], md: f64, pev: [f64; 3]) -> f64 {
    if md < 1e-15 {
        return 0.0;
    }
    let (u, v) = perp_basis(pev);
    let n_samples = 36;
    let mut sum = 0.0;
    for i in 0..n_samples {
        let phi = std::f64::consts::TAU * i as f64 / n_samples as f64;
        let g = [
            u[0] * phi.cos() + v[0] * phi.sin(),
            u[1] * phi.cos() + v[1] * phi.sin(),
            u[2] * phi.cos() + v[2] * phi.sin(),
        ];
        let d_app = quadratic_form_from_elements(d, g);
        if d_app < 1e-15 {
            continue;
        }
        let w_app = compute_w_contraction(w, g);
        sum += (md / d_app).powi(2) * w_app;
    }
    sum / n_samples as f64
}

/// Evaluate `gᵀDg` from the six Voigt elements.
fn quadratic_form_from_elements(d: &[f64; 6], g: [f64; 3]) -> f64 {
    let [gx, gy, gz] = g;
    d[0] * gx * gx
        + d[1] * gy * gy
        + d[2] * gz * gz
        + 2.0 * d[3] * gx * gy
        + 2.0 * d[4] * gx * gz
        + 2.0 * d[5] * gy * gz
}

///Construct two orthonormal vectors spanning the plane perpendicular to `e`.
fn perp_basis(e: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    // Pick an axis not parallel to e.
    let ref_dir = if e[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    // u = ref_dir × e / ‖ref_dir × e‖
    let u = [
        ref_dir[1] * e[2] - ref_dir[2] * e[1],
        ref_dir[2] * e[0] - ref_dir[0] * e[2],
        ref_dir[0] * e[1] - ref_dir[1] * e[0],
    ];
    let norm_u = (u[0].powi(2) + u[1].powi(2) + u[2].powi(2)).sqrt();
    let u_norm = [u[0] / norm_u, u[1] / norm_u, u[2] / norm_u];
    // v = e × u
    let v = [
        e[1] * u_norm[2] - e[2] * u_norm[1],
        e[2] * u_norm[0] - e[0] * u_norm[2],
        e[0] * u_norm[1] - e[1] * u_norm[0],
    ];
    (u_norm, v)
}

/// Generate quasi-uniform directions on the sphere via an equiangular grid.
fn sphere_directions(n: usize) -> impl Iterator<Item = (f64, f64)> {
    let n_theta = (n as f64).sqrt().ceil() as usize;
    let n_phi = 2 * n_theta;
    let total = n_theta * n_phi;
    (0..total).map(move |idx| {
        let ti = idx / n_phi;
        let pi = idx % n_phi;
        let theta = std::f64::consts::PI * (ti as f64 + 0.5) / n_theta as f64;
        let phi = std::f64::consts::TAU * pi as f64 / n_phi as f64;
        (theta, phi)
    })
}

/// Convert spherical coordinates to a unit Cartesian direction.
fn sph_to_cart(theta: f64, phi: f64) -> [f64; 3] {
    let sin_theta = theta.sin();
    [sin_theta * phi.cos(), sin_theta * phi.sin(), theta.cos()]
}


#[cfg(test)]
mod tests;
