//! Diffusion tensor imaging (DTI) — log-linear tensor estimation.
//!
//! DTI is the baseline diffusion model from
//! [ADR 0017](../../../docs/adr/0017-diffusion-mri-pipeline.md).  The
//! normalised signal `S/S₀` at gradient direction `g` and b-value `b` is
//!
//! ```text
//! S/S₀ = exp(−b · gᵀ D g)
//! ```
//!
//! Taking the log gives a linear system in the six unique tensor elements:
//!
//! ```text
//! ln(S/S₀) = −b · [gₓ²  g_y²  g_z²  2gₓg_y  2gₓg_z  2g_yg_z] · d
//! ```
//!
//! where `d = [Dₓₓ, D_yy, D_zz, Dₓy, Dₓz, D_yz]ᵀ`.  RITK assembles the
//! design matrix over all weighted acquisitions and solves via
//! [`leto_ops::solve_least_squares`] — the dense QR path that the
//! [ADR 0036 decision 2](../../../docs/adr/0036-neuroimaging-and-mr-ownership.md)
//! assigns to Leto.
//!
//! Fractional anisotropy, mean diffusivity, and the principal eigenvector
//! are derived from the fitted tensor through a closed-form 3×3 symmetric
//! eigendecomposition.

use leto::{Array1, Array2};
use ritk_diffusion_scheme::{DiffusionWeighting, GradientFrame, GradientScheme};

// ── Error ─────────────────────────────────────────────────────────────────────

/// Failure while estimating a diffusion tensor.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum DtiError {
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
    /// The number of weighted directions is fewer than the six unknowns.
    #[error("{direction_count} diffusion-weighted directions cannot identify 6 tensor elements")]
    Underdetermined {
        /// Weighted measurement count.
        direction_count: usize,
    },
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
    /// Leto could not construct or solve the least-squares system.
    #[error("least-squares solve failed: {0}")]
    SolveFailed(String),
    /// The fitted tensor is not symmetric positive-definite.
    #[error("fitted tensor eigenvalue {index} = {value} is not positive")]
    NonPositiveEigenvalue {
        /// Eigenvalue index (0, 1, 2).
        index: usize,
        /// Invalid eigenvalue.
        value: f64,
    },
}

// ── Configuration ─────────────────────────────────────────────────────────────

/// Validated DTI configuration.
#[derive(Debug, Clone, Copy)]
pub struct DtiConfig {
    b0_threshold: DiffusionWeighting,
}

impl DtiConfig {
    /// Construct a DTI configuration.
    ///
    /// `b0_threshold` classifies reference (≤ threshold) and weighted
    /// volumes.
    pub const fn new(b0_threshold: DiffusionWeighting) -> Self {
        Self { b0_threshold }
    }

    /// Threshold separating b0 and weighted acquisitions.
    #[must_use]
    pub const fn b0_threshold(self) -> DiffusionWeighting {
        self.b0_threshold
    }
}

impl Default for DtiConfig {
    fn default() -> Self {
        Self {
            b0_threshold: DiffusionWeighting::from_seconds_per_square_millimeter(50.0)
                .expect("invariant: default b0 threshold is finite and nonnegative"),
        }
    }
}

// ── Diffusion Tensor ──────────────────────────────────────────────────────────

/// Estimated diffusion tensor at one voxel.
///
/// The six elements are stored in the order `[Dₓₓ, D_yy, D_zz, Dₓy, Dₓz, D_yz]`
/// (Voigt notation, upper triangle).  Units are mm²/s.
#[derive(Debug, Clone)]
pub struct DiffusionTensor {
    elements: [f64; 6],
    eigenvalues: [f64; 3],
    principal_eigenvector: [f64; 3],
    baseline_signal: f64,
    residual_norm: f64,
    frame: GradientFrame,
}

impl DiffusionTensor {
    /// Six unique tensor elements `[Dₓₓ, D_yy, D_zz, Dₓy, Dₓz, D_yz]`
    /// in mm²/s.  The full symmetric 3×3 matrix is recoverable as
    ///
    /// ```text
    /// [Dₓₓ  Dₓy  Dₓz]
    /// [Dₓy  D_yy  D_yz]
    /// [Dₓz  D_yz  D_zz]
    /// ```
    #[must_use]
    pub fn elements(&self) -> &[f64; 6] {
        &self.elements
    }

    /// Reconstruct the full 3×3 symmetric tensor.
    #[must_use]
    pub fn matrix(&self) -> [[f64; 3]; 3] {
        let [dxx, dyy, dzz, dxy, dxz, dyz] = self.elements;
        [[dxx, dxy, dxz], [dxy, dyy, dyz], [dxz, dyz, dzz]]
    }

    /// Three eigenvalues `λ₀ ≥ λ₁ ≥ λ₂` in mm²/s, sorted descending.
    #[must_use]
    pub fn eigenvalues(&self) -> &[f64; 3] {
        &self.eigenvalues
    }

    /// Fractional anisotropy — a rotationally invariant measure in `[0, 1]`.
    ///
    /// ```text
    /// FA = √(3/2) · √(Σ(λᵢ − MD)²) / √(Σ λᵢ²)
    /// ```
    ///
    /// Zero for perfectly isotropic diffusion; approaches one for a
    /// maximally anisotropic prolate tensor.
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

    /// Mean diffusivity in mm²/s — the average of the three eigenvalues.
    #[must_use]
    pub fn md(&self) -> f64 {
        (self.eigenvalues[0] + self.eigenvalues[1] + self.eigenvalues[2]) / 3.0
    }

    /// Principal eigenvector `∥PEV∥ = 1` corresponding to the largest
    /// eigenvalue, in the scheme's coordinate frame.
    #[must_use]
    pub fn principal_eigenvector(&self) -> [f64; 3] {
        self.principal_eigenvector
    }

    /// Mean signal over b0 acquisitions.
    #[must_use]
    pub const fn baseline_signal(&self) -> f64 {
        self.baseline_signal
    }

    /// ‖design · d − ln(S/S₀)‖₂ after the least-squares solve.
    #[must_use]
    pub const fn residual_norm(&self) -> f64 {
        self.residual_norm
    }

    /// Coordinate frame the tensor axes are expressed in.
    #[must_use]
    pub const fn frame(&self) -> GradientFrame {
        self.frame
    }

    /// Predicted signal at a unit gradient direction for a given b-value.
    #[must_use]
    pub fn predict_signal(&self, direction: [f64; 3], b_value: f64) -> f64 {
        let q = self.quadratic_form(direction);
        self.baseline_signal * (-b_value * q).exp()
    }

    /// Quadratic form `gᵀ D g` for a unit direction `g`.
    #[must_use]
    pub fn quadratic_form(&self, direction: [f64; 3]) -> f64 {
        let [dxx, dyy, dzz, dxy, dxz, dyz] = self.elements;
        let [gx, gy, gz] = direction;
        dxx * gx * gx
            + dyy * gy * gy
            + dzz * gz * gz
            + 2.0 * dxy * gx * gy
            + 2.0 * dxz * gx * gz
            + 2.0 * dyz * gy * gz
    }
}

// ── Estimation ────────────────────────────────────────────────────────────────

/// Estimate a diffusion tensor from one voxel's signals via log-linear fit.
///
/// # Errors
///
/// Returns a typed error for count mismatch, non-finite signals, missing b0
/// or weighted samples, fewer than six DWI directions, invalid
/// baseline/normalised signals, a failed least-squares solve, or a fitted
/// tensor that is not positive-definite.
pub fn estimate_dti(
    scheme: &GradientScheme,
    signals: &[f64],
    config: DtiConfig,
) -> Result<DiffusionTensor, DtiError> {
    // ── Validation ────────────────────────────────────────────────────────
    if signals.len() != scheme.len() {
        return Err(DtiError::SignalLengthMismatch {
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
        return Err(DtiError::NonFiniteSignal { index, value });
    }

    let b0_indices = scheme.b0_indices(config.b0_threshold());
    let dwi_indices = scheme.dwi_indices(config.b0_threshold());
    if b0_indices.is_empty() {
        return Err(DtiError::NoB0Volumes);
    }
    if dwi_indices.is_empty() {
        return Err(DtiError::NoDwiDirections);
    }
    if dwi_indices.len() < 6 {
        return Err(DtiError::Underdetermined {
            direction_count: dwi_indices.len(),
        });
    }

    let baseline_signal =
        b0_indices.iter().map(|index| signals[*index]).sum::<f64>() / b0_indices.len() as f64;
    if !baseline_signal.is_finite() || baseline_signal <= 0.0 {
        return Err(DtiError::InvalidBaseline {
            value: baseline_signal,
        });
    }

    // ── Build design matrix ───────────────────────────────────────────────
    let n_dwi = dwi_indices.len();
    let mut design = Array2::zeros([n_dwi, 6]);
    let mut log_signals = Array1::zeros([n_dwi]);

    for (row, &global_index) in dwi_indices.iter().enumerate() {
        let entry = &scheme.directions()[global_index];
        let b = entry.weighting().seconds_per_square_millimeter();
        let [gx, gy, gz] = entry.direction().to_array();
        let signal = signals[global_index];
        let normalised = signal / baseline_signal;
        if !normalised.is_finite() || normalised <= 0.0 {
            return Err(DtiError::InvalidNormalisedSignal {
                index: global_index,
                value: normalised,
            });
        }
        // Row: [−b·gₓ², −b·g_y², −b·g_z², −2b·gₓg_y, −2b·gₓg_z, −2b·g_yg_z]
        design[[row, 0]] = -b * gx * gx;
        design[[row, 1]] = -b * gy * gy;
        design[[row, 2]] = -b * gz * gz;
        design[[row, 3]] = -2.0 * b * gx * gy;
        design[[row, 4]] = -2.0 * b * gx * gz;
        design[[row, 5]] = -2.0 * b * gy * gz;
        log_signals[row] = normalised.ln();
    }

    // ── Solve ─────────────────────────────────────────────────────────────
    let solution = leto_ops::solve_least_squares(&design.view(), &log_signals.view())
        .map_err(|error| DtiError::SolveFailed(error.to_string()))?;

    let elements = [
        solution[0],
        solution[1],
        solution[2],
        solution[3],
        solution[4],
        solution[5],
    ];

    // ── Decompose ─────────────────────────────────────────────────────────
    let (eigenvalues, principal_eigenvector) = decompose_3x3_symmetric(elements)
        .map_err(|(index, value)| DtiError::NonPositiveEigenvalue { index, value })?;

    let residual = compute_residual(&design, &solution, &log_signals);

    Ok(DiffusionTensor {
        elements,
        eigenvalues,
        principal_eigenvector,
        baseline_signal,
        residual_norm: residual,
        frame: scheme.frame(),
    })
}

// ── 3×3 symmetric eigendecomposition ──────────────────────────────────────────

/// Compute eigenvalues (sorted descending) and principal eigenvector of a
/// 3×3 symmetric matrix from its six unique Voigt elements.
///
/// Uses the analytic solution of the cubic characteristic polynomial via
/// the trigonometric formula (three real roots guaranteed for symmetric
/// matrices).  The principal eigenvector is extracted from the nullspace of
/// `D − λ₀I`.
///
/// Returns `Err((index, value))` when an eigenvalue is not strictly positive.
pub(crate) fn decompose_3x3_symmetric(
    elements: [f64; 6],
) -> Result<([f64; 3], [f64; 3]), (usize, f64)> {
    let [dxx, dyy, dzz, dxy, dxz, dyz] = elements;

    // Invariants of the characteristic polynomial λ³ − I₁λ² + I₂λ − I₃.
    let i1 = dxx + dyy + dzz; // trace
    let i2 = dxx * dyy + dxx * dzz + dyy * dzz - dxy * dxy - dxz * dxz - dyz * dyz;
    let i3 = dxx * dyy * dzz + 2.0 * dxy * dxz * dyz
        - dxx * dyz * dyz
        - dyy * dxz * dxz
        - dzz * dxy * dxy;

    // Shifted cubic: μ³ + p·μ + q = 0, μ = λ − i1/3.
    // p ≤ 0 for symmetric real matrices, but fp rounding may nudge it positive.
    let p = (i2 - i1 * i1 / 3.0).min(0.0);
    let q = -2.0 * i1 * i1 * i1 / 27.0 + i1 * i2 / 3.0 - i3;
    let shift = i1 / 3.0;

    // Near-isotropic / degenerate: p ≈ 0 ⇒ all eigenvalues ≈ shift.
    let sqrt_neg_p_over_3 = (-p / 3.0).sqrt();
    if sqrt_neg_p_over_3 < 1e-15 {
        let eigenvalues = [shift, shift, shift];
        // Strict positivity check.
        for (idx, &val) in eigenvalues.iter().enumerate() {
            if val <= 0.0 {
                return Err((idx, val));
            }
        }
        return Ok((eigenvalues, [1.0, 0.0, 0.0]));
    }

    // Three real roots via trigonometric formula (symmetric ⇒ all real).
    let arg = (-q / (2.0 * sqrt_neg_p_over_3.powi(3))).clamp(-1.0, 1.0);
    let phi = arg.acos();
    let two_r = 2.0 * sqrt_neg_p_over_3;

    let mu0 = two_r * (phi / 3.0).cos();
    let mu1 = two_r * ((phi + 2.0 * std::f64::consts::PI) / 3.0).cos();
    let mu2 = two_r * ((phi + 4.0 * std::f64::consts::PI) / 3.0).cos();

    let mut eigenvalues = [mu0 + shift, mu1 + shift, mu2 + shift];
    eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    for (idx, &val) in eigenvalues.iter().enumerate() {
        // Allow tiny negative values from fp error on near-zero eigenvalues.
        if val < -1e-10 {
            return Err((idx, val));
        }
    }
    // Clamp fp-negative eigenvalues to zero.
    for val in &mut eigenvalues {
        if *val < 0.0 {
            *val = 0.0;
        }
    }

    // Principal eigenvector: try all three row-pair cross products of
    // (D − λ₀I) and keep the one with the largest norm.  A single pair
    // can be degenerate when the two rows are linearly dependent.
    let l0 = eigenvalues[0];
    let m = [
        [dxx - l0, dxy, dxz],
        [dxy, dyy - l0, dyz],
        [dxz, dyz, dzz - l0],
    ];
    let cross_products = [
        // row0 × row1
        [
            m[0][1] * m[1][2] - m[0][2] * m[1][1],
            m[0][2] * m[1][0] - m[0][0] * m[1][2],
            m[0][0] * m[1][1] - m[0][1] * m[1][0],
        ],
        // row0 × row2
        [
            m[0][1] * m[2][2] - m[0][2] * m[2][1],
            m[0][2] * m[2][0] - m[0][0] * m[2][2],
            m[0][0] * m[2][1] - m[0][1] * m[2][0],
        ],
        // row1 × row2
        [
            m[1][1] * m[2][2] - m[1][2] * m[2][1],
            m[1][2] * m[2][0] - m[1][0] * m[2][2],
            m[1][0] * m[2][1] - m[1][1] * m[2][0],
        ],
    ];
    let norms: [f64; 3] = cross_products.map(|v| v[0].powi(2) + v[1].powi(2) + v[2].powi(2));
    let (best_idx, _) = norms
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, &0.0));
    let pev = cross_products[best_idx];
    let norm = norms[best_idx].sqrt();
    if norm < 1e-15 {
        return Ok((eigenvalues, [1.0, 0.0, 0.0]));
    }
    let principal_eigenvector = [pev[0] / norm, pev[1] / norm, pev[2] / norm];

    Ok((eigenvalues, principal_eigenvector))
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Infallible wrapper around [`decompose_3x3_symmetric`] for use after a
/// successful DTI or LM fit where non-positive eigenvalues indicate a
/// solver defect rather than recoverable data error.
///
/// # Panics
///
/// Panics if any eigenvalue is not strictly positive.
pub(crate) fn decompose_3x3_symmetric_infallible(elements: [f64; 6]) -> ([f64; 3], [f64; 3]) {
    match decompose_3x3_symmetric(elements) {
        Ok(result) => result,
        Err((idx, val)) => {
            panic!(
                "post-fit D tensor eigenvalue {idx} = {val} is not positive — \
                 this is a solver defect, not a data error"
            );
        }
    }
}

fn compute_residual(
    design: &Array2<f64>,
    solution: &Array1<f64>,
    log_signals: &Array1<f64>,
) -> f64 {
    let n_rows = design.shape()[0];
    let n_cols = design.shape()[1];
    let mut sum_sq = 0.0;
    for row in 0..n_rows {
        let mut pred = 0.0;
        for col in 0..n_cols {
            pred += design[[row, col]] * solution[col];
        }
        let diff = pred - log_signals[row];
        sum_sq += diff * diff;
    }
    sum_sq.sqrt()
}

#[cfg(test)]
mod tests;
