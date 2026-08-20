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
//! [`leto_ops::solve_least_squares`] — the dense QR path that
//! [ADR 0017](../../../docs/adr/0017-diffusion-mri-pipeline.md) assigns to
//! Leto.
//!
//! # Module map
//!
//! | Module | Responsibility |
//! |--------|----------------|
//! | [`fit`] | The linear system, and the row weighting that makes it a sound estimator |
//! | `eigen` | Closed-form eigendecomposition of the fitted symmetric tensor |
//! | [`invariants`] | Rotationally invariant scalars derived from the eigenvalues |
//!
//! The log transform does not preserve the noise model, so the estimator is
//! weighted by default; [`fit`] derives why and what the weights are.

mod eigen;
pub mod fit;
pub mod invariants;

use leto::{Array1, Array2};
use ritk_diffusion_scheme::{DiffusionWeighting, GradientFrame, GradientScheme};

pub use fit::TensorFit;

pub(crate) use eigen::{SymmetricEigen, symmetric_eigen};

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
    fit: TensorFit,
}

impl DtiConfig {
    /// Construct a DTI configuration with the default estimator.
    ///
    /// `b0_threshold` classifies reference (≤ threshold) and weighted
    /// volumes.  The estimator is [`TensorFit::default`] — weighted least
    /// squares, which is what real data needs; select another with
    /// [`Self::with_fit`].
    pub fn new(b0_threshold: DiffusionWeighting) -> Self {
        Self {
            b0_threshold,
            fit: TensorFit::default(),
        }
    }

    /// Replace the estimator.
    #[must_use]
    pub const fn with_fit(self, fit: TensorFit) -> Self {
        Self { fit, ..self }
    }

    /// Threshold separating b0 and weighted acquisitions.
    #[must_use]
    pub const fn b0_threshold(self) -> DiffusionWeighting {
        self.b0_threshold
    }

    /// The estimator the log-linear system is solved with.
    #[must_use]
    pub const fn fit(self) -> TensorFit {
        self.fit
    }
}

impl Default for DtiConfig {
    fn default() -> Self {
        Self::new(
            DiffusionWeighting::from_seconds_per_square_millimeter(50.0)
                .expect("invariant: default b0 threshold is finite and nonnegative"),
        )
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
    eigen: SymmetricEigen,
    baseline_signal: f64,
    residual_norm: f64,
    frame: GradientFrame,
    fit: TensorFit,
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

    /// Three eigenvalues `λ₁ ≥ λ₂ ≥ λ₃` in mm²/s, sorted descending.
    #[must_use]
    pub fn eigenvalues(&self) -> &[f64; 3] {
        &self.eigen.values
    }

    /// The orthonormal eigenbasis, ordered to match [`Self::eigenvalues`].
    ///
    /// `eigenvectors()[0]` is the principal eigenvector; the other two span the
    /// plane transverse to it. Repeated eigenvalues leave their eigenvectors
    /// non-unique, and the basis is then one valid representative rather than a
    /// distinguished one — see the tensor's [`Self::mode`] and Westin measures
    /// for whether that degeneracy is present.
    #[must_use]
    pub fn eigenvectors(&self) -> &[[f64; 3]; 3] {
        &self.eigen.vectors
    }

    /// Principal eigenvector `∥PEV∥ = 1` corresponding to the largest
    /// eigenvalue, in the scheme's coordinate frame.
    #[must_use]
    pub fn principal_eigenvector(&self) -> [f64; 3] {
        self.eigen.vectors[0]
    }

    /// Fractional anisotropy — a rotationally invariant measure in `[0, 1]`.
    ///
    /// See [`invariants::fractional_anisotropy`].
    #[must_use]
    pub fn fa(&self) -> f64 {
        invariants::fractional_anisotropy(self.eigen.values)
    }

    /// Mean diffusivity in mm²/s — the average of the three eigenvalues.
    #[must_use]
    pub fn md(&self) -> f64 {
        invariants::mean_diffusivity(self.eigen.values)
    }

    /// Axial diffusivity `λ₁` in mm²/s — diffusivity along the principal axis.
    #[must_use]
    pub fn ad(&self) -> f64 {
        invariants::axial_diffusivity(self.eigen.values)
    }

    /// Radial diffusivity `(λ₂ + λ₃)/2` in mm²/s — diffusivity across the
    /// principal axis.
    #[must_use]
    pub fn rd(&self) -> f64 {
        invariants::radial_diffusivity(self.eigen.values)
    }

    /// Relative anisotropy — see [`invariants::relative_anisotropy`].
    #[must_use]
    pub fn relative_anisotropy(&self) -> f64 {
        invariants::relative_anisotropy(self.eigen.values)
    }

    /// Westin linear, planar, and spherical measures `(cₗ, cₚ, cₛ)`.
    ///
    /// See [`invariants::westin_measures`].
    #[must_use]
    pub fn westin_measures(&self) -> (f64, f64, f64) {
        invariants::westin_measures(self.eigen.values)
    }

    /// Mode of anisotropy in `[−1, 1]` — see [`invariants::mode`].
    #[must_use]
    pub fn mode(&self) -> f64 {
        invariants::mode(self.eigen.values)
    }

    /// Frobenius norm `‖D‖` in mm²/s.
    #[must_use]
    pub fn norm(&self) -> f64 {
        invariants::tensor_norm(self.eigen.values)
    }

    /// Direction-encoded colour `FA · |v₁|` in the tensor's own
    /// [`Self::frame`] — see [`invariants::colour_by_orientation`].
    #[must_use]
    pub fn colour_by_orientation(&self) -> [f64; 3] {
        invariants::colour_by_orientation(self.eigen.values, self.principal_eigenvector())
    }

    /// Mean signal over b0 acquisitions.
    #[must_use]
    pub const fn baseline_signal(&self) -> f64 {
        self.baseline_signal
    }

    /// ‖design · d − ln(S/S₀)‖₂ after the least-squares solve.
    ///
    /// Reported unweighted whichever estimator produced the tensor, so the
    /// number is comparable across [`TensorFit`] variants.
    #[must_use]
    pub const fn residual_norm(&self) -> f64 {
        self.residual_norm
    }

    /// Coordinate frame the tensor axes are expressed in.
    #[must_use]
    pub const fn frame(&self) -> GradientFrame {
        self.frame
    }

    /// The estimator that produced this tensor.
    #[must_use]
    pub const fn fit(&self) -> TensorFit {
        self.fit
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

    #[expect(
        clippy::cast_precision_loss,
        reason = "a diffusion series has far fewer volumes than f64's exact-integer range"
    )]
    let reference_count = b0_indices.len() as f64;
    let baseline_signal =
        b0_indices.iter().map(|index| signals[*index]).sum::<f64>() / reference_count;
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
    let solution = fit::solve_log_linear(&design, &log_signals, config.fit())?;
    let elements = [
        solution[0],
        solution[1],
        solution[2],
        solution[3],
        solution[4],
        solution[5],
    ];

    // ── Decompose ─────────────────────────────────────────────────────────
    let eigen = diffusion_eigen(elements)
        .map_err(|(index, value)| DtiError::NonPositiveEigenvalue { index, value })?;

    Ok(DiffusionTensor {
        elements,
        eigen,
        baseline_signal,
        residual_norm: fit::residual_norm(&design, &solution, &log_signals),
        frame: scheme.frame(),
        fit: config.fit(),
    })
}

// ── Diffusion-tensor eigen contract ───────────────────────────────────────────

/// Fraction of the leading eigenvalue below which a negative root is rounding
/// rather than a measurement.
///
/// A tensor fitted to an almost perfectly anisotropic voxel has a smallest
/// eigenvalue at the noise floor, and the closed-form decomposition resolves a
/// repeated root only to `√ε ≈ 1.5·10⁻⁸` of the tensor magnitude, so such a root
/// can come back slightly negative. The tolerance is relative to `λ₁` rather
/// than an absolute diffusivity because that is the scale the error actually
/// carries: an absolute bound would be far too loose for a low-diffusivity
/// tensor and would tighten into false rejections if the caller ever worked in
/// different units. A thousandfold margin over `√ε` still rejects any genuinely
/// negative fit, which sits at the scale of the eigenvalues themselves.
const EIGENVALUE_ROUNDING: f64 = 1.0e-5;

/// Eigendecompose a *fitted diffusion tensor*, enforcing positivity.
///
/// Positivity is a property of a measurement, not of a symmetric matrix: a
/// tensor whose eigenvalue is negative describes water that concentrates rather
/// than spreads, so the fit is rejected rather than reported. Callers whose
/// matrix is legitimately semi-definite use [`symmetric_eigen`] instead.
///
/// Returns `Err((index, value))` for the first eigenvalue that fails.
pub(crate) fn diffusion_eigen(elements: [f64; 6]) -> Result<SymmetricEigen, (usize, f64)> {
    let mut eigen = symmetric_eigen(elements);

    // An isotropic tensor is held to strict positivity, an anisotropic one to a
    // tolerance that absorbs fp error on a near-zero eigenvalue. The two rules
    // differ because an isotropic result has no small eigenvalue to round: all
    // three are the same number, so a non-positive one is the fit, not noise.
    let isotropic = (eigen.values[0] - eigen.values[2]).abs() <= 0.0;
    let rounding = EIGENVALUE_ROUNDING * eigen.values[0].abs();
    for (index, &value) in eigen.values.iter().enumerate() {
        let invalid = if isotropic {
            value <= 0.0
        } else {
            value < -rounding
        };
        if invalid {
            return Err((index, value));
        }
    }
    for value in &mut eigen.values {
        *value = value.max(0.0);
    }
    Ok(eigen)
}

/// Infallible [`diffusion_eigen`] for use after a successful fit, where a
/// non-positive eigenvalue indicates a solver defect rather than a data error.
///
/// # Panics
///
/// Panics if any eigenvalue is not positive.
pub(crate) fn diffusion_eigen_infallible(elements: [f64; 6]) -> SymmetricEigen {
    match diffusion_eigen(elements) {
        Ok(eigen) => eigen,
        Err((index, value)) => panic!(
            "post-fit D tensor eigenvalue {index} = {value} is not positive — \
             this is a solver defect, not a data error"
        ),
    }
}

#[cfg(test)]
mod tests;
