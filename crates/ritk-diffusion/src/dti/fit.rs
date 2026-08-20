//! The linear system behind tensor estimation, and how its rows are weighted.
//!
//! # Why weighting is not optional
//!
//! Taking the logarithm makes the tensor system linear, which is the whole
//! reason the log-linear fit exists — but it does not preserve the noise model.
//! Writing a measurement as `S = Ŝ + n` with `n` of roughly constant variance
//! `σ²` across the acquisition,
//!
//! ```text
//! ln S = ln Ŝ + ln(1 + n/Ŝ) ≈ ln Ŝ + n/Ŝ
//! ```
//!
//! so `var(ln Sᵢ) ≈ σ²/Ŝᵢ²`. The variance of the *transformed* measurement
//! therefore scales with the inverse square of the signal. Ordinary least
//! squares assumes it does not, and so gives every row equal say. The rows it
//! over-trusts are exactly the least reliable ones: the strongly attenuated
//! measurements — high b-value, gradient along the fibre — where `Ŝ` is
//! smallest. The result is a systematic bias, not just added variance, and it
//! runs in a consistent direction: the attenuated directions are pulled toward
//! the noise floor, which flattens the recovered principal eigenvalue and
//! inflates the small ones.
//!
//! Generalised least squares removes it. The optimal weight is the inverse
//! variance, `wᵢ = Ŝᵢ²/σ²`; the common `σ²` factor divides out of the normal
//! equations, leaving `wᵢ = Ŝᵢ²`. The weighted problem is solved by scaling
//! each row of the design matrix and the corresponding right-hand side by
//! `√wᵢ = Ŝᵢ`, then running the same unweighted solver — minimising
//! `Σ wᵢ rᵢ²` is exactly minimising the residual of the scaled system.
//!
//! # Why the weights come from a prior fit, not from the data
//!
//! `Ŝᵢ` is the *noise-free* signal, which is unknown. Substituting the measured
//! `Sᵢ` is tempting and wrong: the weight then correlates with the same noise
//! realisation as the residual it multiplies, and a measurement that happened
//! to fluctuate high is rewarded with extra influence. That reintroduces a bias
//! of its own.
//!
//! The fix is to take `Ŝᵢ` from a previous fit's prediction, which depends on
//! the noise only through the whole fit rather than through the single sample
//! being weighted. So the estimator runs one ordinary pass to get a tensor,
//! predicts the signals it implies, weights by those, and solves again. Further
//! passes are available but yield little: the first reweighting captures
//! essentially all the improvement.
//!
//! # References
//!
//! * Veraart, J., Sijbers, J., Sunaert, S., Leemans, A. & Jeurissen, B. (2013).
//!   Weighted linear least squares estimation of diffusion MRI parameters:
//!   Strengths, limitations, and pitfalls. *NeuroImage* 81:335–346.
//! * Salvador, R., Peña, A., Menon, D. K., Carpenter, T. A., Pickard, J. D. &
//!   Bullmore, E. T. (2005). Formal characterization and extension of the
//!   linearized diffusion tensor model. *Human Brain Mapping* 24(2):144–155.

use std::num::NonZeroU8;

use leto::{Array1, Array2};

use super::DtiError;

/// Predicted signals below this fraction of the baseline carry no weight.
///
/// A weight is a signal squared, so a prediction that has underflowed toward
/// zero would zero out its row entirely and silently drop a measurement. The
/// floor keeps every acquisition in the system. It is set far below the noise
/// floor of any real acquisition — an attenuation of `10⁻⁶` is 120 dB, which no
/// diffusion measurement reaches — so it never binds on data, only on a
/// diverging intermediate fit.
const MINIMUM_RELATIVE_PREDICTION: f64 = 1.0e-6;

/// How the six tensor elements are recovered from the log-linear system.
///
/// The variants differ in noise model, not in the tensor being estimated: both
/// solve the same overdetermined system, and on noiseless data — where the
/// system is consistent — they return the identical exact solution. They part
/// company as soon as the measurements carry noise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorFit {
    /// Unweighted least squares on the log-signals.
    ///
    /// Treats every acquisition as equally reliable, which the log transform
    /// makes untrue. Retained because it is the exact estimator the weighted
    /// form is measured against, and because it is what a caller reproducing a
    /// published log-linear pipeline needs.
    Ordinary,

    /// Iteratively reweighted least squares with weights from the predicted
    /// signal.
    ///
    /// The default, and the estimator to use on real data.
    Weighted {
        /// Reweighted solves performed after the initial unweighted one.
        ///
        /// One is the documented recommendation; further passes change the
        /// result by progressively less.
        reweight_passes: NonZeroU8,
    },
}

impl Default for TensorFit {
    fn default() -> Self {
        Self::Weighted {
            reweight_passes: NonZeroU8::MIN,
        }
    }
}

impl TensorFit {
    /// Number of reweighted solves this variant performs.
    #[must_use]
    pub const fn reweight_passes(self) -> u8 {
        match self {
            Self::Ordinary => 0,
            Self::Weighted { reweight_passes } => reweight_passes.get(),
        }
    }
}

/// Solve the log-linear tensor system under the configured weighting.
///
/// `design` holds one row per diffusion-weighted acquisition and `log_signals`
/// the matching `ln(S/S₀)`. Returns the six Voigt tensor elements.
///
/// # Errors
///
/// [`DtiError::SolveFailed`] when the underlying least-squares solve fails on
/// any pass.
pub(crate) fn solve_log_linear(
    design: &Array2<f64>,
    log_signals: &Array1<f64>,
    fit: TensorFit,
) -> Result<Array1<f64>, DtiError> {
    let mut solution = least_squares(design, log_signals)?;

    for _ in 0..fit.reweight_passes() {
        // √wᵢ = Ŝᵢ/S₀ = exp(row · d): the prediction the current tensor implies
        // for this acquisition, normalised by the baseline exactly as the
        // right-hand side is.
        let scales = predicted_relative_signals(design, &solution);
        let (scaled_design, scaled_rhs) = scale_rows(design, log_signals, &scales);
        solution = least_squares(&scaled_design, &scaled_rhs)?;
    }

    Ok(solution)
}

/// `‖design · d − y‖₂` — the unweighted residual of a solution.
///
/// Reported unweighted whichever estimator produced the solution, so that the
/// number means the same thing across configurations and remains comparable to
/// the log-signals it is a residual of.
pub(crate) fn residual_norm(
    design: &Array2<f64>,
    solution: &Array1<f64>,
    log_signals: &Array1<f64>,
) -> f64 {
    let rows = design.shape()[0];
    let columns = design.shape()[1];
    let mut sum_of_squares = 0.0;
    for row in 0..rows {
        let mut predicted = 0.0;
        for column in 0..columns {
            predicted += design[[row, column]] * solution[column];
        }
        let difference = predicted - log_signals[row];
        sum_of_squares += difference * difference;
    }
    sum_of_squares.sqrt()
}

/// `exp(row · d)` per row — the baseline-normalised signal the solution
/// predicts, floored so no acquisition can be weighted out of the system.
fn predicted_relative_signals(design: &Array2<f64>, solution: &Array1<f64>) -> Vec<f64> {
    let rows = design.shape()[0];
    let columns = design.shape()[1];
    (0..rows)
        .map(|row| {
            let mut exponent = 0.0;
            for column in 0..columns {
                exponent += design[[row, column]] * solution[column];
            }
            // A diverging intermediate fit can push the exponent positive, which
            // would predict a signal above baseline and over-weight the row.
            // Clamping at the baseline keeps the weight a signal amplitude.
            exponent.exp().clamp(MINIMUM_RELATIVE_PREDICTION, 1.0)
        })
        .collect()
}

/// Scale each row of the system by its weight's square root.
fn scale_rows(
    design: &Array2<f64>,
    log_signals: &Array1<f64>,
    scales: &[f64],
) -> (Array2<f64>, Array1<f64>) {
    let rows = design.shape()[0];
    let columns = design.shape()[1];
    let mut scaled_design = Array2::zeros([rows, columns]);
    let mut scaled_rhs = Array1::zeros([rows]);
    for (row, &scale) in scales.iter().enumerate() {
        for column in 0..columns {
            scaled_design[[row, column]] = design[[row, column]] * scale;
        }
        scaled_rhs[row] = log_signals[row] * scale;
    }
    (scaled_design, scaled_rhs)
}

fn least_squares(design: &Array2<f64>, rhs: &Array1<f64>) -> Result<Array1<f64>, DtiError> {
    leto_ops::solve_least_squares(&design.view(), &rhs.view())
        .map_err(|error| DtiError::SolveFailed(error.to_string()))
}

#[cfg(test)]
mod tests;
