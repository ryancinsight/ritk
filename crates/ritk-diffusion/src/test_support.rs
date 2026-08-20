//! Synthetic acquisitions shared by the crate's tests.
//!
//! Model tests need a gradient scheme and signals generated from a known
//! tensor, so that what a fit recovers can be checked against what was put in.
//! These live here rather than in one model's test module because every model
//! needs the same two things, and a second copy would be free to drift from the
//! first.

use rand::{Rng, SeedableRng, rngs::StdRng};
use ritk_diffusion_scheme::{DiffusionWeighting, GradientDirection, GradientFrame, GradientScheme};
use ritk_spatial::Vector;

/// A reproducible generator for the noise realisations a fitting test compares
/// estimators over.
///
/// Seeded explicitly so a failure is replayable: comparing two estimators over
/// random noise is only evidence if both saw the same noise, and only
/// actionable if the run that failed can be run again.
pub(crate) fn seeded_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

/// A diffusion weighting in s/mm².
pub(crate) fn weighting(value: f64) -> DiffusionWeighting {
    DiffusionWeighting::from_seconds_per_square_millimeter(value).expect("finite weighting")
}

/// A scheme with one b = 0 reference and `direction_count` weighted directions.
///
/// Directions are placed by the Fibonacci spiral, which spreads them near
/// uniformly over the sphere at any count — a scheme clustered on one hemisphere
/// would leave the tensor poorly conditioned and confound a fit failure with a
/// sampling failure.
pub(crate) fn scheme(direction_count: usize) -> GradientScheme {
    schemes_with_references(direction_count, 1)
}

/// A scheme with `reference_count` b = 0 volumes ahead of the weighted set.
///
/// Multiple references matter where the code under test averages them, which a
/// single-reference scheme cannot distinguish from picking the first.
pub(crate) fn schemes_with_references(
    direction_count: usize,
    reference_count: usize,
) -> GradientScheme {
    let mut entries = Vec::with_capacity(reference_count + direction_count);
    for _ in 0..reference_count {
        entries.push(
            GradientDirection::new(weighting(0.0), Vector::new([0.0, 0.0, 0.0])).expect("valid b0"),
        );
    }

    let golden_angle = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    #[expect(
        clippy::cast_precision_loss,
        reason = "direction counts are small integers"
    )]
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

/// Signals a tensor would produce under the Stejskal-Tanner relation.
///
/// `S = S₀ exp(-b gᵀDg)`, with `tensor_elements` in Voigt order
/// `[Dₓₓ, D_yy, D_zz, Dₓy, Dₓz, D_yz]`. Generating signals from the forward
/// model is what makes the fit checkable: the answer is known exactly.
pub(crate) fn dti_signal(scheme: &GradientScheme, tensor_elements: [f64; 6], s0: f64) -> Vec<f64> {
    let [dxx, dyy, dzz, dxy, dxz, dyz] = tensor_elements;
    scheme
        .directions()
        .iter()
        .map(|entry| {
            let b = entry.weighting().seconds_per_square_millimeter();
            if b == 0.0 {
                return s0;
            }
            let [gx, gy, gz] = entry.direction().to_array();
            let q = dxx * gx * gx
                + dyy * gy * gy
                + dzz * gz * gz
                + 2.0 * dxy * gx * gy
                + 2.0 * dxz * gx * gz
                + 2.0 * dyz * gy * gz;
            s0 * (-b * q).exp()
        })
        .collect()
}

/// Add Rician noise of scale `sigma` to a magnitude series.
///
/// MRI magnitude images are Rician, not Gaussian: the reconstruction takes the
/// modulus of a complex pair whose real and imaginary channels each carry
/// independent Gaussian noise of the same scale, so
/// `S_noisy = √((S + n₁)² + n₂²)`. The distinction matters for a fitting test —
/// the modulus is strictly positive, so noise biases low signals *upward*
/// rather than averaging out, and an estimator can only be judged against the
/// noise it will actually meet.
///
/// `sigma` is the per-channel Gaussian scale in signal units; an acquisition's
/// nominal SNR is `S₀ / sigma`.
pub(crate) fn add_rician_noise(signals: &[f64], sigma: f64, rng: &mut StdRng) -> Vec<f64> {
    signals
        .iter()
        .map(|signal| {
            let real = signal + sigma * standard_normal(rng);
            let imaginary = sigma * standard_normal(rng);
            real.hypot(imaginary)
        })
        .collect()
}

/// One standard normal sample by the Box-Muller transform.
///
/// The uniform is drawn on `(0, 1]` rather than `[0, 1)` because the transform
/// takes its logarithm, which is unbounded at zero.
fn standard_normal(rng: &mut StdRng) -> f64 {
    let uniform = 1.0 - rng.random::<f64>();
    let angle = std::f64::consts::TAU * rng.random::<f64>();
    (-2.0 * uniform.ln()).sqrt() * angle.cos()
}

/// Root-mean-square error of a sample against a known truth.
pub(crate) fn rmse(estimates: &[f64], truth: f64) -> f64 {
    #[expect(
        clippy::cast_precision_loss,
        reason = "trial counts are small integers"
    )]
    let count = estimates.len() as f64;
    let sum: f64 = estimates
        .iter()
        .map(|value| (value - truth) * (value - truth))
        .sum();
    (sum / count).sqrt()
}

/// Mean of a sample — the estimator's bias against a known truth is
/// `mean(estimates) − truth`.
pub(crate) fn mean(values: &[f64]) -> f64 {
    #[expect(
        clippy::cast_precision_loss,
        reason = "trial counts are small integers"
    )]
    let count = values.len() as f64;
    values.iter().sum::<f64>() / count
}
