//! Multi-start mutual information registration with independent random restarts.
//!
//! # Algorithm
//!
//! This module implements a multi-start (random restart) strategy for global
//! mutual information registration. Instead of running a single gradient-based
//! optimisation from one initial transform, `num_starts` independent runs are
//! launched from perturbed initialisations and the run yielding the highest
//! final mutual information is returned.
//!
//! The approach is loosely inspired by *basin-hopping* (Wales & Doye, 1997):
//! accept the basin reached from each perturbed start unconditionally (no
//! Metropolis criterion), so the method reduces to pure independent restarts.
//! This is adequate when the Mattes MI landscape for rigid registration has a
//! moderate number of local maxima dominated by gross rotational ambiguity
//! (cf. Klein et al., 2007).
//!
//! # Perturbation Model
//!
//! Each restart (except the first, which uses `initial_transform` unmodified)
//! draws independent Gaussian perturbations for every rotation angle and every
//! translation component:
//!
//! ```text
//! θ'ᵢ = θᵢ + σ_rot   · N(0,1)      (i = α, β, γ)
//! t'ᵢ = tᵢ + σ_trans · N(0,1)      (i = tz, ty, tx)
//! ```
//!
//! Gaussian samples are produced by the Box–Muller transform driven by a
//! 64-bit linear congruential generator (LCG) with Knuth–Lewis constants,
//! removing any dependency on `rand` or the standard library's thread-local
//! RNG.
//!
//! # References
//!
//! - Wales, D. J., & Doye, J. P. K. (1997). Global optimization by
//!   basin-hopping and the lowest energy structures of Lennard-Jones clusters
//!   containing up to 110 atoms. *Journal of Physical Chemistry A*, 101(28),
//!   5111–5116. <https://doi.org/10.1021/jp970984n>
//!
//! - Klein, S., Staring, M., Murphy, K., Viergever, M. A., & Pluim, J. P. W.
//!   (2007). elastix: A toolbox for intensity-based medical image registration.
//!   *IEEE Transactions on Medical Imaging*, 29(1), 196–205.
//!   <https://doi.org/10.1109/TMI.2009.2035616>

use std::f64::consts::PI;

use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Tensor, TensorData};
use ritk_core::image::Image;
use ritk_core::transform::RigidTransform;

use super::registration::GlobalMiRegistration;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for multi-start Mattes MI rigid registration.
///
/// Controls the number of independent restarts, the magnitude of Gaussian
/// perturbations applied to each non-initial start, the PRNG seed, and the
/// underlying single-start MI registration parameters.
#[derive(Debug, Clone)]
pub struct MultiStartConfig {
    /// Number of independent optimisation starts (must be ≥ 1).
    ///
    /// Start 0 uses `initial_transform` unmodified; subsequent starts receive
    /// Gaussian-perturbed copies. Default: `5`.
    pub num_starts: usize,

    /// Standard deviation of rotation perturbations in radians.
    ///
    /// Applied independently to each Euler angle [α, β, γ]. Default: `0.2`
    /// (≈ 11°).
    pub rotation_perturbation_rad: f64,

    /// Standard deviation of translation perturbations in millimetres.
    ///
    /// Applied independently to each translation component [tz, ty, tx].
    /// Default: `15.0`.
    pub translation_perturbation_mm: f64,

    /// Seed for the 64-bit LCG used to draw perturbation samples.
    ///
    /// Default: `0xdeadbeef_cafebabe`.
    pub seed: u64,

    /// Configuration forwarded to each single-start MI registration.
    ///
    /// Default: [`GlobalMiConfig::rigid_default()`](super::config::GlobalMiConfig::rigid_default).
    pub base_config: super::config::GlobalMiConfig,
}

impl Default for MultiStartConfig {
    fn default() -> Self {
        Self {
            num_starts: 5,
            rotation_perturbation_rad: 0.2,
            translation_perturbation_mm: 15.0,
            seed: 0xdeadbeef_cafebabe,
            base_config: super::config::GlobalMiConfig::rigid_default(),
        }
    }
}

// ─── Result ───────────────────────────────────────────────────────────────────

/// Result of a multi-start Mattes MI rigid registration.
#[derive(Debug, Clone)]
pub struct MultiStartResult {
    /// 4×4 homogeneous matrix of the best-found rigid transform.
    pub matrix: [f64; 16],

    /// Mutual information achieved by the best start (higher is better).
    pub best_mi: f64,

    /// Zero-based index of the start that produced the best transform.
    pub best_start_index: usize,

    /// Final MI value for every start, in start order.
    pub per_start_mi: Vec<f64>,

    /// Total optimiser iterations (summed across all pyramid levels) for every
    /// start, in start order.
    pub per_start_iterations: Vec<usize>,

    /// Euler angles [α, β, γ] (radians) of the best-found rigid transform.
    pub best_rotation: [f32; 3],

    /// Translation [tz, ty, tx] (mm) of the best-found rigid transform.
    pub best_translation: [f32; 3],
}

// ─── Registration ─────────────────────────────────────────────────────────────

/// Multi-start wrapper around [`GlobalMiRegistration`].
///
/// Runs `config.num_starts` independent Mattes MI + RSGD optimisations, each
/// launched from a randomly perturbed copy of `initial_transform`, and returns
/// the transform with the highest final mutual information together with a
/// detailed [`MultiStartResult`].
pub struct MultiStartMiRegistration;

impl MultiStartMiRegistration {
    /// Run multi-start rigid registration.
    ///
    /// # Arguments
    ///
    /// * `fixed`             – Fixed (reference) image.
    /// * `moving`            – Moving (source) image to be aligned.
    /// * `initial_transform` – Base transform; used unmodified for start 0,
    ///                         and as the perturbation centre for subsequent
    ///                         starts.
    /// * `config`            – Multi-start and per-start optimisation config.
    ///
    /// # Panics
    ///
    /// Panics if `config.num_starts < 1`.
    pub fn register_rigid<B: AutodiffBackend>(
        fixed: &Image<B, 3>,
        moving: &Image<B, 3>,
        initial_transform: RigidTransform<B, 3>,
        config: &MultiStartConfig,
    ) -> (RigidTransform<B, 3>, MultiStartResult) {
        assert!(
            config.num_starts >= 1,
            "MultiStartConfig::num_starts must be >= 1, got {}",
            config.num_starts
        );

        let device = fixed.data().device();

        let mut best_mi = f64::NEG_INFINITY;
        let mut best_transform = initial_transform.clone();
        let mut best_start: usize = 0;

        let mut per_start_mi: Vec<f64> = Vec::with_capacity(config.num_starts);
        let mut per_start_iterations: Vec<usize> = Vec::with_capacity(config.num_starts);

        let mut rng = config.seed;

        tracing::info!(
            "MultiStartMiRegistration: beginning {} starts \
             (rot_σ = {:.4} rad, trans_σ = {:.2} mm)",
            config.num_starts,
            config.rotation_perturbation_rad,
            config.translation_perturbation_mm,
        );

        for start_idx in 0..config.num_starts {
            let start_transform = if start_idx == 0 {
                initial_transform.clone()
            } else {
                perturb_rigid_transform(&initial_transform, config, &mut rng, &device)
            };

            tracing::info!(
                "MultiStartMiRegistration: start {}/{} — launching MI registration",
                start_idx + 1,
                config.num_starts,
            );

            let (final_transform, result) = GlobalMiRegistration::register_rigid_full(
                fixed,
                moving,
                start_transform,
                &config.base_config,
            );

            let mi = result.final_mi;
            let total_iters: usize = result.iterations_per_level.iter().sum();

            tracing::info!(
                "MultiStartMiRegistration: start {}/{} — final MI = {:.6e}, \
                 total iterations = {}",
                start_idx + 1,
                config.num_starts,
                mi,
                total_iters,
            );

            per_start_mi.push(mi);
            per_start_iterations.push(total_iters);

            if mi > best_mi {
                best_mi = mi;
                best_transform = final_transform;
                best_start = start_idx;
            }
        }

        tracing::info!(
            "MultiStartMiRegistration: complete — best start index = {}, best MI = {:.6e}",
            best_start,
            best_mi,
        );

        // Extract rotation [α, β, γ] and translation [tz, ty, tx] from best.
        let rot_data = best_transform.rotation().into_data();
        let rot_slice = rot_data.as_slice::<f32>().unwrap();
        let best_rotation = [rot_slice[0], rot_slice[1], rot_slice[2]];

        let trans_data = best_transform.translation().into_data();
        let trans_slice = trans_data.as_slice::<f32>().unwrap();
        let best_translation = [trans_slice[0], trans_slice[1], trans_slice[2]];

        let matrix = super::transforms::rigid_matrix_to_homogeneous(&best_transform);

        let ms_result = MultiStartResult {
            matrix,
            best_mi,
            best_start_index: best_start,
            per_start_mi,
            per_start_iterations,
            best_rotation,
            best_translation,
        };

        (best_transform, ms_result)
    }
}

// ─── Private Helpers ──────────────────────────────────────────────────────────

/// Draw one standard-normal sample using the Box–Muller transform.
///
/// The generator state `rng` is advanced twice: once to produce the uniform
/// variate u₁ and once for u₂. Both are mapped to (0, 1] via the 53
/// most-significant bits of the 64-bit LCG output, with an additive floor of
/// `1e-30` to keep u₁ away from zero and avoid `ln(0)`.
///
/// # LCG constants (Knuth–Lewis / MMIX)
///
/// ```text
/// a = 6_364_136_223_846_793_005
/// c = 1_442_695_040_888_963_407
/// ```
fn lcg_gaussian(rng: &mut u64) -> f64 {
    // Advance for u1.
    *rng = rng
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let u1 = (*rng >> 11) as f64 / (1u64 << 53) as f64 + 1e-30;

    // Advance for u2.
    *rng = rng
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let u2 = (*rng >> 11) as f64 / (1u64 << 53) as f64 + 1e-30;

    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

/// Return a copy of `transform` with independent Gaussian noise added to every
/// rotation angle and translation component.
///
/// The center of rotation is preserved unchanged.
fn perturb_rigid_transform<B: AutodiffBackend>(
    transform: &RigidTransform<B, 3>,
    config: &MultiStartConfig,
    rng: &mut u64,
    device: &B::Device,
) -> RigidTransform<B, 3> {
    // Read current rotation [α, β, γ].
    let rot_data = transform.rotation().into_data();
    let rot_slice = rot_data.as_slice::<f32>().unwrap();

    // Read current translation [tz, ty, tx].
    let trans_data = transform.translation().into_data();
    let trans_slice = trans_data.as_slice::<f32>().unwrap();

    // Perturb rotation components.
    let new_rot: [f32; 3] = [
        rot_slice[0] + (lcg_gaussian(rng) * config.rotation_perturbation_rad) as f32,
        rot_slice[1] + (lcg_gaussian(rng) * config.rotation_perturbation_rad) as f32,
        rot_slice[2] + (lcg_gaussian(rng) * config.rotation_perturbation_rad) as f32,
    ];

    // Perturb translation components.
    let new_trans: [f32; 3] = [
        trans_slice[0] + (lcg_gaussian(rng) * config.translation_perturbation_mm) as f32,
        trans_slice[1] + (lcg_gaussian(rng) * config.translation_perturbation_mm) as f32,
        trans_slice[2] + (lcg_gaussian(rng) * config.translation_perturbation_mm) as f32,
    ];

    let new_rotation = Tensor::<B, 1>::from_data(TensorData::from(new_rot), device);
    let new_translation = Tensor::<B, 1>::from_data(TensorData::from(new_trans), device);

    // Preserve center of rotation unchanged.
    let center = transform.center();

    RigidTransform::new(new_translation, new_rotation, center)
}
