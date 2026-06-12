//! NGF-driven rigid registration: gradient-free CMA-ES over the Normalized
//! Gradient Fields metric ([`crate::metric::NormalizedGradientField`]).
//!
//! Cross-modal (CT↔MRI) rigid alignment from identity is unreliable for
//! intensity mutual information — a near-uniform CT brain interior gives almost
//! no MI signal, so gradient-based MI gets trapped. NGF aligns edge *orientation*
//! instead, which co-locates across modalities, and CMA-ES (derivative-free)
//! escapes the local optima that defeat gradient descent. This module pairs the
//! two: the same normalized `[−1,1]⁶` rigid parameterization and boundary
//! penalty as [`crate::CmaMiRegistration`], but with NGF as the objective.
//!
//! NGF needs no autodiff (it reads gradients on the host), so this runs on a
//! plain [`Backend`] — pre-mask the images (e.g. to a brain mask) to focus the
//! metric on the shared rigid structure.

use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use ritk_image::Image;
use ritk_transform::RigidTransform;

use ritk_filter::BinShrinkImageFilter;

use crate::classical::compute_center_of_mass;
use crate::metric::ngf::NgfFixedPrep;
use crate::optimizer::{CmaEsConfig, CmaEsOptimizer, HistoryPolicy, PopulationEval};

/// Configuration for [`register_rigid_ngf`].
#[derive(Debug, Clone)]
pub struct NgfRigidConfig {
    /// Half-range of the per-axis rotation search [rad] (the normalized `±1`
    /// parameter maps to `±rotation_range_rad`).
    pub rotation_range_rad: f64,
    /// Half-range of the per-axis translation search [mm].
    pub translation_range_mm: f64,
    /// Optional brain-centroid Gaussian weighting of the NGF metric. `Some(frac)`
    /// weights each masked voxel by `exp(−‖x−c‖²/(2σ²))`, `σ = frac · r_rms`
    /// (RMS mask radius, physical units), suppressing the high-gradient skull/scalp
    /// rim so the optimiser aligns deep structure (ventricles, deep gray) instead
    /// of the periphery. `None` keeps the uniform Haber–Modersitzki average.
    /// `frac ≈ 0.7` ≈ σ at ⅓ the outer brain radius (multimodal-edge convention).
    pub center_weight_sigma_frac: Option<f64>,
    /// Optional stochastic-sampling count. `Some(s)` estimates NGF on a fixed
    /// deterministic subset of `s` mask voxels per evaluation instead of the full
    /// grid — the dominant speed lever (elastix-style), trading a bounded-variance
    /// estimate for orders-of-magnitude fewer resample/gradient ops. `None`
    /// evaluates densely. Typical: a few thousand (2048–8192).
    pub sample_count: Option<usize>,
    /// CMA-ES optimizer configuration.
    pub cma: CmaEsConfig,
}

impl Default for NgfRigidConfig {
    fn default() -> Self {
        Self {
            rotation_range_rad: std::f64::consts::FRAC_PI_4, // ±45°
            translation_range_mm: 60.0,
            center_weight_sigma_frac: None,
            sample_count: None,
            cma: CmaEsConfig {
                sigma0: 0.3,
                lambda: 0, // auto: 4 + ⌊3 ln n⌋
                max_generations: 300,
                sigma_tol: 1e-8,
                // −NGF ∈ [−1, 0]; disable the f-tolerance stop (as the MI path does).
                ftol: f64::NEG_INFINITY,
                seed: 0xcafe_babe_dead_beef,
                parallel_population: PopulationEval::Sequential,
                record_history: HistoryPolicy::Discard,
            },
        }
    }
}

/// Result of [`register_rigid_ngf`].
#[derive(Debug, Clone)]
pub struct NgfRigidResult {
    /// Recovered rigid transform as a row-major `4×4` homogeneous matrix
    /// (ritk `[z, y, x]` convention; same layout as `CmaMiResult::matrix`).
    pub matrix: [f64; 16],
    /// Recovered Euler rotation `[α, β, γ]` [rad] about the x, y, z world axes
    /// (`RigidTransform` `R = R_z R_y R_x`).
    pub rotation_rad: [f64; 3],
    /// Recovered translation `[tx, ty, tz]` [mm] in world (LPS) space.
    pub translation_mm: [f64; 3],
    /// NGF edge-alignment at the recovered pose (`−best_f`, in `[0, 1]`).
    pub best_ngf: f64,
    /// CMA-ES generations run.
    pub generations: usize,
}

/// Build a [`RigidTransform`] from normalized CMA-ES residual parameters
/// `[α_n, β_n, γ_n, tz_n, ty_n, tx_n] ∈ [−1, 1]⁶`, composed onto a fixed base
/// pose `(base_rot, base_trans)` and rotating about `center` (the fixed-image
/// centroid). Searching a tight residual about a centroid pre-alignment — rather
/// than a wide box about the world origin — keeps rotation and translation
/// decoupled (rotation about the brain centre induces no spurious translation),
/// so the optimiser stays in the correct basin instead of trading rotation
/// against a large compensating translation.
#[allow(clippy::too_many_arguments)]
fn build_rigid<B: Backend>(
    params: &[f64],
    rot_scale: f64,
    trans_scale: f64,
    base_rot: [f64; 3],
    base_trans: [f64; 3],
    center: [f64; 3],
    device: &B::Device,
) -> RigidTransform<B, 3> {
    let rot = Tensor::<B, 1>::from_data(
        TensorData::from([
            (base_rot[0] + params[0] * rot_scale) as f32,
            (base_rot[1] + params[1] * rot_scale) as f32,
            (base_rot[2] + params[2] * rot_scale) as f32,
        ]),
        device,
    );
    let trans = Tensor::<B, 1>::from_data(
        TensorData::from([
            (base_trans[0] + params[3] * trans_scale) as f32,
            (base_trans[1] + params[4] * trans_scale) as f32,
            (base_trans[2] + params[5] * trans_scale) as f32,
        ]),
        device,
    );
    let center = Tensor::<B, 1>::from_data(
        TensorData::from([center[0] as f32, center[1] as f32, center[2] as f32]),
        device,
    );
    RigidTransform::<B, 3>::new(trans, rot, center)
}

/// Register `moving` to `fixed` rigidly by maximizing NGF with CMA-ES, seeded at
/// `initial_rotation` [rad] / `initial_translation` [mm].
///
/// Returns the recovered [`RigidTransform`] (mapping fixed → moving space) and a
/// [`NgfRigidResult`]. Run on whatever resolution you pass — downsample first for
/// a fast global search; the world-space transform applies at full resolution.
pub fn register_rigid_ngf<B: Backend>(
    fixed: &Image<B, 3>,
    moving: &Image<B, 3>,
    initial_rotation: [f64; 3],
    initial_translation: [f64; 3],
    fixed_mask: Option<&Image<B, 3>>,
    config: &NgfRigidConfig,
) -> (RigidTransform<B, 3>, NgfRigidResult) {
    let device = fixed.data().device();
    let rot_scale = config.rotation_range_rad;
    let trans_scale = config.translation_range_mm;

    // Restrict NGF to the (brain/skull) mask in fixed-image C-order. Without it,
    // cross-modal NGF locks onto the scalp / scanner-bed / FOV edges and diverges.
    let mask_bool: Option<Vec<bool>> = fixed_mask.map(|mask| {
        let n: usize = mask.shape().iter().product();
        mask.data()
            .clone()
            .reshape([n])
            .into_data()
            .to_vec::<f32>()
            .expect("mask to f32 host vec")
            .into_iter()
            .map(|v| v > 0.5)
            .collect()
    });

    // Optional brain-centroid Gaussian weight field (physical-space, anisotropy
    // aware) that de-emphasises the skull/scalp rim. Built once over the fixed
    // mask; reused for every CMA-ES objective evaluation.
    let weights: Option<Vec<f32>> = config.center_weight_sigma_frac.map(|frac| {
        crate::metric::ngf::center_gaussian_weight_field(
            &fixed.shape(),
            mask_bool.as_deref(),
            &fixed.spacing().to_array(),
            frac,
        )
    });

    // Precompute the fixed-image NGF state ONCE (grid world points, fixed
    // gradient field, η_F, mask, weights). Every CMA-ES objective evaluation
    // reuses it, so only the moving resample + moving gradient run per step.
    // With `sample_count`, only a fixed random voxel subset is evaluated.
    let prep = match config.sample_count {
        Some(s) => NgfFixedPrep::new_sampled(fixed, mask_bool.as_deref(), weights.as_deref(), s),
        None => NgfFixedPrep::new(fixed, mask_bool.as_deref(), weights.as_deref()),
    };

    // Rotate about the fixed-image centroid so the residual search decouples
    // rotation from translation (rotation about the world origin would couple a
    // few degrees into tens of mm for a brain centred far from the origin). The
    // caller's `initial_*` become the fixed base pose; the CMA-ES residual is
    // searched from zero within the configured (tight) ranges.
    let center = compute_center_of_mass(fixed);
    let x0 = [0.0_f64; 6];

    // Objective: −NGF on the inner backend, with the same box + soft-boundary
    // penalty the CMA-MI path uses to suppress out-of-FOV corner optima.
    let obj = |params: &[f64]| -> f64 {
        if params.iter().any(|&p| p.abs() > 1.0) {
            return 10.0;
        }
        let boundary: f64 = params
            .iter()
            .map(|&p| {
                let excess = p.abs() - 0.85;
                if excess > 0.0 {
                    excess * excess * 100.0
                } else {
                    0.0
                }
            })
            .sum();
        if boundary > 0.0 {
            return boundary;
        }
        let t = build_rigid::<B>(
            params,
            rot_scale,
            trans_scale,
            initial_rotation,
            initial_translation,
            center,
            &device,
        );
        // −NGF (optionally center-weighted) reusing the precomputed fixed state.
        -f64::from(prep.eval(moving, &t))
    };

    let cma = CmaEsOptimizer::new(config.cma.clone()).run(obj, &x0);

    let final_transform = build_rigid::<B>(
        &cma.best_x,
        rot_scale,
        trans_scale,
        initial_rotation,
        initial_translation,
        center,
        &device,
    );

    // Pack the 3×4 rigid matrix into a 4×4 homogeneous [f64; 16] (matches the
    // CmaMiResult convention used downstream).
    let m_data = final_transform.matrix().to_data();
    let m_slice = m_data
        .as_slice::<f32>()
        .expect("rigid matrix tensor must be contiguous f32");
    let mut matrix = [0.0f64; 16];
    for (i, &v) in m_slice.iter().enumerate() {
        matrix[i] = f64::from(v);
    }
    matrix[15] = 1.0;

    // Recovered pose = base + residual (the residual is the CMA-ES solution).
    let b = &cma.best_x;
    let result = NgfRigidResult {
        matrix,
        rotation_rad: [
            initial_rotation[0] + b[0] * rot_scale,
            initial_rotation[1] + b[1] * rot_scale,
            initial_rotation[2] + b[2] * rot_scale,
        ],
        translation_mm: [
            initial_translation[0] + b[3] * trans_scale,
            initial_translation[1] + b[4] * trans_scale,
            initial_translation[2] + b[5] * trans_scale,
        ],
        best_ngf: -cma.best_f,
        generations: cma.generations,
    };
    (final_transform, result)
}

/// One level of an [`register_rigid_ngf_multires`] Gaussian-pyramid schedule.
#[derive(Debug, Clone)]
pub struct NgfPyramidLevel {
    /// Per-axis bin-shrink factor for this level (`[1, s, s]` keeps thick slices
    /// and downsamples in-plane). Coarser levels (larger factors) run the global
    /// search cheaply; finer levels refine where the NGF optimum is sharp.
    pub shrink: [usize; 3],
    /// Per-level rigid-search configuration. Coarse levels use a wide residual
    /// range; fine levels tighten it (and need fewer generations from a good seed).
    pub config: NgfRigidConfig,
}

/// Multi-resolution coarse-to-fine NGF rigid registration. Each level
/// bin-shrinks `fixed`/`moving`/`mask` (averaging — anti-aliasing low-pass),
/// runs [`register_rigid_ngf`] seeded from the previous level's recovered pose,
/// and passes the result down. Coarse levels remove high-frequency edge texture
/// so the objective is smoother and the global basin is found cheaply; fine
/// levels then lock onto the sharp full-resolution optimum that a single coarse
/// pass blurs away. Bin-shrink preserves world coordinates, so the recovered
/// transform applies at full resolution.
///
/// `levels` must be ordered COARSE → FINE and be non-empty.
pub fn register_rigid_ngf_multires<B: Backend>(
    fixed: &Image<B, 3>,
    moving: &Image<B, 3>,
    initial_rotation: [f64; 3],
    initial_translation: [f64; 3],
    fixed_mask: Option<&Image<B, 3>>,
    levels: &[NgfPyramidLevel],
) -> (RigidTransform<B, 3>, NgfRigidResult) {
    assert!(!levels.is_empty(), "multires schedule must have ≥ 1 level");
    let mut rot = initial_rotation;
    let mut trans = initial_translation;
    let mut out: Option<(RigidTransform<B, 3>, NgfRigidResult)> = None;

    for (li, level) in levels.iter().enumerate() {
        let shrink = BinShrinkImageFilter::new(level.shrink.to_vec());
        let f = shrink.apply(fixed);
        let m = shrink.apply(moving);
        let mask = fixed_mask.map(|mk| shrink.apply(mk));
        let (t, res) = register_rigid_ngf(&f, &m, rot, trans, mask.as_ref(), &level.config);
        // Warm-start the next (finer) level from this pose.
        rot = res.rotation_rad;
        trans = res.translation_mm;
        let tr = res.matrix[0] + res.matrix[5] + res.matrix[10];
        let ang = (((tr - 1.0) / 2.0).clamp(-1.0, 1.0)).acos().to_degrees();
        tracing::debug!(
            level = li,
            shrink = ?level.shrink,
            rot_deg = ang,
            best_ngf = res.best_ngf,
            rot_euler_deg = ?rot.map(|r| r.to_degrees()),
            t_mm = ?trans,
            "ngf multires level"
        );
        out = Some((t, res));
    }
    out.expect("non-empty schedule yields a result")
}

/// Default 3-level head CT↔MR schedule for a COM-pre-aligned same-patient pair:
/// shrink 4 → 2 → 1 (full-resolution finest). Rotation ranges are deliberately
/// TIGHT (±12°→±5°): a same-patient head pose differs by little, and the cross-
/// modal NGF landscape develops spurious far optima at large rotation/translation
/// that a wide search locks onto (confirmed: a ±30° coarse level diverged to ~25°).
/// The through-plane axis is never shrunk; the finest level is full resolution so
/// the NGF optimum is sharp (a heavily downsampled grid blurs the very edges the
/// metric needs). Tune per data.
#[must_use]
pub fn default_ngf_pyramid(center_weight_sigma_frac: Option<f64>) -> Vec<NgfPyramidLevel> {
    let level = |shrink_xy: usize, rot_deg: f64, trans_mm: f64, gens: usize, samples: usize| {
        NgfPyramidLevel {
            shrink: [1, shrink_xy, shrink_xy],
            config: NgfRigidConfig {
                rotation_range_rad: rot_deg.to_radians(),
                translation_range_mm: trans_mm,
                center_weight_sigma_frac,
                // Stochastic sampling keeps every level cheap; finer levels use
                // more samples to resolve the sharper optimum.
                sample_count: Some(samples),
                cma: CmaEsConfig {
                    max_generations: gens,
                    // The λ candidates per generation are independent NGF
                    // evaluations — evaluate them across threads.
                    parallel_population: PopulationEval::Parallel,
                    ..NgfRigidConfig::default().cma
                },
            },
        }
    };
    vec![
        level(4, 12.0, 25.0, 150, 5000),  // coarse: tight rot, residual-bounded trans
        level(2, 8.0, 15.0, 100, 10000),  // medium: refine
        level(1, 5.0, 8.0, 60, 20000),    // fine: full resolution, sharp optimum
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Shape, TensorData};
    use burn_ndarray::NdArray;
    use ritk_core::spatial::{Direction, Point, Spacing};

    type B = NdArray<f32>;

    fn image3d(data: Vec<f32>, shape: [usize; 3]) -> Image<B, 3> {
        let device = Default::default();
        let tensor = Tensor::from_data(TensorData::new(data, Shape::new(shape)), &device);
        Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        )
    }

    /// A slab whose edge is at `edge` along x (the fastest, w-axis), constant in
    /// y,z; `sign` sets the contrast.
    fn x_slab(d: usize, h: usize, w: usize, edge: usize, sign: f32) -> Vec<f32> {
        let mut v = vec![0.0f32; d * h * w];
        for z in 0..d {
            for y in 0..h {
                for x in 0..w {
                    v[(z * h + y) * w + x] = if x < edge { 0.0 } else { sign };
                }
            }
        }
        v
    }

    /// CMA-ES + NGF recovers a known x-translation across modalities (the moving
    /// slab is shifted 3 voxels in x with OPPOSITE contrast). The registrar
    /// rotates about the fixed centroid, so the `(R, t)` decomposition of an
    /// alignment is not unique — the physical edge displacement is
    /// `t + (I − R)·c`, equal to the raw `t` only when `R = I`. We therefore
    /// assert the decomposition-INVARIANT quantity: the net x-displacement the
    /// recovered matrix applies to a point on the fixed edge (≈ +3 mm), plus a
    /// high NGF confirming the slabs actually align.
    #[test]
    fn recovers_known_translation_cross_modal() {
        let (d, h, w) = (6usize, 24usize, 24usize);
        let fixed = image3d(x_slab(d, h, w, 12, 1.0), [d, h, w]);
        let moving = image3d(x_slab(d, h, w, 15, -1.0), [d, h, w]); // +3 in x, inverted

        // Near-zero rotation range isolates TRANSLATION recovery: an x-slab is
        // rotation-degenerate (constant in y,z), so a free rotation about the
        // centroid only adds an ambiguous degree of freedom. Constraining it keeps
        // this a clean translation test.
        let cfg = NgfRigidConfig {
            rotation_range_rad: 1e-3,
            translation_range_mm: 10.0,
            cma: CmaEsConfig {
                max_generations: 200,
                ..NgfRigidConfig::default().cma
            },
            ..Default::default()
        };
        let (_t, res) = register_rigid_ngf(&fixed, &moving, [0.0; 3], [0.0; 3], None, &cfg);

        // Net x-shift the matrix (row-major [z,y,x]) applies at the edge point
        // (z,y,x)=(3,12,12): x' = m[8]·z + m[9]·y + m[10]·x + m[11].
        let m = res.matrix;
        let (z, y, x) = (3.0_f64, 12.0_f64, 12.0_f64);
        let net_x = (m[8] * z + m[9] * y + m[10] * x + m[11]) - x;
        assert!(
            (net_x.abs() - 3.0).abs() < 1.5,
            "net x-displacement {net_x} mm, expected ≈ ±3 mm"
        );
        // Mean NGF over the whole volume is modest: the slab is mostly flat
        // (zero-gradient) voxels that contribute ~0; only the single edge plane
        // scores ~1. A low-but-positive value confirms the edges co-locate.
        assert!(
            res.best_ngf > 0.05,
            "NGF too low at recovered pose: {}",
            res.best_ngf
        );
    }
}
