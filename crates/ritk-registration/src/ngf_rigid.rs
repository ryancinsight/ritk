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
use ritk_core::image::Image;
use ritk_transform::RigidTransform;

use crate::metric::{Metric, NormalizedGradientField};
use crate::optimizer::{CmaEsConfig, CmaEsOptimizer, HistoryPolicy, PopulationEval};

/// Configuration for [`register_rigid_ngf`].
#[derive(Debug, Clone)]
pub struct NgfRigidConfig {
    /// Half-range of the per-axis rotation search [rad] (the normalized `±1`
    /// parameter maps to `±rotation_range_rad`).
    pub rotation_range_rad: f64,
    /// Half-range of the per-axis translation search [mm].
    pub translation_range_mm: f64,
    /// CMA-ES optimizer configuration.
    pub cma: CmaEsConfig,
}

impl Default for NgfRigidConfig {
    fn default() -> Self {
        Self {
            rotation_range_rad: std::f64::consts::FRAC_PI_4, // ±45°
            translation_range_mm: 60.0,
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
    /// Recovered ZYX Euler rotation [rad].
    pub rotation_rad: [f64; 3],
    /// Recovered translation `[tz, ty, tx]` [mm].
    pub translation_mm: [f64; 3],
    /// NGF edge-alignment at the recovered pose (`−best_f`, in `[0, 1]`).
    pub best_ngf: f64,
    /// CMA-ES generations run.
    pub generations: usize,
}

/// Build a [`RigidTransform`] from normalized CMA-ES parameters
/// `[α_n, β_n, γ_n, tz_n, ty_n, tx_n] ∈ [−1, 1]⁶`.
fn build_rigid<B: Backend>(
    params: &[f64],
    rot_scale: f64,
    trans_scale: f64,
    device: &B::Device,
) -> RigidTransform<B, 3> {
    let rot = Tensor::<B, 1>::from_data(
        TensorData::from([
            (params[0] * rot_scale) as f32,
            (params[1] * rot_scale) as f32,
            (params[2] * rot_scale) as f32,
        ]),
        device,
    );
    let trans = Tensor::<B, 1>::from_data(
        TensorData::from([
            (params[3] * trans_scale) as f32,
            (params[4] * trans_scale) as f32,
            (params[5] * trans_scale) as f32,
        ]),
        device,
    );
    let center = Tensor::<B, 1>::zeros([3], device);
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
    config: &NgfRigidConfig,
) -> (RigidTransform<B, 3>, NgfRigidResult) {
    let device = fixed.data().device();
    let metric = NormalizedGradientField::new();
    let rot_scale = config.rotation_range_rad;
    let trans_scale = config.translation_range_mm;

    let x0: [f64; 6] = [
        initial_rotation[0] / rot_scale,
        initial_rotation[1] / rot_scale,
        initial_rotation[2] / rot_scale,
        initial_translation[0] / trans_scale,
        initial_translation[1] / trans_scale,
        initial_translation[2] / trans_scale,
    ];

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
        let t = build_rigid::<B>(params, rot_scale, trans_scale, &device);
        let loss = metric.forward(fixed, moving, &t);
        loss.into_data()
            .as_slice::<f32>()
            .expect("ngf loss tensor must be contiguous f32")[0] as f64
    };

    let cma = CmaEsOptimizer::new(config.cma.clone()).run(obj, &x0);

    let final_transform = build_rigid::<B>(&cma.best_x, rot_scale, trans_scale, &device);

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

    let b = &cma.best_x;
    let result = NgfRigidResult {
        matrix,
        rotation_rad: [b[0] * rot_scale, b[1] * rot_scale, b[2] * rot_scale],
        translation_mm: [b[3] * trans_scale, b[4] * trans_scale, b[5] * trans_scale],
        best_ngf: -cma.best_f,
        generations: cma.generations,
    };
    (final_transform, result)
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
    /// slab is shifted 3 voxels in x with OPPOSITE contrast). x is index 2 of the
    /// `[tz, ty, tx]` translation; the recovered magnitude must be ≈ 3 mm. (y, z
    /// and the x-axis rotation are unconstrained by an x-only slab, so only tx is
    /// asserted.)
    #[test]
    fn recovers_known_translation_cross_modal() {
        let (d, h, w) = (6usize, 24usize, 24usize);
        let fixed = image3d(x_slab(d, h, w, 12, 1.0), [d, h, w]);
        let moving = image3d(x_slab(d, h, w, 15, -1.0), [d, h, w]); // +3 in x, inverted

        let cfg = NgfRigidConfig {
            rotation_range_rad: 0.15,
            translation_range_mm: 10.0,
            cma: CmaEsConfig {
                max_generations: 150,
                ..NgfRigidConfig::default().cma
            },
        };
        let (_t, res) = register_rigid_ngf(&fixed, &moving, [0.0; 3], [0.0; 3], &cfg);

        assert!(
            (res.translation_mm[2].abs() - 3.0).abs() < 1.5,
            "recovered tx {} mm, expected ≈ ±3 mm",
            res.translation_mm[2]
        );
        assert!(res.best_ngf > 0.05, "NGF too low at recovered pose: {}", res.best_ngf);
    }
}
