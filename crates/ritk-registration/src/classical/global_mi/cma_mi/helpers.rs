// â”€â”€â”€ Private helpers for CMA-ES MI registration â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
//
// Extracted from `registration.rs` to keep the public API surface in a
// focused file while the heavy lifting lives here.

use ritk_core::image::Image;
use ritk_filter::pyramid::MultiResolutionPyramid;
use ritk_filter::GaussianSigma;
use ritk_image::generate_random_points;
use ritk_image::tensor::{Backend, Tensor};
use ritk_transform::RigidTransform;

use super::super::transforms::estimate_intensity_range;
use super::config::CmaMiConfig;
use crate::metric::{Metric, MutualInformation, MutualInformationVariant};
use crate::optimizer::{CmaEsConfig, CmaEsOptimizer};

/// Strip autodiff from a 3-D image, returning an equivalent image on the inner
/// non-autodiff backend.
///
/// CMA-ES never calls `.backward()`. Evaluating the MI metric on
/// `Image<Autodiff<B>, 3>` silently builds an autodiff tape on every objective
/// call, wasting 2â€“5Ã— CPU time. Converting to the inner backend before the
/// CMA-ES loop eliminates this overhead entirely.
/// Build a `MutualInformation` metric on the selected Coeus backend.
///
/// The Parzen window width is set to `bin_width = (maxâˆ’min)/num_bins`, which
/// is appropriate for both Mattes MI and NMI variants.
///
/// When `mask_points` is `Some`, the metric uses masked joint histogram evaluation
/// (only supplied world-space foreground points contribute) instead of stochastic
/// uniform sampling.
///
/// When `moving_range` is `Some((min, max))`, the moving-image axis of the joint
/// histogram uses its own independent range (elastix-style), giving the moving
/// image the full `num_bins` resolution instead of sharing the combined range.
pub(super) fn build_metric<IB: Backend>(
    variant: MutualInformationVariant,
    num_bins: usize,
    min_int: f32,
    max_int: f32,
    sampling_percentage: f32,
    mask_points: Option<Tensor<f32, IB>>,
    // NEW: separate moving-image range (None â†’ use combined range, backward compat)
    moving_range: Option<(f32, f32)>,
    device: &IB,
) -> MutualInformation<IB> {
    const MIN_HISTOGRAM_BIN_WIDTH: f32 = 1e-6;
    let bin_width = (max_int - min_int).max(MIN_HISTOGRAM_BIN_WIDTH) / num_bins as f32;
    let mi = if let Some((mov_min, mov_max)) = moving_range {
        MutualInformation::new_with_separate_ranges(
            variant, num_bins, min_int, max_int, mov_min, mov_max, device,
        )
        .with_sampling(sampling_percentage)
    } else {
        MutualInformation::new(variant, num_bins, min_int, max_int, bin_width, device)
            .with_sampling(sampling_percentage)
    };
    if let Some(pts) = mask_points {
        mi.with_fixed_mask_points(pts)
    } else {
        mi
    }
}

/// Execute a single CMA-ES level with autodiff-stripped images.
///
/// Builds a one-level pyramid at `per_axis` shrink factors, strips autodiff,
/// and minimises `âˆ’MI` over the normalised 6-DOF parameter space.
///
/// Returns the raw [`crate::optimizer::CmaEsResult`] whose `best_x` is in normalised `[âˆ’1,1]â¶`
/// space; the caller is responsible for denormalising.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_cma_level<B: Backend>(
    fixed: &Image<f32, B, 3>,
    moving: &Image<f32, B, 3>,
    config: &CmaMiConfig,
    per_axis: &[usize; 3],
    sigma_mm: GaussianSigma,
    cma_sigma0: f64,
    max_generations: usize,
    lambda: usize,
    ipop_restarts: usize,
    rot_scale: f64,
    trans_scale: f64,
    center_arr: [f32; 3],
    x_init: &[f64],
    // Optional brain mask in fixed-image space (autodiff backend, same shape as `fixed`).
    // When Some, downsampled to pyramid level and used to restrict MI to foreground voxels.
    fixed_mask: Option<&Image<f32, B, 3>>,
) -> crate::optimizer::CmaEsResult {
    // â”€â”€ Build pyramid â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    // P1-01: stack-allocated per-axis arrays â€” no `Vec<usize>` / `Vec<f64>`
    // per-level allocation, no `Vec<Vec<_>>` outer container.
    let shrink_factors: Vec<[usize; 3]> = vec![[per_axis[0], per_axis[1], per_axis[2]]];
    // Zero smoothing for axes with no downsampling (shrink â‰¤ 1): applying sigma > 0
    // on an axis we are NOT downsampling wastes z-information that the thin-slab
    // preset specifically preserves via anisotropic shrink factors.
    let smoothing_sigmas: Vec<[f64; 3]> = vec![[
        if per_axis[0] <= 1 {
            0.0
        } else {
            sigma_mm.get()
        },
        if per_axis[1] <= 1 {
            0.0
        } else {
            sigma_mm.get()
        },
        if per_axis[2] <= 1 {
            0.0
        } else {
            sigma_mm.get()
        },
    ]];

    let fixed_pyr = MultiResolutionPyramid::new(fixed, &shrink_factors, &smoothing_sigmas);
    let moving_pyr = MultiResolutionPyramid::new(moving, &shrink_factors, &smoothing_sigmas);
    let fixed_c = fixed_pyr.get_level(0).clone();
    let moving_c = moving_pyr.get_level(0).clone();

    tracing::info!(
        "CmaMiRegistration: level â€” fixed {:?}, moving {:?}",
        fixed_c.shape(),
        moving_c.shape(),
    );

    // â”€â”€ Intensity range â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    let (min_f, max_f) = estimate_intensity_range(&fixed_c);
    let (min_m, max_m) = estimate_intensity_range(&moving_c);
    let min_int = min_f.min(min_m);
    let max_int = max_f.max(max_m);

    // OOB note: using COMBINED intensity range (not separate per-image ranges) for
    // the CMA-ES cold-start search. With separate moving range, the zero-pad
    // interpolator returns 0.0 for OOB samples; 0.0 < MRI min (â‰ˆ 2 for RIRE) so it
    // clamps to MRI bin 0, creating a false correlation between CT air (also bin 0 in
    // separate CT range) and OOB samples. With the combined range, OOB=0.0 maps to
    // a middle bin (~11 for RIRE), which is different from CT air (bin 0), reducing
    // false correlations at large-displacement boundary positions.
    let separate_moving_range: Option<(f32, f32)> = None;

    // â”€â”€ Strip autodiff â€” eliminate tape overhead in CMA-ES loop â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    let fixed_inner = fixed_c;
    let moving_inner = moving_c;
    let inner_device = fixed_inner.data().device();

    // â”€â”€ Brain mask / Pre-sampling: extract or generate sample points once per level â”€â”€â”€â”€
    // This avoids random noise on every objective function evaluation and enables
    // caching of the fixed-image Parzen weights for both masked and sampled paths.
    let mask_world_points: Option<Tensor<f32, B>> = if let Some(mask) = fixed_mask {
        let mask_shrink: Vec<[usize; 3]> = vec![[per_axis[0], per_axis[1], per_axis[2]]];
        let mask_smooth: Vec<[f64; 3]> = vec![[0.0, 0.0, 0.0]]; // no smoothing
        let mask_pyr = MultiResolutionPyramid::new(mask, &mask_shrink, &mask_smooth);
        let mask_c = mask_pyr.get_level(0).clone();
        let mask_inner = mask_c;
        tracing::info!(
            "CmaMiRegistration: mask at level â€” shape {:?}",
            mask_inner.shape()
        );
        Some(extract_foreground_world_points(
            &fixed_inner,
            &mask_inner,
            config.sampling_percentage,
        ))
    } else if config.sampling_percentage < 1.0 {
        let total_voxels = fixed_inner.shape().iter().product::<usize>();
        let num_samples = ((total_voxels as f32 * config.sampling_percentage) as usize).max(32);
        let fixed_indices =
            generate_random_points(fixed_inner.shape(), num_samples, &inner_device);
        Some(fixed_inner.index_to_world_tensor(fixed_indices))
    } else {
        None
    };

    let metric_sampling_pct = if mask_world_points.is_some() {
        1.0
    } else {
        config.sampling_percentage
    };

    // â”€â”€ Build metric on inner backend â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    let metric = build_metric::<B>(
        config.mi_variant,
        config.num_mi_bins,
        min_int,
        max_int,
        metric_sampling_pct,
        mask_world_points,
        separate_moving_range,
        &inner_device,
    );

    // â”€â”€ Objective closure â€” all tensors on inner (non-autodiff) backend â”€â”€â”€â”€â”€â”€â”€
    let obj = move |params: &[f64]| -> f64 {
        // Hard penalty for parameters outside the normalised box [âˆ’1, 1]â¶.
        if params.iter().any(|&p| p.abs() > 1.0) {
            return 10.0;
        }
        // Soft boundary penalty: discourage the CMA-ES from converging to
        // corners of the search box. When the transform maps most fixed-image
        // voxels outside the moving-image FOV (large translations), the zero-pad
        // interpolator returns 0.0 for out-of-bounds samples, which can create
        // a false MI maximum near |p_i| â‰ˆ 1. Adding a quadratic penalty for
        // |p_i| > 0.85 (= 85% of the search range, e.g. 51 mm for a 60 mm
        // range) strongly suppresses this artefact without restricting the
        // interior of the search space where the true maximum lies.
        let boundary_penalty: f64 = params
            .iter()
            .map(|&p| {
                let excess = p.abs() - 0.85;
                if excess > 0.0 {
                    excess * excess * 100.0 // 2.25 at |p|=1.0, >> any real MI value
                } else {
                    0.0
                }
            })
            .sum::<f64>();
        if boundary_penalty > 0.0 {
            return boundary_penalty; // short-circuit before MI computation
        }

        let alpha = (params[0] * rot_scale) as f32;
        let beta = (params[1] * rot_scale) as f32;
        let gamma = (params[2] * rot_scale) as f32;
        let tz = (params[3] * trans_scale) as f32;
        let ty = (params[4] * trans_scale) as f32;
        let tx = (params[5] * trans_scale) as f32;

        let rotation =
            Tensor::<f32, B>::from_slice_on([3], &[alpha, beta, gamma], &inner_device);
        let translation = Tensor::<f32, B>::from_slice_on([3], &[tz, ty, tx], &inner_device);
        let center = Tensor::<f32, B>::from_slice_on([3], &center_arr, &inner_device);
        let transform = RigidTransform::<B, 3>::new(translation, rotation, center);

        let loss = metric.forward(&fixed_inner, &moving_inner, &transform);
        loss.into_data()
            .as_slice::<f32>()
            .expect("loss value tensor data must be contiguous")[0] as f64
    };

    // â”€â”€ CMA-ES config for this level â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    // Inherit shared settings (seed, sigma_tol, ftol, record_history) from the
    // top-level config; override per-level parameters.
    let level_cfg = CmaEsConfig {
        sigma0: cma_sigma0,
        lambda,
        max_generations,
        ..config.cma_config.clone()
    };
    tracing::info!(
        "CmaMiRegistration: CMA-ES (max_gen={}, sigma0={:.3}, lambda={}, ipop={})",
        level_cfg.max_generations,
        level_cfg.sigma0,
        level_cfg.lambda,
        ipop_restarts,
    );

    if ipop_restarts > 0 {
        CmaEsOptimizer::new(level_cfg).run_ipop(obj, x_init, ipop_restarts)
    } else {
        CmaEsOptimizer::new(level_cfg).run(obj, x_init)
    }
}

/// Extract world-space coordinates of foreground voxels from a brain mask.
///
/// Given a downsampled binary mask (same shape as `fixed_inner` at this pyramid
/// level), returns a `[N, 3]` tensor of world-space coordinates for voxels where
/// `mask > 0.5`. The returned point count is capped at
/// `ceil(total_voxels * sampling_pct)` (with a floor of 32) via stride-based
/// deterministic sub-sampling so that MI evaluation time stays comparable to
/// the unmasked stochastic-sampling path.
///
/// # Fallback
/// If the mask is empty at this pyramid level (all zeros after downsampling),
/// a warning is emitted and a uniform stride-sampled set of all voxels is
/// returned so registration can continue.
pub(super) fn extract_foreground_world_points<IB: Backend>(
    fixed_inner: &Image<f32, IB, 3>,
    mask_inner: &Image<f32, IB, 3>,
    sampling_pct: f32,
) -> Tensor<f32, IB> {
    let [nz, ny, nx] = mask_inner.shape();
    let total_voxels = nz * ny * nx;
    let device = mask_inner.data().device();

    // Read mask to CPU once.
    let mask_data = mask_inner
        .data()
        .clone()
        .reshape([total_voxels])
        .into_data();
    let mask_slice = mask_data
        .as_slice::<f32>()
        .expect("mask tensor data must be contiguous");

    // Collect foreground voxel (x, y, z) coordinates (grid convention: [x, y, z] per row).
    // Capacity: at most total_voxels foreground entries Ã— 3 coords
    let mut fg_coords: Vec<f32> = Vec::with_capacity(total_voxels * 3);
    for (i, &v) in mask_slice.iter().enumerate() {
        if v > 0.5 {
            let z = (i / (ny * nx)) as f32;
            let y = ((i % (ny * nx)) / nx) as f32;
            let x = (i % nx) as f32;
            fg_coords.extend_from_slice(&[x, y, z]);
        }
    }
    let fg_count = fg_coords.len() / 3;
    let target = ((total_voxels as f32 * sampling_pct) as usize).max(32);

    let final_coords: Vec<f32> = if fg_count == 0 {
        // Degenerate: mask fully zero at this pyramid level â€” fall back to uniform.
        tracing::warn!(
            "CmaMiRegistration: brain mask is empty at pyramid level (shape [{nz},{ny},{nx}]); \
             falling back to uniform stride-sampling"
        );
        let step = (total_voxels / target).max(1);
        let mut coords = Vec::with_capacity(target * 3);
        let mut i = 0usize;
        while i < total_voxels && coords.len() / 3 < target {
            let z = (i / (ny * nx)) as f32;
            let y = ((i % (ny * nx)) / nx) as f32;
            let x = (i % nx) as f32;
            coords.extend_from_slice(&[x, y, z]);
            i += step;
        }
        coords
    } else if fg_count > target {
        // Sub-sample foreground by stride so evaluation time â‰ˆ unmasked path.
        let step = (fg_count / target).max(1);
        let mut sub = Vec::with_capacity(target * 3);
        let mut i = 0usize;
        while i < fg_count && sub.len() / 3 < target {
            let base = i * 3;
            sub.extend_from_slice(&fg_coords[base..base + 3]);
            i += step;
        }
        sub
    } else {
        fg_coords
    };

    tracing::info!(
        "CmaMiRegistration: foreground sample count = {} / {} total voxels at level [{nz},{ny},{nx}]",
        final_coords.len() / 3,
        total_voxels
    );

    let n = final_coords.len() / 3;
    let idx_t =
        Tensor::<f32, IB>::from_slice_on([n, 3], &final_coords, &device);
    fixed_inner.index_to_world_tensor(idx_t)
}
