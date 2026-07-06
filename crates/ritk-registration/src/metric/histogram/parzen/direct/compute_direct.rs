//! Direct-path joint histogram computation.
//!
//! Extracted from `mod.rs` (ARCH-330-04) for SRP: this module owns the
//! `compute_joint_histogram_direct` public API — the hot-path that
//! iterates samples and accumulates directly into `[num_bins, num_bins]`.

use ritk_image::tensor::{Shape, TensorData};

use super::accumulate::{accumulate_sample_direct, merge_histograms, validate_inputs};
use super::pool::HistogramPool;
use super::sample::SampleWindow;
use super::types::ParzenConfig;

/// Compute the joint histogram directly from normalized intensity values.
///
/// NdArray hot-path: iterates samples, accumulates directly into
/// `[num_bins, num_bins]` instead of building full `[N, num_bins]` weight
/// matrices. Fixed/moving weights pre-computed as `StackWeights` in
/// `SampleWindow` — heap-free inner loop, no `SparseWFixedEntry`.
///
/// Rayon parallel reduction (OPT-6): thread-local histograms merged in
/// reduce phase — no locks, atomics, or `unsafe`.
///
/// # Arguments
/// * `fixed_norm` — Normalized fixed-image values `[N]` in `[0, num_bins-1]`
/// * `moving_norm` — Normalized moving-image values `[N]` in `[0, num_bins-1]`
/// * `num_bins` — Number of histogram bins
/// * `sigma_sq_fix` — Fixed-image Parzen sigma² (bin-index units)
/// * `sigma_sq_mov` — Moving-image Parzen sigma² (bin-index units)
/// * `oob_mask` — Optional OOB mask `[N]` (1.0 = in-bounds, 0.0 = OOB)
///
/// # Returns
/// Joint histogram `[num_bins, num_bins]` as TensorData.
///
/// # Parallel accumulation trade-off
///
/// Float accumulation order differs under parallel reduction (~1e-5 vs
/// sequential), within the 1e-4 test tolerance.
#[allow(private_interfaces)]
pub fn compute_joint_histogram_direct(
    fixed_norm: &[f32],
    moving_norm: &[f32],
    num_bins: usize,
    sigma_sq_fix: f32,
    sigma_sq_mov: f32,
    oob_mask: Option<&[f32]>,
    pool: Option<&HistogramPool>,
) -> TensorData {
    // Input validation (DRY-327-05)
    assert!(!fixed_norm.is_empty(), "fixed_norm must not be empty");
    assert!(!moving_norm.is_empty(), "moving_norm must not be empty");
    assert_eq!(
        fixed_norm.len(),
        moving_norm.len(),
        "fixed_norm and moving_norm must have same length"
    );
    validate_inputs(num_bins, fixed_norm.len(), oob_mask);

    let n = fixed_norm.len();
    let fix_cfg = ParzenConfig::new(sigma_sq_fix);
    let mov_cfg = ParzenConfig::new(sigma_sq_mov);
    let local_pool_if_none;
    let pool: &HistogramPool = match pool {
        Some(p) => p,
        None => {
            local_pool_if_none = HistogramPool::new(num_bins * num_bins);
            &local_pool_if_none
        }
    };

    let histogram: Vec<f32> = moirai::fold_reduce_with::<moirai::Adaptive, _, _, _, _>(
        n,
        || pool.checkout(),
        |mut local_hist, i| {
            if let Some(window) = SampleWindow::new(
                i,
                fixed_norm,
                moving_norm,
                num_bins,
                &fix_cfg,
                &mov_cfg,
                oob_mask,
            ) {
                accumulate_sample_direct(&mut local_hist, num_bins, &window);
            }
            local_hist
        },
        |mut acc, local| {
            merge_histograms(&mut acc, &local);
            pool.return_buffer(local);
            acc
        },
    );

    TensorData::new(histogram, Shape::new([num_bins, num_bins]))
}
