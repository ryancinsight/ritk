//! Sparse-cache-path joint histogram computation.
//!
//! Extracted from `mod.rs` (ARCH-330-04) for SRP: this module owns the
//! `build_sparse_w_fixed_transposed` and `compute_joint_histogram_from_cache_sparse`
//! public APIs — the sparse W_fixed^T cache build and the CMA-ES iteration
//! hot-path that reuses cached fixed-image weights.

use burn::tensor::{Shape, TensorData};
use moirai::prelude::ParallelSliceMut;

use super::accumulate::{accumulate_sample_sparse, merge_histograms, validate_inputs};
use super::pool::HistogramPool;
use super::sample::{SampleWindow, SparseSampleCache, SparseWFixedEntry, SparseWFixedT};
use super::types::ParzenConfig;

/// Build the sparse W_fixed^T cache from normalized fixed-image values.
///
/// Per sample, computes Gaussian weights only within ±3σ, storing
/// `(bin_index, weight)` pairs where weight > 1e-12. OOB samples get empty Vec.
/// Each entry also stores `inv_sum_f = 1/sum_f` (SPARSE-329-01) so the
/// sparse path can apply full joint normalization matching the direct path.
///
/// # Arguments
/// * `fixed_norm` — Normalized fixed-image values `[N]` in `[0, num_bins-1]`
/// * `num_bins` — Number of histogram bins
/// * `sigma_sq_fix` — Fixed-image Parzen sigma² (bin-index units)
/// * `oob_mask` — Optional OOB mask `[N]` (1.0 = in-bounds, 0.0 = OOB)
pub fn build_sparse_w_fixed_transposed(
    fixed_norm: &[f32],
    num_bins: usize,
    sigma_sq_fix: f32,
    oob_mask: Option<&[f32]>,
) -> SparseWFixedT {
    // Input validation (DRY-327-05)
    assert!(!fixed_norm.is_empty(), "fixed_norm must not be empty");
    validate_inputs(num_bins, fixed_norm.len(), oob_mask);

    let n = fixed_norm.len();
    let fix_cfg = ParzenConfig::new(sigma_sq_fix);

    // SPARSE-329-01: each element is (entries, inv_sum_f)
    let mut entries: SparseWFixedT = (0..n).map(|_| (SparseSampleCache::new(), 0.0f32)).collect();
    entries.par_mut().enumerate(|i, entry| {
        // OOB check — reuse SampleWindow::mask_val (ARCH-321-04)
        if SampleWindow::mask_val(i, oob_mask).is_none() {
            return;
        }
        let f_val = fixed_norm[i];
        // SPARSE-329-01: compute weights and inv_sum in one pass
        let (f_range, f_weights, inv_sum_f) = fix_cfg.compute_weights_with_inv_sum(f_val, num_bins);
        for (j, w_f) in f_weights.iter() {
            if w_f > 1e-12 {
                entry
                    .0
                    .push(SparseWFixedEntry::new(f_range.lo + j as u16, w_f));
            }
        }
        entry.1 = inv_sum_f; // SPARSE-329-01
    });
    entries
}

/// Compute the joint histogram from a sparse W_fixed^T cache and live moving values.
///
/// Sparse hot-loop variant for CMA-ES iterations after the first. Only moving
/// weights recomputed (`StackWeights`); fixed weights from pre-computed sparse
/// cache (~7 non-zero entries/sample, eliminating full `0..num_bins` scan and
/// `if w_f > 0.0` branch). Rayon parallel reduction (OPT-6) with histogram pool.
///
/// SPARSE-329-01: Full joint normalization `inv_norm = inv_sum_f × inv_sum_m`
/// is now applied, matching the direct path. `inv_sum_f` is stored per-sample
/// in the sparse cache; `inv_sum_m` is computed per-sample from moving values.
/// This eliminates the asymmetry where the sparse path only normalized by
/// `1/sum_m` (Sprint 328), making direct↔sparse histograms numerically identical.
///
/// # Arguments
/// * `sparse_w_fixed` — Sparse fixed-image weights per sample (from `build_sparse_w_fixed_transposed`)
/// * `moving_norm` — Normalized moving-image values `[N]` in `[0, num_bins-1]`
/// * `num_bins` — Number of histogram bins
/// * `sigma_sq_mov` — Moving-image Parzen sigma² (bin-index units)
/// * `oob_mask` — Optional OOB mask `[N]` (1.0 = in-bounds, 0.0 = OOB)
#[allow(private_interfaces)]
pub fn compute_joint_histogram_from_cache_sparse(
    sparse_w_fixed: &[(SparseSampleCache, f32)],
    moving_norm: &[f32],
    num_bins: usize,
    sigma_sq_mov: f32,
    oob_mask: Option<&[f32]>,
    pool: Option<&HistogramPool>,
) -> TensorData {
    // Input validation (DRY-327-05)
    assert!(
        !sparse_w_fixed.is_empty(),
        "sparse_w_fixed must not be empty"
    );
    assert_eq!(
        sparse_w_fixed.len(),
        moving_norm.len(),
        "sparse_w_fixed and moving_norm must have same length"
    );
    validate_inputs(num_bins, sparse_w_fixed.len(), oob_mask);

    let n = sparse_w_fixed.len();
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
            if let Some((_m_val, m_range, m_weights, inv_sum_m)) =
                SampleWindow::new_moving_only(i, moving_norm, num_bins, &mov_cfg, oob_mask)
            {
                // SPARSE-329-01: combine inv_sum_f from cache with inv_sum_m
                let inv_sum_f = sparse_w_fixed[i].1; // per-sample inv_sum_f
                let inv_norm = inv_sum_f * inv_sum_m; // full joint normalization
                accumulate_sample_sparse(
                    &mut local_hist,
                    num_bins,
                    m_range,
                    &m_weights,
                    inv_norm,
                    &sparse_w_fixed[i].0, // fixed entries
                );
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
