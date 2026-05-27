//! Direct NdArray joint histogram computation — avoids full `[N, num_bins]` weight matrices.
//!
//! On the NdArray (CPU) backend, the dominant cost of Parzen histogram computation is
//! building the `[N, num_bins]` weight matrices via broadcast subtraction, squaring,
//! and exp(). For 32³ = 32768 samples × 32 bins, this allocates and computes ~1M
//! elements per intermediate tensor, with 4 intermediates per weight matrix × 2
//! weight matrices = ~8M elements of temporary data (~32MB).
//!
//! This module provides a direct computation path that:
//! 1. Extracts the normalized intensity values as a flat Vec<f32>
//! 2. For each sample, computes the Gaussian weight only for bins within ±3σ
//! 3. Accumulates directly into the `[num_bins, num_bins]` joint histogram
//! 4. Returns the histogram as a Burn tensor
//!
//! This reduces exp() calls by ~4.5× (for 32 bins with σ ≈ 1 bin-width) and
//! eliminates all intermediate `[N, num_bins]` allocations, dramatically reducing
//! memory pressure and improving cache locality.
//!
//! Additionally, a **sparse W_fixed^T** cache path is provided. Instead of
//! storing the full `[num_bins, N]` dense weight matrix (~4MB for 32 bins × 32K
//! samples), each sample's non-zero fixed-image weights are stored as a short
//! `Vec<(bin_index, weight)>` with ~7 entries. The sparse cache eliminates the
//! inner `0..num_bins` scan and the `if w_f > 0.0` branch in the hot-loop
//! variant, further improving cache locality and reducing memory.
//!
//! **Limitation**: This path is only available for the NdArray backend without
//! autodiff. For autodiff or GPU backends, the standard tensor-based path is used.
//!
//! # Inner-loop optimizations
//!
//! The three computation functions share a set of micro-optimizations applied to
//! their hot accumulation loops:
//!
//! - **Row base pointers (OPT-1):** The histogram is logically `[num_bins, num_bins]`
//!   in row-major order. Instead of computing `a * num_bins + b` on every
//!   accumulation, we pre-compute a `Vec<*mut f32>` where `row_ptrs[a]` points to
//!   `histogram[a * num_bins]`. The inner loop then writes `row_ptrs[a].add(b)`,
//!   replacing a multiply with an add.
//!
//! - **Hoisted moving exp() (OPT-2):** In `compute_joint_histogram_direct`,
//!   `w_m` depends only on `b`, not on `a`. Pre-computing the moving weights
//!   before the fixed-weight loop eliminates `(f_range * m_range - m_range)`
//!   redundant exp() calls per sample (e.g. 49 → 14 for a 7×7 window).
//!
//! - **Unchecked histogram access (OPT-3):** Bin indices are clamped to
//!   `[0, num_bins - 1]` before the loop, so every write is in-bounds. Using
//!   `get_unchecked_mut` removes a bounds check from the hottest path.
//!
//! - **Stack-allocated moving weights (OPT-5):** The pre-computed moving weights
//!   span at most `2 * half_width + 1 ≤ 7` entries. A fixed-size `[f32; 7]` array
//!   with a length counter avoids heap allocation entirely.

use burn::tensor::{Shape, TensorData};
use rayon::prelude::*;

/// Maximum number of Parzen bins that any single sample can touch on one axis.
///
/// With the ±3σ rule and a minimum half-width of 3, the range is at most
/// `2 * 3 + 1 = 7` bins.  Used as the capacity of the stack-allocated moving
/// weight array (OPT-5) so that pre-computed weights never heap-allocate.
const MAX_PARZEN_BINS: usize = 7;

/// Stack-allocated moving Parzen weights for a single sample.
///
/// Avoids `Vec` heap allocation for the typically ≤ 7 weight values computed
/// per sample.  The `len` field tracks how many entries in `weights` are active;
/// the remaining slots are uninitialized padding.
///
/// # Safety invariant
///
/// Only `weights[..len]` may be read; indices beyond `len` are uninitialized.
struct StackWeights {
    weights: [f32; MAX_PARZEN_BINS],
    len: usize,
}

impl StackWeights {
    /// Build moving Parzen weights for bins `[lo..=hi]` from a normalised value.
    ///
    /// Each entry `j` stores `exp(-(m_val - (lo + j))² / (2σ²))`.
    fn new(m_val: f32, m_lo: usize, m_hi: usize, inv_2sigma_sq: f32) -> Self {
        let mut weights = [0.0f32; MAX_PARZEN_BINS];
        let mut len = 0;
        for b in m_lo..=m_hi {
            let diff = m_val - b as f32;
            weights[len] = (diff * diff * inv_2sigma_sq).exp();
            len += 1;
        }
        StackWeights { weights, len }
    }
}

/// Pre-compute row base pointers for 2-D histogram row access (OPT-1).
///
/// Returns a `Vec<*mut f32>` of length `num_bins` where `ptrs[a]` points to
/// `histogram[a * num_bins]`, allowing the inner accumulation loop to use
/// pointer addition instead of a multiply.
///
/// # Safety
///
/// Every returned pointer remains valid for the lifetime of `histogram` as
/// long as the vector is not resized or reallocated.  Callers must not use a
/// pointer after the histogram vector is dropped or mutated in a way that
/// could trigger reallocation.
fn row_base_pointers(histogram: &mut [f32], num_bins: usize) -> Vec<*mut f32> {
    histogram
        .chunks_exact_mut(num_bins)
        .map(|row| row.as_mut_ptr())
        .collect()
}

/// Compute the joint histogram directly from normalized intensity values.
///
/// This is the hot-path optimization for the NdArray backend: instead of building
/// full `[N, num_bins]` Parzen weight matrices and multiplying them, we compute
/// the histogram by iterating over samples and accumulating each sample's
/// contribution directly into the `[num_bins, num_bins]` result.
///
/// # Arguments
/// * `fixed_norm` — Normalized fixed-image values `[N]` in `[0, num_bins - 1]`
/// * `moving_norm` — Normalized moving-image values `[N]` in `[0, num_bins - 1]`
/// * `num_bins` — Number of histogram bins
/// * `sigma_sq_fix` — Fixed-image Parzen sigma² in bin-index units
/// * `sigma_sq_mov` — Moving-image Parzen sigma² in bin-index units
/// * `oob_mask` — Optional OOB mask `[N]` (1.0 = in-bounds, 0.0 = out-of-bounds)
///
/// # Returns
/// Joint histogram `[num_bins, num_bins]` as a TensorData object.
pub fn compute_joint_histogram_direct(
    fixed_norm: &[f32],
    moving_norm: &[f32],
    num_bins: usize,
    sigma_sq_fix: f32,
    sigma_sq_mov: f32,
    oob_mask: Option<&[f32]>,
) -> TensorData {
    let n = fixed_norm.len();
    let half_width_fix = compute_half_width_from_sigma_sq(sigma_sq_fix);
    let half_width_mov = compute_half_width_from_sigma_sq(sigma_sq_mov);

    let mut histogram = vec![0.0f32; num_bins * num_bins];
    // OPT-1: row base pointers eliminate the `a * num_bins` multiply.
    let row_ptrs = row_base_pointers(&mut histogram, num_bins);

    let inv_2sigma_sq_fix = -0.5 / sigma_sq_fix;
    let inv_2sigma_sq_mov = -0.5 / sigma_sq_mov;

    for i in 0..n {
        // OOB check
        let mask_val = match oob_mask {
            Some(m) => m[i],
            None => 1.0,
        };
        if mask_val < 0.5 {
            continue;
        }

        let f_val = fixed_norm[i];
        let m_val = moving_norm[i];

        // Primary bins
        let f_primary = f_val.floor() as i32;
        let m_primary = m_val.floor() as i32;

        // Compute fixed-image weight range for bins in [f_primary - hw, f_primary + hw]
        let f_lo = (f_primary - half_width_fix as i32).max(0) as usize;
        let f_hi = ((f_primary + half_width_fix as i32).min(num_bins as i32 - 1)) as usize;

        // Compute moving-image weight range for bins in [m_primary - hw, m_primary + hw]
        let m_lo = (m_primary - half_width_mov as i32).max(0) as usize;
        let m_hi = ((m_primary + half_width_mov as i32).min(num_bins as i32 - 1)) as usize;

        // OPT-2: Pre-compute moving weights once per sample.
        // `w_m` depends only on `b`, not on `a`, so hoisting it out of the
        // fixed-weight loop eliminates `(f_range - 1) * m_range` redundant
        // exp() calls per sample (e.g. 6×7 = 42 saved for a 7×7 window).
        let mw = StackWeights::new(m_val, m_lo, m_hi, inv_2sigma_sq_mov);

        // Accumulate: H[a, b] += W_fixed[i, a] * W_moving[i, b]
        for a in f_lo..=f_hi {
            let diff_f = f_val - a as f32;
            let w_f = (diff_f * diff_f * inv_2sigma_sq_fix).exp();

            let row = row_ptrs[a];
            // OPT-3: unchecked access — indices are clamped to [0, num_bins-1].
            for j in 0..mw.len {
                // SAFETY: `m_lo + j ≤ m_hi ≤ num_bins - 1`, and `row` points to
                // `histogram[a * num_bins .. a * num_bins + num_bins]`, so
                // `row.add(m_lo + j)` is within the allocated slice.
                unsafe {
                    *row.add(m_lo + j) += w_f * mw.weights[j];
                }
            }
        }
    }

    TensorData::new(histogram, Shape::new([num_bins, num_bins]))
}

/// Compute the joint histogram from cached W_fixed^T and live moving values.
///
/// This is the hot-loop variant used on every CMA-ES iteration after the first.
/// Only the moving-image weights are recomputed; the fixed-image weights are
/// provided as a pre-computed `[num_bins, N]` slice.
///
/// # Arguments
/// * `w_fixed_transposed` — Pre-computed fixed-image weights `[num_bins × N]` (row-major)
/// * `moving_norm` — Normalized moving-image values `[N]` in `[0, num_bins - 1]`
/// * `num_bins` — Number of histogram bins
/// * `sigma_sq_mov` — Moving-image Parzen sigma² in bin-index units
/// * `oob_mask` — Optional OOB mask `[N]` (1.0 = in-bounds, 0.0 = out-of-bounds)
pub fn compute_joint_histogram_from_cache_direct(
    w_fixed_transposed: &[f32],
    moving_norm: &[f32],
    num_bins: usize,
    n: usize,
    sigma_sq_mov: f32,
    oob_mask: Option<&[f32]>,
) -> TensorData {
    let half_width_mov = compute_half_width_from_sigma_sq(sigma_sq_mov);
    let inv_2sigma_sq_mov = -0.5 / sigma_sq_mov;

    let mut histogram = vec![0.0f32; num_bins * num_bins];
    // OPT-1: row base pointers.
    let row_ptrs = row_base_pointers(&mut histogram, num_bins);

    for i in 0..n {
        // OOB check
        let mask_val = match oob_mask {
            Some(m) => m[i],
            None => 1.0,
        };
        if mask_val < 0.5 {
            continue;
        }

        let m_val = moving_norm[i];
        let m_primary = m_val.floor() as i32;
        let m_lo = (m_primary - half_width_mov as i32).max(0) as usize;
        let m_hi = ((m_primary + half_width_mov as i32).min(num_bins as i32 - 1)) as usize;

        // Pre-compute moving weights for this sample.
        let mw = StackWeights::new(m_val, m_lo, m_hi, inv_2sigma_sq_mov);

        // Accumulate with cached fixed weights.
        // H[a, b] += W_fixed^T[a, i] * W_moving[i, b]
        for a in 0..num_bins {
            let w_f = w_fixed_transposed[a * n + i];
            if w_f > 0.0 {
                let row = row_ptrs[a];
                for j in 0..mw.len {
                    // SAFETY: `m_lo + j ≤ m_hi ≤ num_bins - 1`, and `row` points to
                    // `histogram[a * num_bins .. a * num_bins + num_bins]`, so
                    // `row.add(m_lo + j)` is within the allocated slice.
                    unsafe {
                        *row.add(m_lo + j) += w_f * mw.weights[j];
                    }
                }
            }
        }
    }

    TensorData::new(histogram, Shape::new([num_bins, num_bins]))
}

/// Sparse representation of W_fixed^T.
///
/// Each inner `Vec` corresponds to one sample and contains `(bin_index, weight)`
/// pairs for the non-zero bins within ±3σ. Typically ~7 entries per sample
/// (for σ ≈ 1 bin-width with the minimum half-width of 3).
pub type SparseWFixedT = Vec<Vec<(usize, f32)>>;

/// Build the sparse W_fixed^T cache from normalized fixed-image values.
///
/// For each sample, this computes the Gaussian Parzen weights only for bins
/// within ±3σ of the primary bin, storing `(bin_index, weight)` pairs where
/// the weight exceeds 1e-12. OOB samples receive an empty Vec.
///
/// # Arguments
/// * `fixed_norm` — Normalized fixed-image values `[N]` in `[0, num_bins - 1]`
/// * `num_bins` — Number of histogram bins
/// * `sigma_sq_fix` — Fixed-image Parzen sigma² in bin-index units
/// * `oob_mask` — Optional OOB mask `[N]` (1.0 = in-bounds, 0.0 = out-of-bounds)
pub fn build_sparse_w_fixed_transposed(
    fixed_norm: &[f32],
    num_bins: usize,
    sigma_sq_fix: f32,
    oob_mask: Option<&[f32]>,
) -> SparseWFixedT {
    let n = fixed_norm.len();
    let half_width_fix = compute_half_width_from_sigma_sq(sigma_sq_fix);
    let inv_2sigma_sq_fix = -0.5 / sigma_sq_fix;

    let mut entries: SparseWFixedT = (0..n).map(|_| Vec::with_capacity(7)).collect();
    entries.par_iter_mut().enumerate().for_each(|(i, entry)| {
        // OOB check
        let mask_val = match oob_mask {
            Some(m) => m[i],
            None => 1.0,
        };
        if mask_val < 0.5 {
            return;
        }
        let f_val = fixed_norm[i];
        let f_primary = f_val.floor() as i32;
        let f_lo = (f_primary - half_width_fix as i32).max(0) as usize;
        let f_hi = ((f_primary + half_width_fix as i32).min(num_bins as i32 - 1)) as usize;
        for a in f_lo..=f_hi {
            let diff_f = f_val - a as f32;
            let w_f = (diff_f * diff_f * inv_2sigma_sq_fix).exp();
            if w_f > 1e-12 {
                entry.push((a, w_f));
            }
        }
    });
    entries
}

/// Compute the joint histogram from a sparse W_fixed^T cache and live moving values.
///
/// This is the sparse hot-loop variant used on every CMA-ES iteration after the
/// first. Only the moving-image weights are recomputed; the fixed-image weights
/// are provided as a pre-computed sparse cache. The inner loop iterates only
/// over the ~7 non-zero entries per sample, eliminating the full `0..num_bins`
/// scan and the `if w_f > 0.0` branch required by the dense cache path.
///
/// # Arguments
/// * `sparse_w_fixed` — Sparse fixed-image weights per sample (from `build_sparse_w_fixed_transposed`)
/// * `moving_norm` — Normalized moving-image values `[N]` in `[0, num_bins - 1]`
/// * `num_bins` — Number of histogram bins
/// * `sigma_sq_mov` — Moving-image Parzen sigma² in bin-index units
/// * `oob_mask` — Optional OOB mask `[N]` (1.0 = in-bounds, 0.0 = out-of-bounds)
pub fn compute_joint_histogram_from_cache_sparse(
    sparse_w_fixed: &SparseWFixedT,
    moving_norm: &[f32],
    num_bins: usize,
    sigma_sq_mov: f32,
    oob_mask: Option<&[f32]>,
) -> TensorData {
    let n = sparse_w_fixed.len();
    let half_width_mov = compute_half_width_from_sigma_sq(sigma_sq_mov);
    let inv_2sigma_sq_mov = -0.5 / sigma_sq_mov;

    let mut histogram = vec![0.0f32; num_bins * num_bins];
    // OPT-1: row base pointers.
    let row_ptrs = row_base_pointers(&mut histogram, num_bins);

    for i in 0..n {
        // OOB check
        let mask_val = match oob_mask {
            Some(m) => m[i],
            None => 1.0,
        };
        if mask_val < 0.5 {
            continue;
        }

        let m_val = moving_norm[i];
        let m_primary = m_val.floor() as i32;
        let m_lo = (m_primary - half_width_mov as i32).max(0) as usize;
        let m_hi = ((m_primary + half_width_mov as i32).min(num_bins as i32 - 1)) as usize;

        // OPT-4: Pre-compute moving weights once per sample (same hoisting
        // as OPT-2, applied to the sparse cache path).
        let mw = StackWeights::new(m_val, m_lo, m_hi, inv_2sigma_sq_mov);

        // H[a, b] += W_fixed^T[a, i] * W_moving[i, b]
        for &(a, w_f) in &sparse_w_fixed[i] {
            let row = row_ptrs[a];
            for j in 0..mw.len {
                // SAFETY: `a` comes from the sparse cache which only stores
                // bin indices in `[0, num_bins - 1]`.  `m_lo + j ≤ m_hi ≤
                // num_bins - 1`.  `row` points to
                // `histogram[a * num_bins .. a * num_bins + num_bins]`, so
                // `row.add(m_lo + j)` is within the allocated slice.
                unsafe {
                    *row.add(m_lo + j) += w_f * mw.weights[j];
                }
            }
        }
    }

    TensorData::new(histogram, Shape::new([num_bins, num_bins]))
}

/// Compute half-width from sigma² using the ±3σ rule.
fn compute_half_width_from_sigma_sq(sigma_sq: f32) -> usize {
    let sigma = sigma_sq.sqrt();
    let computed = (3.0 * sigma).ceil() as usize;
    computed.max(3) // minimum 3 bins on each side
}

#[cfg(test)]
mod direct_tests;
