//! Sparse Parzen weight computation â€” only evaluates Gaussian kernel within Â±3Ïƒ support.
#![allow(clippy::needless_range_loop)]
//!
//! The dense Parzen kernel computes `exp(-0.5 * ((val[i] - b) / Ïƒ)Â²)` for every
//! sample-bin pair, producing a full `[N, num_bins]` weight matrix. For typical
//! Mattes MI where Ïƒ â‰ˆ 1 bin-width, the Gaussian has negligible mass beyond Â±3Ïƒ
//! (â‰ˆ Â±3 bins), so ~75% of exp() calls compute values < 1e-4 that contribute
//! nothing meaningful to the histogram.
//!
//! The sparse approach instead:
//! 1. Computes each sample's primary bin via `floor(normalized_value)`.
//! 2. Evaluates the Gaussian only for bins in `[primary - half_width, primary + half_width]`,
//!    where `half_width = ceil(3 Ã— Ïƒ_in_bins)`.
//! 3. Scatters the sparse weights into the full `[N, num_bins]` matrix.
//!
//! This reduces exp() evaluations by a factor of `num_bins / (2 Ã— half_width + 1)`,
//! which is ~4.5Ã— for 32 bins with Ïƒ = 1 bin-width, and ~7Ã— for 50 bins.
//!
//! The scatter operation is fully differentiable in Burn's autodiff backend,
//! so gradient flow through the moving-image weights is preserved.

#[cfg(test)]
use ritk_image::tensor::Backend;
#[cfg(test)]
use ritk_image::tensor::{Int, Shape, Tensor };

/// Minimum support half-width.
///
/// Delegates to the canonical `direct::types::MIN_HALF_WIDTH` when
/// `direct-parzen` is enabled; otherwise defines its own copy (this
/// module is `#[cfg(test)]`-only and not on the hot path).
#[cfg(test)]
#[cfg(not(feature = "direct-parzen"))]
const MIN_HALF_WIDTH: usize = 3;

/// When `direct-parzen` is enabled, import the canonical constant.
#[cfg(test)]
#[cfg(feature = "direct-parzen")]
use super::direct::types::MIN_HALF_WIDTH;

/// Compute the support half-width from sigmaÂ² (SSOT-compatible API).
///
/// When `direct-parzen` is enabled, delegates to the canonical
/// `direct::compute_half_width(sigma_sq)`. Otherwise provides
/// a standalone implementation for this `#[cfg(test)]`-only module.
///
/// Returns `ceil(3 * sqrt(sigma_sq)).max(MIN_HALF_WIDTH)` â€” this captures
/// >99.7% of the Gaussian mass (Â±3Ïƒ rule) while guaranteeing at least
/// `MIN_HALF_WIDTH` bins per side for numerical stability.
#[cfg(test)]
#[cfg(not(feature = "direct-parzen"))]
pub(crate) fn compute_half_width(sigma_sq: f32) -> usize {
    let sigma = sigma_sq.sqrt();
    let computed = (3.0 * sigma).ceil() as usize;
    computed.max(MIN_HALF_WIDTH)
}

/// Delegate to the canonical `direct::compute_half_width` when available.
#[cfg(test)]
#[cfg(feature = "direct-parzen")]
pub(crate) fn compute_half_width(sigma_sq: f32) -> usize {
    super::direct::compute_half_width(sigma_sq)
}

/// Compute sparse Parzen weight matrix `[N, num_bins]` using scatter.
///
/// Instead of the dense broadcast `vals[:, None] - bins[None, :]`, this function
/// identifies the support window around each sample's primary bin and evaluates
/// the Gaussian kernel only within that window. The resulting sparse weights are
/// scattered into the full `[N, num_bins]` matrix.
///
/// # Boundary handling
///
/// When a sample's support window extends beyond `[0, num_bins-1]`, the out-of-bounds
/// entries are **zeroed out** before scattering. This prevents the scatter-add from
/// incorrectly accumulating weights at clamped boundary bins. The Gaussian weight at
/// those out-of-bounds positions would have been near-zero anyway (they're beyond Â±3Ïƒ
/// of the distribution center), so discarding them introduces negligible error.
///
/// # Arguments
/// * `vals_norm` â€” Normalized sample values `[N]` in `[0, num_bins - 1]`.
/// * `num_bins` â€” Number of histogram bins.
/// * `sigma_sq` â€” Parzen sigma squared, in bin-index units.
/// * `half_width` â€” Support half-width (see [`compute_half_width`]).
/// * `_bins_exp` â€” Pre-computed bin centers `[1, num_bins]` (unused â€” indices computed directly).
///
/// # Returns
/// Weight matrix `[N, num_bins]` where `W[i, b] = exp(-0.5 * ((val[i] - b) / Ïƒ)Â²)`
/// for bins within the support window, and 0 elsewhere.
#[cfg(test)]
pub(crate) fn compute_sparse_parzen_weights<B: Backend>(
    vals_norm: Tensor<f32, B>,
    num_bins: usize,
    sigma_sq: f32,
    half_width: usize,
    _bins_exp: &Tensor<f32, B>,
) -> Tensor<f32, B> {
    let [n] = vals_norm.dims();
    let device = vals_norm.device();
    let num_bins_i32 = num_bins as i32;
    let window_size = 2 * half_width + 1;

    // 1. Primary bin for each sample: floor(val)
    let primary = vals_norm.clone().floor();

    // 2. Build offset tensor: [-half_width, ..., 0, ..., +half_width]
    //    Shape: [1, window_size] â€” broadcasts over N samples.
    let offsets_data: Vec<i32> = (-(half_width as i32)..=(half_width as i32)).collect();
    let offsets = Tensor::<B, 1, Int>::from_data(
        ::new(offsets_data.clone(), Shape::new([window_size])),
        &device,
    );

    // 3. Build bin-index tensor for the support window: [N, window_size]
    //    bin_idx[i, k] = floor(val[i]) + offset[k]
    //    Then clamp to [0, num_bins-1] for scatter validity.
    let primary_i32 = primary.clone().int();

    // Unclamped indices: for detecting boundary entries to zero out.
    let bin_idx_unclamped = primary_i32
        .clone()
        .reshape([n, 1])
        .add(offsets.clone().reshape([1, window_size]));

    // Clamped indices: valid for scatter.
    let bin_idx = bin_idx_unclamped.clone().clamp(0, num_bins_i32 - 1);

    // 4. Boundary mask: zero out weights where the bin index was clamped.
    //    scatter is additive â€” clamped entries would incorrectly accumulate
    //    at boundary bins. Zeroing them prevents this artifact.
    let in_bounds = bin_idx_unclamped.equal(bin_idx.clone()).float();

    // 5. Compute Gaussian weights for the support window: [N, window_size]
    //    diff[i, k] = val[i] - bin_idx[i, k]
    //    weight[i, k] = exp(-0.5 * diffÂ² / ÏƒÂ²)
    let bin_idx_float = bin_idx.clone().float();
    let vals_exp = vals_norm.reshape([n, 1]);
    let diff = vals_exp - bin_idx_float; // [N, window_size]
    let sq = diff.clone() * diff;
    let weights = (sq * (-0.5 / sigma_sq)).exp() * in_bounds; // [N, window_size]

    // 6. Scatter sparse weights into the full [N, num_bins] matrix.
    //    For each (i, k), set W[i, bin_idx[i, k]] += weight[i, k].
    let zeros = Tensor::<f32, B>::zeros([n, num_bins], &device);
    zeros.scatter(1, bin_idx, weights)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    fn device() -> <B as Backend>::Device {
        Default::default()
    }

    /// Helper: compute dense Parzen weights for comparison.
    fn dense_parzen_weights(
        vals: &Tensor<f32, B>,
        num_bins: usize,
        sigma_sq: f32,
        dev: &<B as Backend>::Device,
    ) -> Tensor<f32, B> {
        let bins_exp = Tensor::<B, 1, Int>::arange(0..num_bins as i64, dev)
            .float()
            .reshape([1, num_bins]);
        let n = vals.dims()[0];
        let vals_exp = vals.clone().reshape([n, 1]);
        let diff = vals_exp - bins_exp;
        let sq = diff.clone() * diff;
        (sq * (-0.5 / sigma_sq)).exp()
    }

    #[test]
    fn sparse_matches_dense_interior() {
        // For values far from boundaries, sparse weights match dense exactly
        // within the support window, and are zero (instead of near-zero) outside it.
        let dev = device();
        let num_bins = 32;
        let sigma_in_bins = 1.0_f32;
        let sigma_sq = sigma_in_bins * sigma_in_bins;
        let half_width = compute_half_width(sigma_sq);

        // Values safely in the interior
        let vals = Tensor::<f32, B>::from_floats([10.0, 15.0], &dev);
        let bins_exp = Tensor::<B, 1, Int>::arange(0..num_bins as i64, &dev)
            .float()
            .reshape([1, num_bins]);

        let sparse =
            compute_sparse_parzen_weights(vals.clone(), num_bins, sigma_sq, half_width, &bins_exp);

        let dense = dense_parzen_weights(&vals, num_bins, sigma_sq, &dev);

        let sparse_data = sparse.into_data();
        let sparse_slice = sparse_data.as_slice::<f32>().unwrap();
        let dense_data = dense.into_data();
        let dense_slice = dense_data.as_slice::<f32>().unwrap();

        // For each sample, check that weights within the support window match dense.
        for (row, val_f) in [10.0_f32, 15.0_f32].iter().enumerate() {
            let primary = val_f.floor() as usize;
            let lo = primary.saturating_sub(half_width);
            let hi = (primary + half_width).min(num_bins - 1);
            for b in lo..=hi {
                let idx = row * num_bins + b;
                let diff = (dense_slice[idx] - sparse_slice[idx]).abs();
                assert!(
                    diff < 1e-5,
                    "row {row} bin {b}: dense={}, sparse={}, diff={diff}",
                    dense_slice[idx],
                    sparse_slice[idx]
                );
            }
        }

        // Outside the support window, sparse is zero. Dense may have tiny values
        // (e.g. exp(-12.5) â‰ˆ 3.7e-6), but these are negligible for MI computation.
        // Verify total weight difference is < 0.1%.
        let sparse_sum: f32 = sparse_slice.iter().sum();
        let dense_sum: f32 = dense_slice.iter().sum();
        let relative_error = (dense_sum - sparse_sum) / dense_sum;
        assert!(
            relative_error < 0.001,
            "sparse total should be within 0.1% of dense: sparse={sparse_sum}, dense={dense_sum}, rel_err={relative_error}"
        );
    }

    #[test]
    fn sparse_matches_dense_broad_sigma_interior() {
        // With sigma = 3 bins, the support is wider but still matches dense
        // for interior values.
        let dev = device();
        let num_bins = 50;
        let sigma_in_bins = 3.0_f32;
        let sigma_sq = sigma_in_bins * sigma_in_bins;
        let half_width = compute_half_width(sigma_sq);

        let vals = Tensor::<f32, B>::from_floats([25.0], &dev);
        let bins_exp = Tensor::<B, 1, Int>::arange(0..num_bins as i64, &dev)
            .float()
            .reshape([1, num_bins]);

        let sparse =
            compute_sparse_parzen_weights(vals.clone(), num_bins, sigma_sq, half_width, &bins_exp);

        let dense = dense_parzen_weights(&vals, num_bins, sigma_sq, &dev);

        let sparse_data = sparse.into_data();
        let sparse_slice = sparse_data.as_slice::<f32>().unwrap();
        let dense_data = dense.into_data();
        let dense_slice = dense_data.as_slice::<f32>().unwrap();

        // val=25 is safely interior for num_bins=50 and half_width=9.
        // The entire support window [16, 34] is in-bounds.
        for b in 16..=34 {
            let diff = (dense_slice[b] - sparse_slice[b]).abs();
            assert!(
                diff < 1e-4,
                "bin {b}: dense={}, sparse={}, diff={diff}",
                dense_slice[b],
                sparse_slice[b]
            );
        }

        // Total weight should be very close
        let sparse_sum: f32 = sparse_slice.iter().sum();
        let dense_sum: f32 = dense_slice.iter().sum();
        let relative_error = (dense_sum - sparse_sum) / dense_sum;
        assert!(
            relative_error < 0.002,
            "sparse total should be within 0.2% of dense: sparse={sparse_sum}, dense={dense_sum}, rel_err={relative_error}"
        );
    }

    #[test]
    fn sparse_near_boundary_approximates_dense() {
        // Values near the boundary have their out-of-bounds support window
        // entries zeroed. The remaining in-bounds entries match dense exactly.
        let dev = device();
        let num_bins = 32;
        let sigma_in_bins = 1.0_f32;
        let sigma_sq = sigma_in_bins * sigma_in_bins;
        let half_width = compute_half_width(sigma_sq);

        // val = 1.5 â†’ primary bin 1, support bins [-2, ..., 4], clamped to [0, 4]
        let vals = Tensor::<f32, B>::from_floats([1.5], &dev);
        let bins_exp = Tensor::<B, 1, Int>::arange(0..num_bins as i64, &dev)
            .float()
            .reshape([1, num_bins]);

        let sparse =
            compute_sparse_parzen_weights(vals.clone(), num_bins, sigma_sq, half_width, &bins_exp);

        let dense = dense_parzen_weights(&vals, num_bins, sigma_sq, &dev);

        let sparse_data = sparse.into_data();
        let sparse_slice = sparse_data.as_slice::<f32>().unwrap();
        let dense_data = dense.into_data();
        let dense_slice = dense_data.as_slice::<f32>().unwrap();

        // In-bounds bins [0, 4] should match exactly
        for b in 0..5 {
            let diff = (dense_slice[b] - sparse_slice[b]).abs();
            assert!(
                diff < 1e-5,
                "bin {b}: dense={}, sparse={}, diff={diff}",
                dense_slice[b],
                sparse_slice[b]
            );
        }

        // Bins beyond the support window [5, 31] should be zero in sparse
        for b in 5..num_bins {
            assert!(
                sparse_slice[b] < 1e-6,
                "bin {b} should be zero in sparse, got {}",
                sparse_slice[b]
            );
        }

        // The total sparse weight should be close to dense (missing OOB entries are negligible)
        let sparse_sum: f32 = sparse_slice.iter().sum();
        let dense_sum: f32 = dense_slice.iter().sum();
        let relative_error = (dense_sum - sparse_sum) / dense_sum;
        assert!(
            relative_error < 0.05,
            "sparse total should be within 5% of dense: sparse={sparse_sum}, dense={dense_sum}, rel_err={relative_error}"
        );
    }

    #[test]
    fn half_width_minimum_is_3() {
        // Even with sigma â†’ 0, half_width should be at least 3
        // compute_half_width takes sigma_sq: 0.1Â²=0.01, 0.316Â²â‰ˆ0.1
        assert_eq!(compute_half_width(0.0001), MIN_HALF_WIDTH);
        assert_eq!(compute_half_width(0.01), MIN_HALF_WIDTH);
    }

    #[test]
    fn half_width_scales_with_sigma() {
        // compute_half_width takes sigma_sq; sigma = sqrt(sigma_sq)
        assert_eq!(compute_half_width(1.0), 3); // sigma=1, ceil(3*1) = 3
        assert_eq!(compute_half_width(4.0), 6); // sigma=2, ceil(3*2) = 6
        assert_eq!(compute_half_width(9.0), 9); // sigma=3, ceil(3*3) = 9
    }

    #[test]
    fn sparse_boundary_clamping() {
        // Values near 0 should produce valid weights without double-counting.
        let dev = device();
        let num_bins = 32;
        let sigma_in_bins = 1.0_f32;
        let sigma_sq = sigma_in_bins * sigma_in_bins;
        let half_width = compute_half_width(sigma_sq);

        // val = 0.5 â†’ primary bin 0, support [-3, ..., 3], clamped to [0, 3]
        let vals = Tensor::<f32, B>::from_floats([0.5], &dev);
        let bins_exp = Tensor::<B, 1, Int>::arange(0..num_bins as i64, &dev)
            .float()
            .reshape([1, num_bins]);

        let sparse =
            compute_sparse_parzen_weights(vals.clone(), num_bins, sigma_sq, half_width, &bins_exp);

        let data = sparse.into_data();
        let slice = data.as_slice::<f32>().unwrap();
        // First few bins should have non-zero weight
        let sum_first_4: f32 = slice[0..4].iter().sum();
        assert!(
            sum_first_4 > 0.0,
            "Bins 0-3 should have non-zero weight for val=0.5"
        );
        // Bins beyond the support should be zero
        let sum_beyond: f32 = slice[7..32].iter().sum();
        assert!(
            sum_beyond < 1e-6,
            "Bins beyond support should be zero, got sum={sum_beyond}"
        );
    }

    #[test]
    fn sparse_no_double_counting_at_boundary() {
        // The key invariant: when multiple offset positions clamp to the same bin,
        // the scatter must NOT double-count. The boundary mask prevents this.
        let dev = device();
        let num_bins = 32;
        let sigma_in_bins = 1.0_f32;
        let sigma_sq = sigma_in_bins * sigma_in_bins;
        let half_width = compute_half_width(sigma_sq);

        // val = 0.5 â†’ primary bin 0, offsets [-3,-2,-1,0,1,2,3] â†’ clamped bins [0,0,0,0,1,2,3]
        // Without boundary masking, bin 0 would get 4Ã— the correct weight.
        let vals = Tensor::<f32, B>::from_floats([0.5], &dev);
        let bins_exp = Tensor::<B, 1, Int>::arange(0..num_bins as i64, &dev)
            .float()
            .reshape([1, num_bins]);

        let sparse =
            compute_sparse_parzen_weights(vals.clone(), num_bins, sigma_sq, half_width, &bins_exp);
        let sparse_data = sparse.into_data();
        let sparse_slice = sparse_data.as_slice::<f32>().unwrap();

        // The weight at bin 0 should be exp(-0.5 * (0.5 - 0)Â²) = exp(-0.125) â‰ˆ 0.8825
        // (only the in-bounds contribution from offset=0, the out-of-bounds offsets are zeroed)
        let expected_bin0 = (-0.5_f32 * (0.5 - 0.0_f32).powi(2) / sigma_sq).exp();
        let diff = (sparse_slice[0] - expected_bin0).abs();
        assert!(
            diff < 1e-5,
            "bin 0 weight should be {expected_bin0}, got {}, diff={diff}",
            sparse_slice[0]
        );
    }
}
