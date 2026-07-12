//! Deep-learning similarity losses on the Coeus autodiff engine.
//!
//! Atlas migration to the Coeus autodiff engine: these are the terminal image-similarity
//! losses a learned registration model (e.g. a displacement/velocity predictor)
//! minimizes during training. Each is built entirely from Coeus autograd
//! [`coeus_autograd::Var`] ops, so `.backward()` on the returned scalar `Var`
//! propagates gradients to whichever inputs carry `requires_grad` — the moving
//! (warped) image and, through it, the model parameters.
//!
//! Inputs are rank-5 tensors `[Batch, Channel, D, H, W]` and the result is a
//! scalar-shaped (`[1]`) `Var`. All arithmetic executes in the input's native
//! precision `T` (no widen/narrow); the losses are backend- and scalar-generic.

use coeus_autograd::{
    avg_pool3d, broadcast_to, div, exp, log, matmul, mean, mul, neg, permute, reshape, scalar_add,
    scalar_div, scalar_mul, sqrt, sub, sum, sum_axis, Var,
};
use coeus_core::{ComputeBackend, CpuAddressableStorage, CpuAddressableStorageMut, Float};
use coeus_ops::BackendOps;
use coeus_tensor::Tensor;

/// Squared magnitude `x·x` as a tracked op (avoids `pow`'s `Neg` bound).
#[inline]
fn square<T, B>(x: &Var<T, B>) -> Var<T, B>
where
    T: Float,
    B: ComputeBackend + BackendOps<T> + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    mul(x, x)
}

/// Mean Squared Error for rank-5 tensors `[B, C, D, H, W]`.
///
/// `MSE = mean((fixed − moving)²)`, a scalar `Var`. Differentiable in both
/// inputs; the reverse pass yields `∂MSE/∂moving = (2/N)·(moving − fixed)`.
pub fn mse_loss<T, B>(fixed: &Var<T, B>, moving: &Var<T, B>) -> Var<T, B>
where
    T: Float,
    B: ComputeBackend + BackendOps<T> + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    mean(&square(&sub(fixed, moving)))
}

/// Global Normalized Cross-Correlation loss for rank-5 tensors.
///
/// Computes NCC independently per `[Batch, Channel]` slice over the `N = D·H·W`
/// voxels (single-pass algebraic-moments form, Lewis 1995), averages across the
/// `B·C` slices, and returns the **negative** so the loss is minimized: range
/// `[−1, 1]`, `−1` at perfect correlation.
///
/// The moments form keeps the whole reduction on `[B·C]`-shaped vectors, so no
/// broadcasting is needed and gradients flow to both inputs.
pub fn ncc_loss<T, B>(fixed: &Var<T, B>, moving: &Var<T, B>) -> Var<T, B>
where
    T: Float,
    B: ComputeBackend + BackendOps<T> + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let dims = fixed.tensor.shape();
    let (bc, n) = (dims[0] * dims[1], dims[2] * dims[3] * dims[4]);
    let f = reshape(fixed, [bc, n]);
    let m = reshape(moving, [bc, n]);

    // Per-slice raw moments (`[B·C]` vectors, reduced over the voxel axis).
    let s_f = sum_axis(&f, 1);
    let s_m = sum_axis(&m, 1);
    let s_ff = sum_axis(&square(&f), 1);
    let s_mm = sum_axis(&square(&m), 1);
    let s_fm = sum_axis(&mul(&f, &m), 1);

    let inv_n = T::from_f64(n as f64);
    // num = S_FM − S_F·S_M / N ; d_X = S_XX − S_X² / N.
    let num = sub(&s_fm, &scalar_div(&mul(&s_f, &s_m), inv_n));
    let d_f = sub(&s_ff, &scalar_div(&square(&s_f), inv_n));
    let d_m = sub(&s_mm, &scalar_div(&square(&s_m), inv_n));

    let eps = T::from_f64(1e-5);
    let ncc = div(&num, &scalar_add(&sqrt(&mul(&d_f, &d_m)), eps));
    neg(&mean(&ncc))
}

/// Local Normalized Cross-Correlation loss for rank-5 tensors.
///
/// Replaces the global statistics of [`ncc_loss`] with local ones computed over
/// a `kernel_size³` window via a uniform box filter (channel-wise 3-D average
/// pooling, stride 1, `kernel_size/2` zero padding). Robust to spatially varying
/// intensity relationships and bias fields. Returns the **negative** local CC
/// averaged over all windows.
///
/// `Var(X) = E[X²] − E[X]²`, `Cov(F, M) = E[FM] − E[F]·E[M]`, all as pooled
/// fields; `LNCC = Cov / √(Var_F·Var_M + ε)`.
pub fn lncc_loss<T, B>(fixed: &Var<T, B>, moving: &Var<T, B>, kernel_size: usize) -> Var<T, B>
where
    T: Float,
    B: ComputeBackend + BackendOps<T> + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let pool = |x: &Var<T, B>| box_filter(x, kernel_size);
    let mean_f = pool(fixed);
    let mean_m = pool(moving);
    let mean_f2 = pool(&square(fixed));
    let mean_m2 = pool(&square(moving));
    let mean_fm = pool(&mul(fixed, moving));

    let var_f = sub(&mean_f2, &square(&mean_f));
    let var_m = sub(&mean_m2, &square(&mean_m));
    let cov = sub(&mean_fm, &mul(&mean_f, &mean_m));

    let eps = T::from_f64(1e-5);
    let cc = div(&cov, &scalar_add(&sqrt(&mul(&var_f, &var_m)), eps));
    neg(&mean(&cc))
}

/// Channel-wise uniform box filter (mean over each `kernel_size³` window) via
/// tracked 3-D average pooling with stride 1 and `kernel_size/2` zero padding.
fn box_filter<T, B>(x: &Var<T, B>, kernel_size: usize) -> Var<T, B>
where
    T: Float,
    B: ComputeBackend + BackendOps<T> + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let dims = x.tensor.shape();
    let [b, c, d, h, w] = [dims[0], dims[1], dims[2], dims[3], dims[4]];
    let pad = kernel_size / 2;
    let (stride, dilation) = (1usize, 1usize);
    // Output spatial extent for stride 1, dilation 1: size + 2·pad − (k − 1).
    let out_dim = |size: usize| size + 2 * pad + 1 - kernel_size;
    let backend = B::default();
    let mut out = Tensor::zeros_on([b, c, out_dim(d), out_dim(h), out_dim(w)], &backend);
    // The autograd `avg_pool3d` only attaches the backward node; compute the
    // forward pooled field into `out` via the backend kernel first.
    let (out_storage, out_layout) = out.storage_mut_and_layout();
    backend.avg_pool3d(
        x.tensor.storage(),
        x.tensor.layout(),
        kernel_size,
        stride,
        pad,
        dilation,
        out_storage,
        out_layout,
    );
    avg_pool3d(x, out, kernel_size, stride, pad, dilation)
}

/// Mutual-Information loss for rank-5 tensors, differentiable via soft (RBF)
/// histogram binning.
///
/// Assigns each voxel a soft membership `exp(−(I − bin)²/(2σ²))` to `num_bins`
/// bins over `[0, 1]`, forms the normalized joint histogram `Wᶠᵀ·Wᵐ / N` and its
/// marginals, and returns the **negative** mutual information
/// `−Σ P(i,j)·(log P(i,j) − log P(i) − log P(j))`. Inputs are assumed
/// pre-normalized to roughly `[0, 1]`. The `[N, num_bins]` soft-assignment
/// matrices make this memory-intensive for large volumes.
pub fn mi_loss<T, B>(
    fixed: &Var<T, B>,
    moving: &Var<T, B>,
    num_bins: usize,
    sigma: f64,
) -> Var<T, B>
where
    T: Float,
    B: ComputeBackend + BackendOps<T> + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let dims = fixed.tensor.shape();
    let n: usize = dims.iter().product();
    let backend = B::default();

    // Bin centers over [0, 1] (constant leaf, no gradient), row `[1, num_bins]`.
    let denom = (num_bins as f64 - 1.0).max(1.0);
    let bin_data: Vec<T> = (0..num_bins)
        .map(|i| T::from_f64(i as f64 / denom))
        .collect();
    let bins_row = broadcast_to(
        &Var::new(
            Tensor::from_slice_on([1, num_bins], &bin_data, &backend),
            false,
        ),
        [n, num_bins],
    );

    // Soft membership of each voxel to each bin: exp(−(I − bin)²/(2σ²)).
    let inv_2sigma2 = -1.0 / (2.0 * sigma * sigma);
    let soft = |img: &Var<T, B>| -> Var<T, B> {
        let col = broadcast_to(&reshape(img, [n, 1]), [n, num_bins]);
        let d = sub(&col, &bins_row);
        exp(&scalar_mul(&square(&d), T::from_f64(inv_2sigma2)))
    };
    let w_f = soft(fixed); // [N, bins]
    let w_m = soft(moving); // [N, bins]

    // Joint histogram Wᶠᵀ·Wᵐ (`[bins, bins]`), normalized to a probability.
    let joint = matmul(&permute(&w_f, &[1, 0]), &w_m);
    let joint_p = scalar_div(&joint, T::from_f64(n as f64));

    // Marginals broadcast back to `[bins, bins]`.
    let p_f = broadcast_to(&reshape(&sum_axis(&joint_p, 1), [num_bins, 1]), [num_bins, num_bins]);
    let p_m = broadcast_to(&reshape(&sum_axis(&joint_p, 0), [1, num_bins]), [num_bins, num_bins]);

    let eps = T::from_f64(1e-10);
    let log_joint = log(&scalar_add(&joint_p, eps));
    let log_pf = log(&scalar_add(&p_f, eps));
    let log_pm = log(&scalar_add(&p_m, eps));
    // MI = Σ P(i,j)·(log P(i,j) − log P(i) − log P(j)); loss = −MI.
    let term = sub(&sub(&log_joint, &log_pf), &log_pm);
    neg(&sum(&mul(&joint_p, &term)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::SequentialBackend;

    type B = SequentialBackend;

    /// Build a `[1, 1, d, d, d]` non-constant `Var` (ascending values) so
    /// correlation-based losses are well-defined (zero-variance inputs make the
    /// NCC/LNCC denominator vanish).
    fn ascending_volume(d: usize) -> Var<f64, B> {
        let n = d * d * d;
        let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        Var::new(
            Tensor::from_slice_on([1, 1, d, d, d], &data, &SequentialBackend),
            true,
        )
    }

    fn from_vals(vals: Vec<f64>, d: usize) -> Var<f64, B> {
        Var::new(
            Tensor::from_slice_on([1, 1, d, d, d], &vals, &SequentialBackend),
            true,
        )
    }

    fn scalar(v: &Var<f64, B>) -> f64 {
        v.tensor.as_slice()[0]
    }

    #[test]
    fn mse_loss_identical_images_is_exactly_zero() {
        let image = ascending_volume(3);
        let loss = scalar(&mse_loss(&image, &image));
        assert_eq!(loss, 0.0, "MSE of identical images must be exactly zero");
    }

    #[test]
    fn mse_loss_known_constant_difference_matches_closed_form() {
        let zeros = from_vals(vec![0.0; 8], 2);
        let ones = from_vals(vec![1.0; 8], 2);
        // mean((0 − 1)²) = 1.0 exactly.
        let loss = scalar(&mse_loss(&zeros, &ones));
        assert_eq!(loss, 1.0, "MSE of an all-zero/all-one pair must equal 1.0");
    }

    #[test]
    fn mse_loss_gradient_matches_closed_form() {
        // ∂MSE/∂fixed = (2/N)·(fixed − moving). With fixed = 2·moving over N = 8
        // voxels: each component is (2/8)·moving = moving/4.
        let moving = from_vals((0..8).map(|i| i as f64).collect(), 2);
        let fixed = from_vals((0..8).map(|i| 2.0 * i as f64).collect(), 2);
        let loss = mse_loss(&fixed, &moving);
        loss.backward();
        let g = fixed.grad().expect("fixed grad");
        for (i, &gv) in g.as_slice().iter().enumerate() {
            let expected = (2.0 / 8.0) * (2.0 * i as f64 - i as f64);
            assert!(
                (gv - expected).abs() < 1e-9,
                "∂MSE/∂fixed[{i}] = {gv}, expected {expected}"
            );
        }
    }

    #[test]
    fn ncc_loss_identical_non_constant_images_is_near_negative_one() {
        let image = ascending_volume(4);
        // Self-correlation is exactly 1, so the negated NCC loss is exactly −1
        // up to the numerical-stability epsilon in the denominator.
        let loss = scalar(&ncc_loss(&image, &image));
        assert!(
            (loss - (-1.0)).abs() < 1e-3,
            "NCC of an image with itself should be ~−1.0, got {loss}"
        );
    }

    #[test]
    fn lncc_loss_identical_non_constant_images_is_near_negative_one() {
        let image = ascending_volume(5);
        let loss = scalar(&lncc_loss(&image, &image, 3));
        assert!(
            (loss - (-1.0)).abs() < 1e-2,
            "LNCC of an image with itself should be ~−1.0, got {loss}"
        );
    }

    #[test]
    fn mi_loss_self_information_exceeds_unrelated_images() {
        let d = 4;
        let n = d * d * d;
        // A genuinely unrelated image: a low-period repeating pattern. Unlike a
        // monotonic transform of `image` (which carries identical MI, since MI
        // is invariant under invertible per-variable maps), this many-to-one
        // mapping breaks the voxel-wise correspondence.
        let unrelated: Vec<f64> = (0..n).map(|i| (i % 3) as f64).collect();

        // Normalize both to [0, 1] as mi_loss assumes.
        let max = (n - 1) as f64;
        let norm_image = from_vals((0..n).map(|i| i as f64 / max).collect(), d);
        let norm_other = from_vals(unrelated.iter().map(|&v| v / 2.0).collect(), d);

        let self_mi = scalar(&mi_loss(&norm_image, &norm_image, 8, 0.1));
        let cross_mi = scalar(&mi_loss(&norm_image, &norm_other, 8, 0.1));
        assert!(self_mi.is_finite() && cross_mi.is_finite());
        // Self-MI (negated) must be the more negative of the two: an image
        // carries maximal information about itself.
        assert!(
            self_mi < cross_mi,
            "negated self-MI ({self_mi}) should be < negated cross-MI ({cross_mi})"
        );
    }
}
