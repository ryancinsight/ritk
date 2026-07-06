use ritk_image::burn::module::Param;
use ritk_image::burn::nn::conv::Conv3dConfig;
use ritk_image::burn::nn::PaddingConfig3d;
use ritk_image::tensor::Backend;
use ritk_image::tensor::Tensor;

/// Mean Squared Error Loss for 5D tensors [Batch, Channel, D, H, W].
///
/// Computes the mean squared difference between fixed and moving images.
pub fn mse_loss<B: Backend>(fixed: Tensor<B, 5>, moving: Tensor<B, 5>) -> Tensor<B, 1> {
    let diff = fixed - moving;
    diff.powf_scalar(2.0).mean()
}

/// Global Normalized Cross Correlation Loss for 5D tensors.
///
/// Computes the negative NCC (to be minimized).
/// Range: [-1, 1] (where -1 is perfect correlation).
pub fn ncc_loss<B: Backend>(fixed: Tensor<B, 5>, moving: Tensor<B, 5>) -> Tensor<B, 1> {
    // Check shapes
    let dims = fixed.shape().dims; // [B, C, D, H, W]
    let b = dims[0];
    let c = dims[1];
    let d = dims[2];
    let h = dims[3];
    let w = dims[4];

    let n = d * h * w;

    let fixed_flat = fixed.reshape([b, c, n]);
    let moving_flat = moving.reshape([b, c, n]);

    let mean_f = fixed_flat.clone().mean_dim(2); // [B, C, 1]
    let mean_m = moving_flat.clone().mean_dim(2); // [B, C, 1]

    let f_centered = fixed_flat - mean_f;
    let m_centered = moving_flat - mean_m;

    // Covariance: sum((F-meanF)*(M-meanM))
    let num = (f_centered.clone() * m_centered.clone()).sum_dim(2); // [B, C, 1]

    // Variances: sum((F-meanF)^2), sum((M-meanM)^2)
    let denom_f = f_centered.powf_scalar(2.0).sum_dim(2); // [B, C, 1]
    let denom_m = m_centered.powf_scalar(2.0).sum_dim(2); // [B, C, 1]

    let epsilon = 1e-5;
    let denom = (denom_f * denom_m).sqrt() + epsilon;

    let ncc = num / denom; // [B, C, 1]

    // Average over batch and channel
    ncc.mean().neg()
}

/// Local Normalized Cross Correlation (LNCC) Loss for 5D tensors.
///
/// Uses a box filter (average pooling) to compute local statistics.
/// Robust to local intensity variations and bias fields.
///
/// # Arguments
/// * `fixed` - Fixed image tensor [Batch, Channel, D, H, W]
/// * `moving` - Moving image tensor [Batch, Channel, D, H, W]
/// * `kernel_size` - Size of the local window (e.g., 9)
///
/// # Returns
/// * Negative LNCC loss (scalar tensor)
pub fn lncc_loss<B: Backend>(
    fixed: Tensor<B, 5>,
    moving: Tensor<B, 5>,
    kernel_size: usize,
) -> Tensor<B, 1> {
    let padding = kernel_size / 2;
    let k = kernel_size;
    let c = fixed.shape().dims[1];
    let device = fixed.device();
    let weight_val = 1.0 / (k * k * k) as f64;

    // Build the depthwise box-filter conv once; all five forward calls share it.
    // Conv3d::forward takes &self so the single module is reused without cloning.
    let mut conv = Conv3dConfig::new([c, c], [k, k, k])
        .with_groups(c)
        .with_bias(false)
        .with_padding(PaddingConfig3d::Explicit(padding, padding, padding))
        .init(&device);
    let w_shape = conv.weight.shape();
    conv.weight = Param::from_tensor(Tensor::full(w_shape, weight_val, &device));

    // E[F], E[M]: two tensor clones of the originals (consumed by E[FM] last).
    let mean_f = conv.forward(fixed.clone());
    let mean_m = conv.forward(moving.clone());
    // E[F²], E[M²]: two more clones of the originals (last use before consumption below).
    let mean_f2 = conv.forward(fixed.clone().powf_scalar(2.0));
    let mean_m2 = conv.forward(moving.clone().powf_scalar(2.0));
    // E[FM]: consumes fixed and moving (no further clones needed).
    let mean_fm = conv.forward(fixed * moving);

    // Var(X) = E[X²] − (E[X])² — clones of mean_f/mean_m retained for cov_fm below.
    let var_f = mean_f2 - mean_f.clone().powf_scalar(2.0);
    let var_m = mean_m2 - mean_m.clone().powf_scalar(2.0);
    // Cov(F, M) = E[FM] − E[F]·E[M] — consumes mean_f and mean_m.
    let cov_fm = mean_fm - (mean_f * mean_m);

    let epsilon = 1e-5_f64;
    let cc = cov_fm / ((var_f * var_m).sqrt() + epsilon);
    cc.mean().neg()
}

/// Mutual Information Loss (Approximated via Soft Histograms).
///
/// Note: This is computationally expensive and memory intensive for 3D volumes.
/// Consider using LNCC or NCC for better performance.
///
/// # Arguments
/// * `fixed` - Fixed image tensor [Batch, Channel, D, H, W]
/// * `moving` - Moving image tensor [Batch, Channel, D, H, W]
/// * `num_bins` - Number of histogram bins (e.g., 32)
/// * `sigma` - Sigma for soft binning (e.g., 0.1)
pub fn mi_loss<B: Backend>(
    fixed: Tensor<B, 5>,
    moving: Tensor<B, 5>,
    num_bins: usize,
    sigma: f64,
) -> Tensor<B, 1> {
    // Normalize images to [0, 1] range roughly for binning
    // We assume inputs are roughly in [0, 1] or normalize them.
    // For now, we assume they are.

    let dims = fixed.shape().dims;
    let b = dims[0];
    let c = dims[1];
    let d = dims[2];
    let h = dims[3];
    let w = dims[4];

    let num_samples = (b * c * d * h * w) as f32;
    let device = fixed.device();

    // Flatten
    let f_flat = fixed.reshape([num_samples as usize]);
    let m_flat = moving.reshape([num_samples as usize]);

    // Bin centers
    // 0 to 1
    // This part is tricky in Burn without a "linspace" or "arange" easily accessible in tensor ops for all backends.
    // We can construct it.
    let bins = Tensor::arange(0..num_bins as i64, &device).float() / (num_bins as f32 - 1.0);

    // Soft Histogram
    // P(i) = sum(sigmoid((I - bin_i + width)/sigma) - sigmoid((I - bin_i - width)/sigma))
    // Or RBF: exp(-(x - bin)^2 / (2sigma^2))

    // We need to broadcast: [Samples, 1] vs [1, Bins]
    let f_expanded = f_flat.clone().unsqueeze_dim(1); // [N, 1]
    let m_expanded = m_flat.clone().unsqueeze_dim(1); // [N, 1]
    let bins_expanded = bins.clone().unsqueeze_dim(0); // [1, Bins]

    // This creates [N, Bins] matrix - VERY LARGE for 3D images (e.g. 128^3 = 2M pixels * 32 bins = 64M floats = 256MB)
    // Feasible for moderate sizes.

    let compute_weights = |x: Tensor<B, 2>| -> Tensor<B, 2> {
        let diff = x - bins_expanded.clone();
        let exponent = diff.powf_scalar(2.0).neg() / (2.0 * sigma * sigma);
        exponent.exp()
    };

    let w_f = compute_weights(f_expanded); // [N, Bins]
    let w_m = compute_weights(m_expanded); // [N, Bins]

    // Joint Histogram
    // P(i, j) = sum_k (w_f[k, i] * w_m[k, j])
    // Matrix multiplication: W_f^T * W_m -> [Bins, Bins]
    let joint_hist = w_f.transpose().matmul(w_m); // [Bins, Bins]

    // Normalize
    let joint_prob = joint_hist / num_samples;

    // Marginals
    let p_f = joint_prob.clone().sum_dim(1); // [Bins, 1]
    let p_m = joint_prob.clone().sum_dim(0); // [1, Bins]

    // Entropy
    // H(F, M) = - sum P(i,j) log P(i,j)
    // MI(F, M) = H(F) + H(M) - H(F, M)
    // Or sum P(i,j) log (P(i,j) / (P(i)P(j)))

    let epsilon = 1e-10;

    // MI = sum( P(i,j) * (log(P(i,j)) - log(P(i)) - log(P(j))) )

    let log_p_f = (p_f + epsilon).log();
    let log_p_m = (p_m + epsilon).log();
    let log_joint = (joint_prob.clone() + epsilon).log();

    // Broadcast marginals to [Bins, Bins]
    // p_f is [Bins, 1], p_m is [1, Bins]

    let term = log_joint - log_p_f - log_p_m;
    let mi = (joint_prob * term).sum();

    // Return negative MI
    mi.neg()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    /// Build a [1, 1, d, d, d] non-constant tensor (ascending values) so
    /// correlation-based losses are well-defined (zero-variance inputs make
    /// NCC/LNCC's denominator vanish).
    fn ascending_volume(d: usize, device: &<B as Backend>::Device) -> Tensor<B, 5> {
        let n = d * d * d;
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape([1, 1, d, d, d])
    }

    #[test]
    fn mse_loss_identical_images_is_exactly_zero() {
        let device = Default::default();
        let image = ascending_volume(3, &device);
        let loss = mse_loss(image.clone(), image).into_scalar();
        assert_eq!(loss, 0.0, "MSE of identical images must be exactly zero");
    }

    #[test]
    fn mse_loss_known_constant_difference_matches_closed_form() {
        let device = Default::default();
        let zeros = Tensor::<B, 5>::zeros([1, 1, 2, 2, 2], &device);
        let ones = Tensor::<B, 5>::ones([1, 1, 2, 2, 2], &device);
        // mean((0 - 1)^2) = 1.0 exactly.
        let loss = mse_loss(zeros, ones).into_scalar();
        assert_eq!(loss, 1.0, "MSE of an all-zero/all-one pair must equal 1.0");
    }

    #[test]
    fn ncc_loss_identical_non_constant_images_is_near_negative_one() {
        let device = Default::default();
        let image = ascending_volume(4, &device);
        // Self-correlation is exactly 1, so the negated NCC loss is exactly -1
        // up to the numerical-stability epsilon added to the denominator.
        let loss = ncc_loss(image.clone(), image).into_scalar();
        assert!(
            (loss - (-1.0)).abs() < 1e-3,
            "NCC of an image with itself should be ~-1.0, got {loss}"
        );
    }

    #[test]
    fn lncc_loss_identical_non_constant_images_is_near_negative_one() {
        let device = Default::default();
        let image = ascending_volume(5, &device);
        let loss = lncc_loss(image.clone(), image, 3).into_scalar();
        assert!(
            (loss - (-1.0)).abs() < 1e-2,
            "LNCC of an image with itself should be ~-1.0, got {loss}"
        );
    }

    #[test]
    fn mi_loss_self_information_exceeds_unrelated_images() {
        let device = Default::default();
        let d = 4;
        let image = ascending_volume(d, &device);
        let n = d * d * d;
        // A genuinely unrelated image: a low-period repeating pattern. Unlike
        // a monotonic transform of `image` (which would carry identical MI,
        // since MI is invariant under invertible per-variable maps), this
        // many-to-one mapping breaks the voxel-wise correspondence.
        let unrelated: Vec<f32> = (0..n).map(|i| (i % 3) as f32).collect();
        let other =
            Tensor::<B, 1>::from_floats(unrelated.as_slice(), &device).reshape([1, 1, d, d, d]);

        // Normalize both to [0, 1] as mi_loss assumes.
        let max = (n - 1) as f32;
        let norm_image = image.clone() / max;
        let norm_other = other / 2.0;

        let self_mi = mi_loss(norm_image.clone(), norm_image.clone(), 8, 0.1).into_scalar();
        let cross_mi = mi_loss(norm_image, norm_other, 8, 0.1).into_scalar();
        assert!(self_mi.is_finite() && cross_mi.is_finite());
        // Self-MI (negated) must be the more negative of the two: an image
        // carries maximal information about itself.
        assert!(
            self_mi < cross_mi,
            "negated self-MI ({self_mi}) should be < negated cross-MI ({cross_mi})"
        );
    }
}
