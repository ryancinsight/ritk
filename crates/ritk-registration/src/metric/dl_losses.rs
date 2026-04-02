use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn::nn::conv::Conv3dConfig;
use burn::nn::PaddingConfig3d;
use burn::module::Param;

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
    kernel_size: usize
) -> Tensor<B, 1> {
    let padding = kernel_size / 2;
    
    // Helper for box filtering (local mean)
    let box_filter = |x: Tensor<B, 5>| -> Tensor<B, 5> {
        let dims = x.shape().dims;
        let c = dims[1];
        let k = kernel_size;
        
        // Depthwise convolution with constant weights = 1/N
        let weight_val = 1.0 / (k * k * k) as f64;
        
        let mut conv = Conv3dConfig::new([c, c], [k, k, k])
            .with_groups(c)
            .with_bias(false)
            .with_padding(PaddingConfig3d::Explicit(padding, padding, padding))
            .init(&x.device());
            
        let w_shape = conv.weight.shape();
        let w = Tensor::full(w_shape, weight_val, &x.device());
        conv.weight = Param::from_tensor(w);
        
        conv.forward(x)
    };

    // 1. Compute local sums (via means * window_size)
    // Actually avg_pool3d computes the mean directly.
    let mean_f = box_filter(fixed.clone());
    let mean_m = box_filter(moving.clone());
    
    // 2. Compute local variances
    // Var(X) = E[X^2] - (E[X])^2
    let mean_f2 = box_filter(fixed.clone().powf_scalar(2.0));
    let mean_m2 = box_filter(moving.clone().powf_scalar(2.0));
    
    let var_f = mean_f2 - mean_f.clone().powf_scalar(2.0);
    let var_m = mean_m2 - mean_m.clone().powf_scalar(2.0);
    
    // 3. Compute local covariance
    // Cov(X, Y) = E[XY] - E[X]E[Y]
    let mean_fm = box_filter(fixed * moving);
    let cov_fm = mean_fm - (mean_f * mean_m);
    
    // 4. Compute LNCC
    let epsilon = 1e-5;
    let cc = cov_fm / ((var_f * var_m).sqrt() + epsilon);
    
    // 5. Average over all pixels/batches/channels
    // VoxelMorph squares it before averaging? 
    // "We minimize the negative mean of the local cross correlation."
    // Usually standard LNCC is just averaged.
    
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
    sigma: f64
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
