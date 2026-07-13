use ritk_image::tensor::backend::Backend;
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;

/// PSNR over two flat host buffers of length `n`.
///
/// Shared host core for the Burn-backed [`psnr`] and the Coeus-native
/// `native::psnr`. Returns `f32::INFINITY` when MSE is (near) zero.
///
/// # Formula
/// `MSE = (1/n) Σ (Iᵢ − Rᵢ)²`, `PSNR = 10·log₁₀(MAX² / MSE)`.
///
/// # Precision
/// The squared-error sum accumulates in `f64` (the Burn path reduces in `f32`);
/// the two agree to the `f32` epsilon of the ratio, which the differential tests
/// bound.
pub(crate) fn psnr_from_slices(image: &[f32], reference: &[f32], max_val: f32) -> f32 {
    let n = image.len() as f64;
    let sum_sq: f64 = image
        .iter()
        .zip(reference.iter())
        .map(|(&i, &r)| {
            let d = i as f64 - r as f64;
            d * d
        })
        .sum();
    let mse = sum_sq / n;

    if mse < f32::EPSILON as f64 {
        return f32::INFINITY;
    }
    (10.0 * (max_val as f64 * max_val as f64 / mse).log10()) as f32
}

/// Global SSIM over two flat host buffers (Wang et al. 2004, single global
/// window). Shared host core for the Burn-backed [`ssim`] and the Coeus-native
/// `native::ssim`.
pub(crate) fn ssim_from_slices(img_slice: &[f32], ref_slice: &[f32], max_val: f32) -> f32 {
    let n = img_slice.len() as f64;

    let mu_x: f64 = img_slice.iter().map(|&v| v as f64).sum::<f64>() / n;
    let mu_y: f64 = ref_slice.iter().map(|&v| v as f64).sum::<f64>() / n;

    let sigma_x_sq: f64 = img_slice
        .iter()
        .map(|&v| {
            let d = v as f64 - mu_x;
            d * d
        })
        .sum::<f64>()
        / n;

    let sigma_y_sq: f64 = ref_slice
        .iter()
        .map(|&v| {
            let d = v as f64 - mu_y;
            d * d
        })
        .sum::<f64>()
        / n;

    let sigma_xy: f64 = img_slice
        .iter()
        .zip(ref_slice.iter())
        .map(|(&x, &y)| (x as f64 - mu_x) * (y as f64 - mu_y))
        .sum::<f64>()
        / n;

    let c1 = (0.01 * max_val as f64).powi(2);
    let c2 = (0.03 * max_val as f64).powi(2);

    let numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2);
    let denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x_sq + sigma_y_sq + c2);

    (numerator / denominator) as f32
}

/// Compute the Peak Signal-to-Noise Ratio (PSNR) between two images.
///
/// Returns `f32::INFINITY` when the images are identical.
pub fn psnr<B: Backend, const D: usize>(
    image: &Image<B, D>,
    reference: &Image<B, D>,
    max_val: f32,
) -> f32 {
    let diff = image.data().clone() - reference.data().clone();
    let sq_diff = diff.clone() * diff;
    let sum_sq_data = sq_diff.sum().into_data();
    let sum_sq: f32 = sum_sq_data
        .as_slice::<f32>()
        .expect("f32 sum of squared differences")[0];

    let n: f32 = image.shape().iter().product::<usize>() as f32;
    let mse = sum_sq / n;

    if mse < f32::EPSILON {
        return f32::INFINITY;
    }

    10.0 * (max_val * max_val / mse).log10()
}

/// Compute the global Structural Similarity Index (SSIM) between two images.
///
/// Uses Wang et al. (2004) computed over all voxels as a single global window.
pub fn ssim<B: Backend, const D: usize>(
    image: &Image<B, D>,
    reference: &Image<B, D>,
    max_val: f32,
) -> f32 {
    let img_slice: &[f32] = &extract_vec_infallible(image).0;
    let ref_slice: &[f32] = &extract_vec_infallible(reference).0;
    ssim_from_slices(img_slice, ref_slice, max_val)
}
