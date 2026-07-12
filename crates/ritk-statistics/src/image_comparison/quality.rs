use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_image::native::Image as NativeImage;
use ritk_image::tensor::backend::Backend;
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;

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

/// Compute PSNR between two Coeus-native images.
///
/// Returns positive infinity for identical inputs.
pub fn psnr_native<B, const D: usize>(
    image: &NativeImage<f32, B, D>,
    reference: &NativeImage<f32, B, D>,
    max_val: f32,
) -> anyhow::Result<f32>
where
    B: ComputeBackend,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    let image_values = image.data_slice()?;
    let reference_values = reference.data_slice()?;
    anyhow::ensure!(
        image_values.len() == reference_values.len(),
        "psnr requires equal element counts: {} != {}",
        image_values.len(),
        reference_values.len()
    );
    let sum_squared_error = image_values
        .iter()
        .zip(reference_values)
        .map(|(&value, &reference)| {
            let delta = value - reference;
            delta * delta
        })
        .sum::<f32>();
    let mean_squared_error = sum_squared_error / image_values.len() as f32;
    Ok(if mean_squared_error < f32::EPSILON {
        f32::INFINITY
    } else {
        10.0 * (max_val * max_val / mean_squared_error).log10()
    })
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
