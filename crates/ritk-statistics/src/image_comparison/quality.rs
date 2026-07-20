use coeus_core::CpuAddressableStorage;
use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;

/// Compute the Pearson correlation coefficient between equal-length images.
///
/// The two-pass computation uses f64 accumulators and Moirai's indexed
/// reduction so the mean and centered covariance share one parallel execution
/// policy without allocating intermediate buffers.
pub fn pearson_correlation(image: &[f32], reference: &[f32]) -> anyhow::Result<f64> {
    anyhow::ensure!(
        image.len() == reference.len(),
        "pearson correlation requires equal element counts: {} != {}",
        image.len(),
        reference.len()
    );
    anyhow::ensure!(
        !image.is_empty(),
        "pearson correlation requires at least one element"
    );

    let (image_sum, reference_sum) = moirai::fold_reduce_with::<moirai::Adaptive, _, _, _, _>(
        image.len(),
        || (0.0_f64, 0.0_f64),
        |(image_acc, reference_acc), index| {
            (
                image_acc + image[index] as f64,
                reference_acc + reference[index] as f64,
            )
        },
        |(image_left, reference_left), (image_right, reference_right)| {
            (image_left + image_right, reference_left + reference_right)
        },
    );
    let count = image.len() as f64;
    let image_mean = image_sum / count;
    let reference_mean = reference_sum / count;

    let (covariance, image_variance, reference_variance) =
        moirai::fold_reduce_with::<moirai::Adaptive, _, _, _, _>(
            image.len(),
            || (0.0_f64, 0.0_f64, 0.0_f64),
            |(covariance_acc, image_acc, reference_acc), index| {
                let image_delta = image[index] as f64 - image_mean;
                let reference_delta = reference[index] as f64 - reference_mean;
                (
                    covariance_acc + image_delta * reference_delta,
                    image_acc + image_delta * image_delta,
                    reference_acc + reference_delta * reference_delta,
                )
            },
            |(covariance_left, image_left, reference_left),
             (covariance_right, image_right, reference_right)| {
                (
                    covariance_left + covariance_right,
                    image_left + image_right,
                    reference_left + reference_right,
                )
            },
        );

    let denominator = (image_variance * reference_variance).sqrt();
    Ok(if denominator == 0.0 {
        0.0
    } else {
        (covariance / denominator).clamp(-1.0, 1.0)
    })
}

/// PSNR over two flat host buffers of length `n`.
///
/// Shared host core for the Coeus-backed [`psnr`] and the Coeus-native
/// `native::psnr`. Returns `f32::INFINITY` when MSE is (near) zero.
///
/// # Formula
/// `MSE = (1/n) Σ (Iᵢ − Rᵢ)²`, `PSNR = 10·log₁₀(MAX² / MSE)`.
///
/// # Precision
/// The squared-error sum accumulates in `f64` (the Coeus path reduces in `f32`);
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
/// window). Shared host core for the Coeus-backed [`ssim`] and the Coeus-native
/// `native::ssim`.
pub(crate) fn ssim_from_slices(
    image: &[f32],
    reference: &[f32],
    max_val: f32,
) -> anyhow::Result<f32> {
    anyhow::ensure!(
        image.len() == reference.len(),
        "ssim requires equal element counts: {} != {}",
        image.len(),
        reference.len()
    );
    anyhow::ensure!(!image.is_empty(), "ssim requires at least one element");

    let n = image.len() as f64;

    let mu_x: f64 = image.iter().map(|&v| v as f64).sum::<f64>() / n;
    let mu_y: f64 = reference.iter().map(|&v| v as f64).sum::<f64>() / n;

    let sigma_x_sq: f64 = image
        .iter()
        .map(|&v| {
            let d = v as f64 - mu_x;
            d * d
        })
        .sum::<f64>()
        / n;

    let sigma_y_sq: f64 = reference
        .iter()
        .map(|&v| {
            let d = v as f64 - mu_y;
            d * d
        })
        .sum::<f64>()
        / n;

    let sigma_xy: f64 = image
        .iter()
        .zip(reference.iter())
        .map(|(&x, &y)| (x as f64 - mu_x) * (y as f64 - mu_y))
        .sum::<f64>()
        / n;

    let c1 = (0.01 * max_val as f64).powi(2);
    let c2 = (0.03 * max_val as f64).powi(2);

    let numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2);
    let denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x_sq + sigma_y_sq + c2);

    Ok((numerator / denominator) as f32)
}

/// Compute the Peak Signal-to-Noise Ratio (PSNR) between two images.
///
/// Returns `f32::INFINITY` when the images are identical.
pub fn psnr<B: Backend, const D: usize>(
    image: &Image<f32, B, D>,
    reference: &Image<f32, B, D>,
    max_val: f32,
) -> f32
where
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    let (img, _) = extract_vec_infallible(image);
    let (ref_, _) = extract_vec_infallible(reference);
    psnr_from_slices(&img, &ref_, max_val)
}

/// Compute the global Structural Similarity Index (SSIM) between two images.
///
/// Uses Wang et al. (2004) computed over all voxels as a single global window.
pub fn ssim<B: Backend, const D: usize>(
    image: &Image<f32, B, D>,
    reference: &Image<f32, B, D>,
    max_val: f32,
) -> f32
where
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    let img_slice: &[f32] = &extract_vec_infallible(image).0;
    let ref_slice: &[f32] = &extract_vec_infallible(reference).0;
    ssim_from_slices(img_slice, ref_slice, max_val)
        .expect("invariant: legacy image shapes have equal element counts")
}
