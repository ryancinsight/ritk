use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_image::native::Image as NativeImage;
use ritk_image::tensor::backend::Backend;
use ritk_image::Image;

/// Compute the Dice similarity coefficient between two binary segmentation masks.
///
/// # Arguments
/// * `prediction` - Binary mask with values 0.0 or 1.0.
/// * `ground_truth` - Binary mask with values 0.0 or 1.0.
///
/// # Returns
/// Dice score in `[0.0, 1.0]`. Returns 1.0 when both masks are empty.
///
/// # Formula
/// `Dice = 2 * |P intersect G| / (|P| + |G|)`
pub fn dice_coefficient<B: Backend, const D: usize>(
    prediction: &Image<B, D>,
    ground_truth: &Image<B, D>,
) -> f32 {
    let intersection_sum: f32 = {
        let t = prediction.data().clone() * ground_truth.data().clone();
        let d = t.sum().into_data();
        d.as_slice::<f32>().expect("f32 intersection tensor")[0]
    };

    let pred_vol: f32 = {
        let d = prediction.data().clone().sum().into_data();
        d.as_slice::<f32>().expect("f32 prediction sum")[0]
    };

    let gt_vol: f32 = {
        let d = ground_truth.data().clone().sum().into_data();
        d.as_slice::<f32>().expect("f32 ground truth sum")[0]
    };

    let denom = pred_vol + gt_vol;
    if denom < f32::EPSILON {
        return 1.0;
    }

    2.0 * intersection_sum / denom
}

/// Compute Dice overlap between two Coeus-native binary masks.
///
/// # Errors
/// Returns an error when either image is not CPU-addressable or the images
/// have different element counts.
pub fn dice_coefficient_native<B, const D: usize>(
    prediction: &NativeImage<f32, B, D>,
    ground_truth: &NativeImage<f32, B, D>,
) -> anyhow::Result<f32>
where
    B: ComputeBackend,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    let prediction_values = prediction.data_slice()?;
    let ground_truth_values = ground_truth.data_slice()?;
    anyhow::ensure!(
        prediction_values.len() == ground_truth_values.len(),
        "dice coefficient requires equal element counts: {} != {}",
        prediction_values.len(),
        ground_truth_values.len()
    );
    let (intersection, prediction_sum, ground_truth_sum) =
        prediction_values.iter().zip(ground_truth_values).fold(
            (0.0_f32, 0.0_f32, 0.0_f32),
            |(intersection, prediction, ground_truth), (&left, &right)| {
                (
                    intersection + left * right,
                    prediction + left,
                    ground_truth + right,
                )
            },
        );
    let denominator = prediction_sum + ground_truth_sum;
    Ok(if denominator < f32::EPSILON {
        1.0
    } else {
        2.0 * intersection / denominator
    })
}

/// Compute the ITK `SimilarityIndexImageFilter` overlap (`sitk.SimilarityIndex`).
///
/// Unlike [`dice_coefficient`], this binarizes both inputs — **any nonzero
/// voxel is foreground** — so it is correct for multi-valued label maps, and it
/// returns `0.0` (not `1.0`) when both foreground sets are empty, matching ITK.
///
/// # Formula
/// `SI = 2 * |A intersect B| / (|A| + |B|)` over the binarized sets
/// `A = {x : a(x) != 0}`, `B = {x : b(x) != 0}`.
pub fn similarity_index<B: Backend, const D: usize>(a: &Image<B, D>, b: &Image<B, D>) -> f32 {
    let sa = a.data_slice();
    let sb = b.data_slice();
    let (mut count_a, mut count_b, mut inter) = (0u64, 0u64, 0u64);
    for (&va, &vb) in sa.iter().zip(sb.iter()) {
        let fa = va != 0.0;
        let fb = vb != 0.0;
        count_a += fa as u64;
        count_b += fb as u64;
        inter += (fa && fb) as u64;
    }
    let denom = count_a + count_b;
    if denom == 0 {
        return 0.0;
    }
    (2.0 * inter as f64 / denom as f64) as f32
}
