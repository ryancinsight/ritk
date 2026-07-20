use coeus_core::{Backend, CpuAddressableStorage};
use ritk_image::Image;

/// Dice similarity coefficient over two flat host buffers.
///
/// This is the single host implementation shared by the Coeus-backed
/// [`dice_coefficient`] and the Coeus-native `native::dice_coefficient`; both
/// backend adapters extract contiguous `&[f32]` storage and delegate here so the
/// overlap math has exactly one home.
///
/// # Formula
/// `Dice = 2 * |P ∩ G| / (|P| + |G|)` with `|·|` the sum of voxel values (masks
/// are 0.0/1.0). Returns `1.0` when both masks are empty (`|P| + |G| < ε`),
/// matching the empty-set convention.
///
/// # Precision
/// The intersection and volumes accumulate in `f64` to keep the ratio exact for
/// large masks (a plain `f32` running sum over ~10⁷ unit voxels loses the low
/// bits once the partial sum exceeds `2²⁴`).
pub(crate) fn dice_from_slices(prediction: &[f32], ground_truth: &[f32]) -> f32 {
    let mut intersection = 0.0_f64;
    let mut pred_vol = 0.0_f64;
    let mut gt_vol = 0.0_f64;
    for (&p, &g) in prediction.iter().zip(ground_truth.iter()) {
        intersection += (p * g) as f64;
        pred_vol += p as f64;
        gt_vol += g as f64;
    }

    let denom = pred_vol + gt_vol;
    if denom < f32::EPSILON as f64 {
        return 1.0;
    }
    (2.0 * intersection / denom) as f32
}

/// ITK `SimilarityIndex` over two flat host buffers (binarized: any nonzero voxel
/// is foreground). Shared host core for the Coeus and Coeus-native adapters.
///
/// # Formula
/// `SI = 2 * |A ∩ B| / (|A| + |B|)` over `A = {x : a(x) ≠ 0}`,
/// `B = {x : b(x) ≠ 0}`; returns `0.0` when both foreground sets are empty
/// (ITK convention, distinct from Dice's `1.0`).
pub(crate) fn similarity_index_from_slices(a: &[f32], b: &[f32]) -> f32 {
    let (mut count_a, mut count_b, mut inter) = (0u64, 0u64, 0u64);
    for (&va, &vb) in a.iter().zip(b.iter()) {
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
pub fn dice_coefficient<B, const D: usize>(
    prediction: &Image<f32, B, D>,
    ground_truth: &Image<f32, B, D>,
) -> anyhow::Result<f32>
where
    B: Backend + Default,
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
    let (intersection, prediction_sum, ground_truth_sum) = prediction_values
        .iter()
        .zip(ground_truth_values.iter())
        .fold(
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
pub fn similarity_index<B, const D: usize>(
    a: &Image<f32, B, D>,
    b: &Image<f32, B, D>,
) -> anyhow::Result<f32>
where
    B: Backend + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    let a_values = a.data_slice()?;
    let b_values = b.data_slice()?;
    Ok(similarity_index_from_slices(a_values, b_values))
}
