use ritk_image::Image;
use burn::tensor::backend::Backend;

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
