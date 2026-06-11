//! Per-label overlap measures suite for multi-label segmentation evaluation.
//!
//! # Mathematical Specification
//!
//! For label k given prediction mask P and ground-truth mask G of identical shape:
//!
//! Let:
//!   TP_k = |{x: P(x)=k ∧ G(x)=k}|
//!   FP_k = |{x: P(x)=k ∧ G(x)≠k}|
//!   FN_k = |{x: P(x)≠k ∧ G(x)=k}|
//!   TN_k = N − TP_k − FP_k − FN_k
//!   V_P_k = TP_k + FP_k    (predicted volume for label k)
//!   V_G_k = TP_k + FN_k    (ground-truth volume for label k)
//!
//! Metrics (all in \[0,1\]):
//!   dice_k             = 2·TP_k / (V_P_k + V_G_k);       1.0 when both empty
//!   jaccard_k          = TP_k / (TP_k + FP_k + FN_k);    1.0 when both empty
//!   volume_similarity_k = 1 − |V_P_k − V_G_k| / (V_P_k + V_G_k); 1.0 when both empty
//!   false_negative_rate_k = FN_k / V_G_k;                0.0 when V_G_k = 0
//!   false_positive_rate_k = FP_k / (FP_k + TN_k);        0.0 when (FP_k + TN_k) = 0
//!   sensitivity_k      = TP_k / V_G_k;                   1.0 when V_G_k = 0
//!   specificity_k      = TN_k / (TN_k + FP_k);           1.0 when (TN_k + FP_k) = 0
//!
//! Background label (0) is excluded.  All labels present in either mask are reported.
//! Complexity: O(N) single parallel pass.
//!
//! # ITK Parity
//! `itk::LabelOverlapMeasuresImageFilter` (ITK 5.x).

use crate::filter::ops::extract_vec_infallible;
use ritk_image::Image;
use burn::tensor::backend::Backend;
use std::collections::HashMap;

// ── Public types ──────────────────────────────────────────────────────────────

/// Overlap measures for a single label.
#[derive(Debug, Clone, PartialEq)]
pub struct LabelOverlapMeasures {
    /// Label index (≥ 1; background 0 is excluded).
    pub label: u32,
    /// Sørensen–Dice coefficient: 2·TP / (V_P + V_G). Range \[0,1\].
    pub dice: f32,
    /// Jaccard / Union Overlap: TP / (TP + FP + FN). Range \[0,1\].
    pub jaccard: f32,
    /// Volumetric similarity: 1 − |V_P − V_G| / (V_P + V_G). Range \[0,1\].
    pub volume_similarity: f32,
    /// False-negative rate: FN / V_G. Range \[0,1\].
    pub false_negative_rate: f32,
    /// False-positive rate: FP / (FP + TN). Range \[0,1\].
    pub false_positive_rate: f32,
    /// Sensitivity / recall: TP / V_G. Range \[0,1\].
    pub sensitivity: f32,
    /// Specificity: TN / (TN + FP). Range \[0,1\].
    pub specificity: f32,
    /// Predicted volume (voxel count) for this label.
    pub predicted_volume: usize,
    /// Ground-truth volume (voxel count) for this label.
    pub ground_truth_volume: usize,
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Compute per-label overlap measures from two N-D images.
///
/// Label values in `prediction` and `ground_truth` are cast to `u32` via
/// truncation. Background (label 0) is excluded from results.
///
/// # Panics
/// Panics if the two images have different shapes.
pub fn label_overlap_measures<B: Backend, const D: usize>(
    prediction: &Image<B, D>,
    ground_truth: &Image<B, D>,
) -> Vec<LabelOverlapMeasures> {
    assert_eq!(
        prediction.shape(),
        ground_truth.shape(),
        "prediction and ground_truth must have identical shapes"
    );
    let (pred_vals, _) = extract_vec_infallible(prediction);
    let (gt_vals, _) = extract_vec_infallible(ground_truth);
    label_overlap_measures_from_slices(&pred_vals, &gt_vals)
}

/// Compute per-label overlap measures from pre-extracted flat slices.
///
/// Zero-copy variant. Label values are cast to `u32` via truncation.
/// Background (label 0) is excluded.
///
/// # Panics
/// Panics if slices have different lengths.
pub fn label_overlap_measures_from_slices(
    pred_slice: &[f32],
    gt_slice: &[f32],
) -> Vec<LabelOverlapMeasures> {
    assert_eq!(
        pred_slice.len(),
        gt_slice.len(),
        "pred_slice and gt_slice must have equal length"
    );
    let n = pred_slice.len();

    // Parallel accumulation: (tp, fp, fn_) per label.
    // HashMap<label, (tp: usize, fp: usize, fn_: usize)>
    type Acc = (usize, usize, usize); // (TP, FP, FN)

    let combined: HashMap<u32, Acc> = moirai::fold_reduce_with::<moirai::Adaptive, _, _, _, _>(
        pred_slice.len(),
        HashMap::<u32, Acc>::new,
        |mut acc, i| {
            let p = pred_slice[i] as u32;
            let g = gt_slice[i] as u32;
            // Contribution for predicted label (if non-background)
            if p != 0 {
                let e = acc.entry(p).or_insert((0, 0, 0));
                if g == p {
                    e.0 += 1; // TP for p
                } else {
                    e.1 += 1; // FP for p
                }
            }
            // Contribution for ground-truth label (if non-background and different from pred)
            if g != 0 && g != p {
                let e = acc.entry(g).or_insert((0, 0, 0));
                e.2 += 1; // FN for g
            }
            acc
        },
        |mut a, b| {
            for (k, (btp, bfp, bfn)) in b {
                let e = a.entry(k).or_insert((0, 0, 0));
                e.0 += btp;
                e.1 += bfp;
                e.2 += bfn;
            }
            a
        },
    );

    let mut result: Vec<LabelOverlapMeasures> = combined
        .into_iter()
        .map(|(label, (tp, fp, fn_))| {
            let tn = n.saturating_sub(tp + fp + fn_);
            let v_p = tp + fp; // predicted volume
            let v_g = tp + fn_; // ground-truth volume

            // Dice
            let dice = {
                let denom = v_p + v_g;
                if denom == 0 {
                    1.0_f32
                } else {
                    2.0 * tp as f32 / denom as f32
                }
            };

            // Jaccard
            let jaccard = {
                let union = tp + fp + fn_;
                if union == 0 {
                    1.0_f32
                } else {
                    tp as f32 / union as f32
                }
            };

            // Volume similarity
            let volume_similarity = {
                let denom = v_p + v_g;
                if denom == 0 {
                    1.0_f32
                } else {
                    let diff = (v_p as isize - v_g as isize).unsigned_abs();
                    1.0 - diff as f32 / denom as f32
                }
            };

            // False-negative rate
            let false_negative_rate = if v_g == 0 {
                0.0_f32
            } else {
                fn_ as f32 / v_g as f32
            };

            // False-positive rate: FP / (FP + TN)
            let false_positive_rate = {
                let denom = fp + tn;
                if denom == 0 {
                    0.0_f32
                } else {
                    fp as f32 / denom as f32
                }
            };

            // Sensitivity = TP / V_G
            let sensitivity = if v_g == 0 {
                1.0_f32
            } else {
                tp as f32 / v_g as f32
            };

            // Specificity = TN / (TN + FP)
            let specificity = {
                let denom = tn + fp;
                if denom == 0 {
                    1.0_f32
                } else {
                    tn as f32 / denom as f32
                }
            };

            LabelOverlapMeasures {
                label,
                dice,
                jaccard,
                volume_similarity,
                false_negative_rate,
                false_positive_rate,
                sensitivity,
                specificity,
                predicted_volume: v_p,
                ground_truth_volume: v_g,
            }
        })
        .collect();

    result.sort_by_key(|m| m.label);
    result
}
