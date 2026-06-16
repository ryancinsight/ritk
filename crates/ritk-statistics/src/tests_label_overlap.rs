//! Tests for per-label overlap measures.

use super::label_overlap::label_overlap_measures_from_slices;

// ── Helper ────────────────────────────────────────────────────────────────────

fn find(
    measures: &[super::label_overlap::LabelOverlapMeasures],
    label: u32,
) -> &super::label_overlap::LabelOverlapMeasures {
    measures
        .iter()
        .find(|m| m.label == label)
        .unwrap_or_else(|| panic!("label {} not found in measures", label))
}

// ── Positive tests ────────────────────────────────────────────────────────────

/// Perfect overlap: prediction == ground-truth → all metrics optimal.
#[test]
fn test_perfect_overlap_single_label() {
    let pred = vec![1.0_f32, 1.0, 1.0, 0.0, 0.0];
    let gt = vec![1.0_f32, 1.0, 1.0, 0.0, 0.0];
    let m = label_overlap_measures_from_slices(&pred, &gt);
    assert_eq!(m.len(), 1, "one label");
    let r = find(&m, 1);
    assert!((r.dice - 1.0).abs() < 1e-6, "dice={}", r.dice);
    assert!((r.jaccard - 1.0).abs() < 1e-6, "jaccard={}", r.jaccard);
    // ITK signed convention: equal volumes → 2·(3−3)/(3+3) = 0.
    assert!(
        r.volume_similarity.abs() < 1e-6,
        "vol_sim={}",
        r.volume_similarity
    );
    assert!(
        (r.false_negative_rate).abs() < 1e-6,
        "fnr={}",
        r.false_negative_rate
    );
    assert!(
        (r.sensitivity - 1.0).abs() < 1e-6,
        "sensitivity={}",
        r.sensitivity
    );
}

/// Complete mismatch: disjoint single-label → dice=0, jaccard=0.
#[test]
fn test_disjoint_single_label() {
    // pred has label 1 in first 3; gt has label 1 in last 3.
    let pred = vec![1.0_f32, 1.0, 1.0, 0.0, 0.0, 0.0];
    let gt = vec![0.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0];
    let m = label_overlap_measures_from_slices(&pred, &gt);
    let r = find(&m, 1);
    assert!(r.dice.abs() < 1e-6, "dice=0 for disjoint, got {}", r.dice);
    assert!(
        r.jaccard.abs() < 1e-6,
        "jaccard=0 for disjoint, got {}",
        r.jaccard
    );
    assert!((r.false_negative_rate - 1.0).abs() < 1e-6, "fnr=1");
    assert!((r.sensitivity).abs() < 1e-6, "sensitivity=0");
}

/// Both masks empty for a label never present → label absent from results.
#[test]
fn test_background_only_returns_empty() {
    let pred = vec![0.0_f32; 8];
    let gt = vec![0.0_f32; 8];
    let m = label_overlap_measures_from_slices(&pred, &gt);
    assert!(m.is_empty(), "no non-background labels → empty result");
}

/// Dice formula: 2*TP / (|pred| + |ref|).
/// pred=[1,1,1,1,0,0,0,0], gt=[0,0,1,1,1,1,0,0].
/// TP=2, FP=2, FN=2. Dice = 2*2/(4+4) = 0.5.
#[test]
fn test_dice_known_value() {
    let pred = vec![1.0_f32, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    let gt = vec![0.0_f32, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0];
    let m = label_overlap_measures_from_slices(&pred, &gt);
    let r = find(&m, 1);
    assert_eq!(r.predicted_volume, 4, "pred vol");
    assert_eq!(r.ground_truth_volume, 4, "gt vol");
    assert!((r.dice - 0.5).abs() < 1e-6, "dice={}", r.dice);
    // Jaccard = TP/(TP+FP+FN) = 2/(2+2+2) = 1/3
    assert!(
        (r.jaccard - 1.0 / 3.0).abs() < 1e-5,
        "jaccard={}",
        r.jaccard
    );
}

/// Volume similarity (ITK signed): pred_vol=3, gt_vol=1 → 2·(3−1)/(3+1) = 1.0.
#[test]
fn test_volume_similarity_known_value() {
    let pred = vec![1.0_f32, 1.0, 1.0, 0.0];
    let gt = vec![1.0_f32, 0.0, 0.0, 0.0];
    let m = label_overlap_measures_from_slices(&pred, &gt);
    let r = find(&m, 1);
    // TP=1, FP=2, FN=0, pred_vol=3, gt_vol=1
    // vol_sim = 2·(V_P − V_G)/(V_P + V_G) = 2·(3−1)/(3+1) = 1.0
    assert!(
        (r.volume_similarity - 1.0).abs() < 1e-6,
        "vol_sim={}",
        r.volume_similarity
    );
}

/// Sensitivity = TP / gt_vol.  pred=[1,1,0,0], gt=[1,1,1,1]. TP=2, FN=2. sens=0.5.
#[test]
fn test_sensitivity_known_value() {
    let pred = vec![1.0_f32, 1.0, 0.0, 0.0];
    let gt = vec![1.0_f32, 1.0, 1.0, 1.0];
    let m = label_overlap_measures_from_slices(&pred, &gt);
    let r = find(&m, 1);
    assert!(
        (r.sensitivity - 0.5).abs() < 1e-6,
        "sensitivity={}",
        r.sensitivity
    );
    assert!(
        (r.false_negative_rate - 0.5).abs() < 1e-6,
        "fnr={}",
        r.false_negative_rate
    );
}

/// Multi-label scenario: independent labels produce correct per-label results.
#[test]
fn test_two_labels_independent() {
    // 6 voxels: label1 in [0..3), label2 in [3..6)
    // pred: [1,1,1, 2,2,2]   gt: [1,1,0, 2,0,2]
    // label1: TP=2, FP=1, FN=1 → dice = 2*2/(3+2) = 4/5 = 0.8
    // label2: TP=2, FP=1, FN=1 → dice = 4/5 = 0.8
    let pred = vec![1.0_f32, 1.0, 1.0, 2.0, 2.0, 2.0];
    let gt = vec![1.0_f32, 1.0, 0.0, 2.0, 0.0, 2.0];
    let m = label_overlap_measures_from_slices(&pred, &gt);
    assert_eq!(m.len(), 2, "two labels");
    let r1 = find(&m, 1);
    let r2 = find(&m, 2);
    assert!((r1.dice - 0.8).abs() < 1e-5, "label1 dice={}", r1.dice);
    assert!((r2.dice - 0.8).abs() < 1e-5, "label2 dice={}", r2.dice);
}

/// Results are sorted ascending by label.
#[test]
fn test_results_sorted_by_label() {
    let pred = vec![3.0_f32, 1.0, 2.0, 3.0, 1.0];
    let gt = vec![3.0_f32, 1.0, 2.0, 3.0, 1.0];
    let m = label_overlap_measures_from_slices(&pred, &gt);
    assert_eq!(m.len(), 3);
    assert_eq!(m[0].label, 1);
    assert_eq!(m[1].label, 2);
    assert_eq!(m[2].label, 3);
}

/// When prediction is empty but gt has non-zero: dice=0, sensitivity=0, fnr=1.
#[test]
fn test_pred_empty_gt_nonempty() {
    let pred = vec![0.0_f32; 4];
    let gt = vec![1.0_f32, 1.0, 0.0, 0.0];
    let m = label_overlap_measures_from_slices(&pred, &gt);
    let r = find(&m, 1);
    assert!(r.dice.abs() < 1e-6, "dice=0");
    assert!((r.sensitivity).abs() < 1e-6, "sensitivity=0");
    assert!((r.false_negative_rate - 1.0).abs() < 1e-6, "fnr=1");
    assert_eq!(r.predicted_volume, 0);
    assert_eq!(r.ground_truth_volume, 2);
}

/// When gt is empty but prediction has non-zero: fnr=0, fpr positive.
#[test]
fn test_gt_empty_pred_nonempty() {
    let pred = vec![1.0_f32, 1.0, 0.0, 0.0];
    let gt = vec![0.0_f32; 4];
    let m = label_overlap_measures_from_slices(&pred, &gt);
    let r = find(&m, 1);
    assert!(r.dice.abs() < 1e-6, "dice=0 when gt empty");
    assert!((r.false_negative_rate).abs() < 1e-6, "fnr=0 (no gt)");
    assert_eq!(r.ground_truth_volume, 0);
    assert_eq!(r.predicted_volume, 2);
}

/// Specificity = TN / (TN + FP): N=6, TP=2, FP=2, FN=2, TN=0. spec=0.
/// N=6, label1 TN=0 because all 6 voxels are involved:
/// pred=[1,1,1,1,0,0], gt=[0,0,1,1,1,1]. TP=2,FP=2,FN=2,TN=0.
#[test]
fn test_specificity_zero_when_all_foreground() {
    let pred = vec![1.0_f32, 1.0, 1.0, 1.0, 0.0, 0.0];
    let gt = vec![0.0_f32, 0.0, 1.0, 1.0, 1.0, 1.0];
    let m = label_overlap_measures_from_slices(&pred, &gt);
    let r = find(&m, 1);
    // TN = 6 - 2 - 2 - 2 = 0; FP=2; spec = 0/(0+2) = 0
    assert!(r.specificity.abs() < 1e-6, "spec={}", r.specificity);
}

/// Adversarial: large uniform prediction. 100 voxels all label=1.
#[test]
fn test_large_uniform_perfect_overlap() {
    let n = 100;
    let pred = vec![1.0_f32; n];
    let gt = vec![1.0_f32; n];
    let m = label_overlap_measures_from_slices(&pred, &gt);
    assert_eq!(m.len(), 1);
    let r = find(&m, 1);
    assert!((r.dice - 1.0).abs() < 1e-6);
    assert!((r.jaccard - 1.0).abs() < 1e-6);
    assert!((r.sensitivity - 1.0).abs() < 1e-6);
    assert_eq!(r.predicted_volume, n);
    assert_eq!(r.ground_truth_volume, n);
}

/// Length mismatch panics.
#[test]
#[should_panic(expected = "equal length")]
fn test_length_mismatch_panics() {
    let _ = label_overlap_measures_from_slices(&[1.0_f32; 4], &[1.0_f32; 5]);
}
