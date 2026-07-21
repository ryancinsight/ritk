//! Majority voting tests.

use super::super::*;

// ── Majority voting – positive tests ──────────────────────────────────

/// All 3 atlases assign label 7 at every voxel.
///
/// Expected: fused label = 7 everywhere, confidence = 3/3 = 1.0.
#[test]
fn majority_vote_unanimous() {
    let dims = [2, 2, 2];
    let n = 8;
    let l1 = vec![7u32; n];
    let l2 = vec![7u32; n];
    let l3 = vec![7u32; n];
    let atlas_labels: Vec<&[u32]> = vec![&l1, &l2, &l3];

    let result = majority_vote(&atlas_labels, dims).expect("infallible: validated precondition");
    assert_eq!(result.labels.len(), n);
    assert_eq!(result.confidence.len(), n);
    for i in 0..n {
        assert_eq!(result.labels[i], 7, "voxel {}", i);
        assert!((result.confidence[i] - 1.0).abs() < 1e-6, "voxel {}", i);
    }
}

/// 3 atlases: two assign label 1, one assigns label 2.
///
/// Expected: fused label = 1 everywhere, confidence = 2/3 ≈ 0.6667.
#[test]
fn majority_vote_majority_label() {
    let dims = [2, 2, 2];
    let n = 8;
    let l1 = vec![1u32; n];
    let l2 = vec![1u32; n];
    let l3 = vec![2u32; n];
    let atlas_labels: Vec<&[u32]> = vec![&l1, &l2, &l3];

    let result = majority_vote(&atlas_labels, dims).expect("infallible: validated precondition");
    let expected_conf = 2.0f32 / 3.0;
    for i in 0..n {
        assert_eq!(result.labels[i], 1, "voxel {}", i);
        assert!(
            (result.confidence[i] - expected_conf).abs() < 1e-6,
            "voxel {} confidence {} != {}",
            i,
            result.confidence[i],
            expected_conf
        );
    }
}

/// Tie-breaking: 2 atlases, one votes label 3, one votes label 1.
/// Tie → smallest label (1) wins. Confidence = 0.5.
#[test]
fn majority_vote_tie_smallest_label_wins() {
    let dims = [1, 1, 2];
    let l1 = vec![3u32; 2];
    let l2 = vec![1u32; 2];
    let atlas_labels: Vec<&[u32]> = vec![&l1, &l2];

    let result = majority_vote(&atlas_labels, dims).expect("infallible: validated precondition");
    for i in 0..2 {
        assert_eq!(
            result.labels[i], 1,
            "smallest label wins tie at voxel {}",
            i
        );
        assert!((result.confidence[i] - 0.5).abs() < 1e-6);
    }
}

// ── Majority voting – boundary tests ──────────────────────────────────

/// Single atlas: fused labels equal the atlas labels, confidence = 1.0.
#[test]
fn majority_vote_single_atlas() {
    let dims = [2, 2, 2];
    let n = 8;
    let mut labels = vec![0u32; n];
    for (i, v) in labels.iter_mut().enumerate() {
        *v = i as u32 + 10;
    }
    let atlas_labels: Vec<&[u32]> = vec![&labels];

    let result = majority_vote(&atlas_labels, dims).expect("infallible: validated precondition");
    for i in 0..n {
        assert_eq!(result.labels[i], i as u32 + 10, "voxel {}", i);
        assert!((result.confidence[i] - 1.0).abs() < 1e-6);
    }
}

// ── Majority voting – negative tests ──────────────────────────────────

/// Empty atlas list returns `InvalidConfiguration`.
#[test]
fn majority_vote_empty_error() {
    let atlas_labels: Vec<&[u32]> = vec![];
    let err = majority_vote(&atlas_labels, [2, 2, 2]).unwrap_err();
    assert!(
        matches!(err, RegistrationError::InvalidConfiguration(_)),
        "expected InvalidConfiguration, got {:?}",
        err
    );
}

/// Atlas label map with wrong length returns `DimensionMismatch`.
#[test]
fn majority_vote_dimension_mismatch() {
    let l1 = vec![1u32; 8]; // correct for [2,2,2]
    let l2 = vec![1u32; 5]; // wrong
    let atlas_labels: Vec<&[u32]> = vec![&l1, &l2];
    let err = majority_vote(&atlas_labels, [2, 2, 2]).unwrap_err();
    assert!(
        matches!(err, RegistrationError::DimensionMismatch(_)),
        "expected DimensionMismatch, got {:?}",
        err
    );
}
