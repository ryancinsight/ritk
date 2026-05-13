//! Tests for label_fusion
//! Extracted to keep the 500-line structural limit.
use super::*;

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

    let result = majority_vote(&atlas_labels, dims).unwrap();
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

    let result = majority_vote(&atlas_labels, dims).unwrap();
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
/// Tie → smallest label (1) wins.  Confidence = 0.5.
#[test]
fn majority_vote_tie_smallest_label_wins() {
    let dims = [1, 1, 2];
    let l1 = vec![3u32; 2];
    let l2 = vec![1u32; 2];
    let atlas_labels: Vec<&[u32]> = vec![&l1, &l2];

    let result = majority_vote(&atlas_labels, dims).unwrap();
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

    let result = majority_vote(&atlas_labels, dims).unwrap();
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

// ── JLF – positive tests ─────────────────────────────────────────────

/// 3 equidistant atlases (all intensity 1.0, target 2.0), all label 5.
///
/// Analytical derivation (patch_radius = 0, single-voxel patches):
///   dᵢ = (1.0 − 2.0)² = 1.0  for all i.
///   M = [[2, 2, 2], [2, 2, 2], [2, 2, 2]].
///   min(M) = 2, α = 0.1 × 2 = 0.2.
///   M_reg = [[2.2, 2, 2], [2, 2.2, 2], [2, 2, 2.2]].
///   By symmetry w₁ = w₂ = w₃ = w. Row sum: 6.2w = 1 → w = 1/6.2.
///   Normalized: wᵢ = 1/3.
///   All labels = 5 → fused = 5, confidence = 1.0.
#[test]
fn jlf_equidistant_same_labels() {
    let dims = [2, 2, 2];
    let n = 8;
    let a1 = vec![1.0f32; n];
    let a2 = vec![1.0f32; n];
    let a3 = vec![1.0f32; n];
    let target = vec![2.0f32; n];
    let l1 = vec![5u32; n];
    let l2 = vec![5u32; n];
    let l3 = vec![5u32; n];
    let atlas_images: Vec<&[f32]> = vec![&a1, &a2, &a3];
    let atlas_labels: Vec<&[u32]> = vec![&l1, &l2, &l3];

    let config = LabelFusionConfig {
        patch_radius: 0,
        beta: 0.1,
    };
    let result =
        joint_label_fusion(&target, &atlas_images, &atlas_labels, dims, &config).unwrap();

    for i in 0..n {
        assert_eq!(result.labels[i], 5, "voxel {}", i);
        assert!(
            (result.confidence[i] - 1.0).abs() < 1e-4,
            "voxel {} confidence {} != 1.0",
            i,
            result.confidence[i]
        );
    }
}

/// 3 equidistant atlases, 2 with label 1, 1 with label 2.
///
/// Same M derivation as above: equal weights wᵢ = 1/3.
/// Vote for label 1: w₁ + w₂ = 2/3.
/// Vote for label 2: w₃ = 1/3.
/// Fused = 1, confidence = 2/3.
#[test]
fn jlf_equidistant_majority() {
    let dims = [2, 2, 2];
    let n = 8;
    let a1 = vec![1.0f32; n];
    let a2 = vec![1.0f32; n];
    let a3 = vec![1.0f32; n];
    let target = vec![2.0f32; n];
    let l1 = vec![1u32; n];
    let l2 = vec![1u32; n];
    let l3 = vec![2u32; n];
    let atlas_images: Vec<&[f32]> = vec![&a1, &a2, &a3];
    let atlas_labels: Vec<&[u32]> = vec![&l1, &l2, &l3];

    let config = LabelFusionConfig {
        patch_radius: 0,
        beta: 0.1,
    };
    let result =
        joint_label_fusion(&target, &atlas_images, &atlas_labels, dims, &config).unwrap();

    let expected_conf = 2.0f64 / 3.0;
    for i in 0..n {
        assert_eq!(result.labels[i], 1, "voxel {}", i);
        assert!(
            (result.confidence[i] as f64 - expected_conf).abs() < 1e-4,
            "voxel {} confidence {} != {}",
            i,
            result.confidence[i],
            expected_conf
        );
    }
}

/// 2 equidistant atlases with different labels: equal weights → tie →
/// smallest label wins.
///
/// A1 = 1.0, A2 = 3.0, T = 2.0.  d₁ = d₂ = 1.0.
/// M = [[2,2],[2,2]], min = 2, α = 0.2.
/// M_reg = [[2.2,2],[2,2.2]], det = 0.84.
/// M⁻¹·1 = (1/0.84)[0.2, 0.2] → normalized w = [0.5, 0.5].
/// L1 = 1, L2 = 2, both weight 0.5 → tie → label 1 wins.
/// Confidence = 0.5.
#[test]
fn jlf_equidistant_tie_smallest_label() {
    let dims = [2, 2, 2];
    let n = 8;
    let a1 = vec![1.0f32; n];
    let a2 = vec![3.0f32; n];
    let target = vec![2.0f32; n];
    let l1 = vec![1u32; n];
    let l2 = vec![2u32; n];
    let atlas_images: Vec<&[f32]> = vec![&a1, &a2];
    let atlas_labels: Vec<&[u32]> = vec![&l1, &l2];

    let config = LabelFusionConfig {
        patch_radius: 0,
        beta: 0.1,
    };
    let result =
        joint_label_fusion(&target, &atlas_images, &atlas_labels, dims, &config).unwrap();

    for i in 0..n {
        assert_eq!(
            result.labels[i], 1,
            "smallest label wins tie at voxel {}",
            i
        );
        assert!(
            (result.confidence[i] - 0.5).abs() < 1e-4,
            "voxel {} confidence {} != 0.5",
            i,
            result.confidence[i]
        );
    }
}

/// Atlases identical to target: all dᵢ = 0, min(M) = 0, α = 0.
/// M is singular (all zeros) → uniform fallback → equal weights.
/// All labels = 3 → fused = 3, confidence = 1.0.
#[test]
fn jlf_zero_distance_singular_fallback() {
    let dims = [2, 2, 2];
    let n = 8;
    let a = vec![5.0f32; n];
    let target = vec![5.0f32; n];
    let l = vec![3u32; n];
    let atlas_images: Vec<&[f32]> = vec![&a, &a, &a];
    let atlas_labels: Vec<&[u32]> = vec![&l, &l, &l];

    let config = LabelFusionConfig {
        patch_radius: 0,
        beta: 0.1,
    };
    let result =
        joint_label_fusion(&target, &atlas_images, &atlas_labels, dims, &config).unwrap();

    for i in 0..n {
        assert_eq!(result.labels[i], 3, "voxel {}", i);
        assert!(
            (result.confidence[i] - 1.0).abs() < 1e-4,
            "voxel {} confidence {} != 1.0",
            i,
            result.confidence[i]
        );
    }
}

// ── JLF – boundary tests ─────────────────────────────────────────────

/// Single atlas: 1×1 system M=[2d+α], w=[1/(2d+α)], normalized to 1.0.
/// Fused labels equal the atlas labels, confidence = 1.0.
#[test]
fn jlf_single_atlas() {
    let dims = [2, 2, 2];
    let n = 8;
    let a = vec![1.0f32; n];
    let target = vec![2.0f32; n];
    let mut l = vec![0u32; n];
    for (i, v) in l.iter_mut().enumerate() {
        *v = i as u32 + 100;
    }
    let atlas_images: Vec<&[f32]> = vec![&a];
    let atlas_labels: Vec<&[u32]> = vec![&l];

    let config = LabelFusionConfig {
        patch_radius: 0,
        beta: 0.1,
    };
    let result =
        joint_label_fusion(&target, &atlas_images, &atlas_labels, dims, &config).unwrap();

    for i in 0..n {
        assert_eq!(result.labels[i], i as u32 + 100, "voxel {}", i);
        assert!(
            (result.confidence[i] - 1.0).abs() < 1e-4,
            "voxel {} confidence {} != 1.0",
            i,
            result.confidence[i]
        );
    }
}

/// Patch radius > 0 on a uniform image gives the same result as radius 0
/// (all voxels in the patch have the same value, so d scales but the
/// relative weights remain equal for equidistant atlases).
#[test]
fn jlf_nonzero_patch_radius_equidistant() {
    let dims = [4, 4, 4];
    let n = 64;
    let a1 = vec![1.0f32; n];
    let a2 = vec![1.0f32; n];
    let target = vec![2.0f32; n];
    let l1 = vec![1u32; n];
    let l2 = vec![1u32; n];
    let atlas_images: Vec<&[f32]> = vec![&a1, &a2];
    let atlas_labels: Vec<&[u32]> = vec![&l1, &l2];

    let config = LabelFusionConfig {
        patch_radius: 1,
        beta: 0.1,
    };
    let result =
        joint_label_fusion(&target, &atlas_images, &atlas_labels, dims, &config).unwrap();

    for i in 0..n {
        assert_eq!(result.labels[i], 1, "voxel {}", i);
        // Equal weights → confidence = 1.0 (all same label).
        assert!(
            (result.confidence[i] - 1.0).abs() < 1e-4,
            "voxel {} confidence {}",
            i,
            result.confidence[i]
        );
    }
}

// ── JLF – negative tests ─────────────────────────────────────────────

/// Empty atlas list returns `InvalidConfiguration`.
#[test]
fn jlf_empty_error() {
    let target = vec![1.0f32; 8];
    let atlas_images: Vec<&[f32]> = vec![];
    let atlas_labels: Vec<&[u32]> = vec![];
    let config = LabelFusionConfig::default();
    let err = joint_label_fusion(&target, &atlas_images, &atlas_labels, [2, 2, 2], &config)
        .unwrap_err();
    assert!(
        matches!(err, RegistrationError::InvalidConfiguration(_)),
        "expected InvalidConfiguration, got {:?}",
        err
    );
}

/// Mismatched atlas_images / atlas_labels count → `DimensionMismatch`.
#[test]
fn jlf_atlas_count_mismatch() {
    let target = vec![1.0f32; 8];
    let a = vec![1.0f32; 8];
    let l1 = vec![1u32; 8];
    let l2 = vec![1u32; 8];
    let atlas_images: Vec<&[f32]> = vec![&a];
    let atlas_labels: Vec<&[u32]> = vec![&l1, &l2];
    let config = LabelFusionConfig::default();
    let err = joint_label_fusion(&target, &atlas_images, &atlas_labels, [2, 2, 2], &config)
        .unwrap_err();
    assert!(
        matches!(err, RegistrationError::DimensionMismatch(_)),
        "expected DimensionMismatch, got {:?}",
        err
    );
}

/// Target with wrong length → `DimensionMismatch`.
#[test]
fn jlf_target_dimension_mismatch() {
    let target = vec![1.0f32; 5]; // wrong for [2,2,2]
    let a = vec![1.0f32; 8];
    let l = vec![1u32; 8];
    let atlas_images: Vec<&[f32]> = vec![&a];
    let atlas_labels: Vec<&[u32]> = vec![&l];
    let config = LabelFusionConfig::default();
    let err = joint_label_fusion(&target, &atlas_images, &atlas_labels, [2, 2, 2], &config)
        .unwrap_err();
    assert!(
        matches!(err, RegistrationError::DimensionMismatch(_)),
        "expected DimensionMismatch, got {:?}",
        err
    );
}

/// Atlas image with wrong length → `DimensionMismatch`.
#[test]
fn jlf_atlas_image_dimension_mismatch() {
    let target = vec![1.0f32; 8];
    let a = vec![1.0f32; 5]; // wrong
    let l = vec![1u32; 8];
    let atlas_images: Vec<&[f32]> = vec![&a];
    let atlas_labels: Vec<&[u32]> = vec![&l];
    let config = LabelFusionConfig::default();
    let err = joint_label_fusion(&target, &atlas_images, &atlas_labels, [2, 2, 2], &config)
        .unwrap_err();
    assert!(
        matches!(err, RegistrationError::DimensionMismatch(_)),
        "expected DimensionMismatch, got {:?}",
        err
    );
}

/// Atlas label map with wrong length → `DimensionMismatch`.
#[test]
fn jlf_atlas_label_dimension_mismatch() {
    let target = vec![1.0f32; 8];
    let a = vec![1.0f32; 8];
    let l = vec![1u32; 5]; // wrong
    let atlas_images: Vec<&[f32]> = vec![&a];
    let atlas_labels: Vec<&[u32]> = vec![&l];
    let config = LabelFusionConfig::default();
    let err = joint_label_fusion(&target, &atlas_images, &atlas_labels, [2, 2, 2], &config)
        .unwrap_err();
    assert!(
        matches!(err, RegistrationError::DimensionMismatch(_)),
        "expected DimensionMismatch, got {:?}",
        err
    );
}

// ── solve_linear_system unit tests ───────────────────────────────────

/// 2×2 identity system: Ix = [3, 7] → x = [3, 7].
#[test]
fn solve_identity_2x2() {
    let mut a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let mut b = vec![3.0, 7.0];
    let x = solve_linear_system(&mut a, &mut b).unwrap();
    assert!((x[0] - 3.0).abs() < 1e-12, "x[0] = {}", x[0]);
    assert!((x[1] - 7.0).abs() < 1e-12, "x[1] = {}", x[1]);
}

/// 2×2 system: [[2, 1], [1, 3]] x = [5, 10] → x = [5/5, 15/5] = [1, 3].
///
/// Verification: 2·1 + 1·3 = 5 ✓, 1·1 + 3·3 = 10 ✓.
#[test]
fn solve_2x2_known() {
    let mut a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
    let mut b = vec![5.0, 10.0];
    let x = solve_linear_system(&mut a, &mut b).unwrap();
    assert!((x[0] - 1.0).abs() < 1e-12, "x[0] = {}", x[0]);
    assert!((x[1] - 3.0).abs() < 1e-12, "x[1] = {}", x[1]);
}

/// Singular 2×2 system: [[1, 1], [1, 1]] x = [1, 1] → None.
#[test]
fn solve_singular_returns_none() {
    let mut a = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
    let mut b = vec![1.0, 1.0];
    assert!(solve_linear_system(&mut a, &mut b).is_none());
}

/// 1×1 system: [4] x = [8] → x = 2.
#[test]
fn solve_1x1() {
    let mut a = vec![vec![4.0]];
    let mut b = vec![8.0];
    let x = solve_linear_system(&mut a, &mut b).unwrap();
    assert!((x[0] - 2.0).abs() < 1e-12, "x[0] = {}", x[0]);
}
