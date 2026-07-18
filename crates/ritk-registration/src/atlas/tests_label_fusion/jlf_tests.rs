//! Joint label fusion (JLF) tests.

use super::super::*;

// â”€â”€ JLF â€“ positive tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// 3 equidistant atlases (all intensity 1.0, target 2.0), all label 5.
///
/// Analytical derivation (patch_radius = 0, single-voxel patches):
/// dáµ¢ = (1.0 âˆ’ 2.0)Â² = 1.0 for all i.
/// M = [[2, 2, 2], [2, 2, 2], [2, 2, 2]].
/// min(M) = 2, Î± = 0.1 Ã— 2 = 0.2.
/// M_reg = [[2.2, 2, 2], [2, 2.2, 2], [2, 2, 2.2]].
/// By symmetry wâ‚ = wâ‚‚ = wâ‚ƒ = w. Row sum: 6.2w = 1 â†’ w = 1/6.2.
/// Normalized: wáµ¢ = 1/3.
/// All labels = 5 â†’ fused = 5, confidence = 1.0.
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
        beta: 0.1 };
    let result = joint_label_fusion(&target, &atlas_images, &atlas_labels, dims, &config).unwrap();

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
/// Same M derivation as above: equal weights wáµ¢ = 1/3.
/// Vote for label 1: wâ‚ + wâ‚‚ = 2/3.
/// Vote for label 2: wâ‚ƒ = 1/3.
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
        beta: 0.1 };
    let result = joint_label_fusion(&target, &atlas_images, &atlas_labels, dims, &config).unwrap();

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

/// 2 equidistant atlases with different labels: equal weights â†’ tie â†’
/// smallest label wins.
///
/// A1 = 1.0, A2 = 3.0, T = 2.0. dâ‚ = dâ‚‚ = 1.0.
/// M = [[2,2],[2,2]], min = 2, Î± = 0.2.
/// M_reg = [[2.2,2],[2,2.2]], det = 0.84.
/// Mâ»Â¹Â·1 = (1/0.84)[0.2, 0.2] â†’ normalized w = [0.5, 0.5].
/// L1 = 1, L2 = 2, both weight 0.5 â†’ tie â†’ label 1 wins.
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
        beta: 0.1 };
    let result = joint_label_fusion(&target, &atlas_images, &atlas_labels, dims, &config).unwrap();

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

/// Atlases identical to target: all dáµ¢ = 0, min(M) = 0, Î± = 0.
/// M is singular (all zeros) â†’ uniform fallback â†’ equal weights.
/// All labels = 3 â†’ fused = 3, confidence = 1.0.
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
        beta: 0.1 };
    let result = joint_label_fusion(&target, &atlas_images, &atlas_labels, dims, &config).unwrap();

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

// â”€â”€ JLF â€“ boundary tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Single atlas: 1Ã—1 system M=[2d+Î±], w=[1/(2d+Î±)], normalized to 1.0.
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
        beta: 0.1 };
    let result = joint_label_fusion(&target, &atlas_images, &atlas_labels, dims, &config).unwrap();

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
        beta: 0.1 };
    let result = joint_label_fusion(&target, &atlas_images, &atlas_labels, dims, &config).unwrap();

    for i in 0..n {
        assert_eq!(result.labels[i], 1, "voxel {}", i);
        // Equal weights â†’ confidence = 1.0 (all same label).
        assert!(
            (result.confidence[i] - 1.0).abs() < 1e-4,
            "voxel {} confidence {}",
            i,
            result.confidence[i]
        );
    }
}

// â”€â”€ JLF â€“ negative tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Empty atlas list returns `InvalidConfiguration`.
#[test]
fn jlf_empty_error() {
    let target = vec![1.0f32; 8];
    let atlas_images: Vec<&[f32]> = vec![];
    let atlas_labels: Vec<&[u32]> = vec![];
    let config = LabelFusionConfig::default();
    let err =
        joint_label_fusion(&target, &atlas_images, &atlas_labels, [2, 2, 2], &config).unwrap_err();
    assert!(
        matches!(err, RegistrationError::InvalidConfiguration(_)),
        "expected InvalidConfiguration, got {:?}",
        err
    );
}

/// Mismatched atlas_images / atlas_labels count â†’ `DimensionMismatch`.
#[test]
fn jlf_atlas_count_mismatch() {
    let target = vec![1.0f32; 8];
    let a = vec![1.0f32; 8];
    let l1 = vec![1u32; 8];
    let l2 = vec![1u32; 8];
    let atlas_images: Vec<&[f32]> = vec![&a];
    let atlas_labels: Vec<&[u32]> = vec![&l1, &l2];
    let config = LabelFusionConfig::default();
    let err =
        joint_label_fusion(&target, &atlas_images, &atlas_labels, [2, 2, 2], &config).unwrap_err();
    assert!(
        matches!(err, RegistrationError::DimensionMismatch(_)),
        "expected DimensionMismatch, got {:?}",
        err
    );
}

/// Target with wrong length â†’ `DimensionMismatch`.
#[test]
fn jlf_target_dimension_mismatch() {
    let target = vec![1.0f32; 5]; // wrong for [2,2,2]
    let a = vec![1.0f32; 8];
    let l = vec![1u32; 8];
    let atlas_images: Vec<&[f32]> = vec![&a];
    let atlas_labels: Vec<&[u32]> = vec![&l];
    let config = LabelFusionConfig::default();
    let err =
        joint_label_fusion(&target, &atlas_images, &atlas_labels, [2, 2, 2], &config).unwrap_err();
    assert!(
        matches!(err, RegistrationError::DimensionMismatch(_)),
        "expected DimensionMismatch, got {:?}",
        err
    );
}

/// Atlas image with wrong length â†’ `DimensionMismatch`.
#[test]
fn jlf_atlas_image_dimension_mismatch() {
    let target = vec![1.0f32; 8];
    let a = vec![1.0f32; 5]; // wrong
    let l = vec![1u32; 8];
    let atlas_images: Vec<&[f32]> = vec![&a];
    let atlas_labels: Vec<&[u32]> = vec![&l];
    let config = LabelFusionConfig::default();
    let err =
        joint_label_fusion(&target, &atlas_images, &atlas_labels, [2, 2, 2], &config).unwrap_err();
    assert!(
        matches!(err, RegistrationError::DimensionMismatch(_)),
        "expected DimensionMismatch, got {:?}",
        err
    );
}

/// Atlas label map with wrong length â†’ `DimensionMismatch`.
#[test]
fn jlf_atlas_label_dimension_mismatch() {
    let target = vec![1.0f32; 8];
    let a = vec![1.0f32; 8];
    let l = vec![1u32; 5]; // wrong
    let atlas_images: Vec<&[f32]> = vec![&a];
    let atlas_labels: Vec<&[u32]> = vec![&l];
    let config = LabelFusionConfig::default();
    let err =
        joint_label_fusion(&target, &atlas_images, &atlas_labels, [2, 2, 2], &config).unwrap_err();
    assert!(
        matches!(err, RegistrationError::DimensionMismatch(_)),
        "expected DimensionMismatch, got {:?}",
        err
    );
}
