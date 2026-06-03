use super::*;

// ── Positive: Otsu creates binary output file ─────────────────────────────

/// Otsu segmentation must produce a file with the correct shape.
#[test]
fn test_segment_otsu_creates_output_file_with_correct_shape() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("mask.nii");

    ritk_io::write_nifti(&input, &make_bimodal_image()).unwrap();

    run(default_args(input.clone(), output.clone(), "otsu")).unwrap();

    assert!(output.exists(), "output mask must be created");
    let mask = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
    assert_eq!(mask.shape(), [4, 4, 4], "output shape must match input");
}

// ── Positive: Otsu output is strictly binary ──────────────────────────────

/// Every voxel in the Otsu output mask must be exactly 0.0 or 1.0.
#[test]
fn test_segment_otsu_output_is_strictly_binary() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.mha");
    let output = dir.path().join("mask.mha");

    ritk_io::write_metaimage(&input, &make_bimodal_image()).unwrap();

    run(default_args(input.clone(), output.clone(), "otsu")).unwrap();

    let mask = ritk_io::read_metaimage::<Backend, _>(&output, &Default::default()).unwrap();
    mask.with_data_slice(|values| {
        for &v in values {
            assert!(
                v == 0.0 || v == 1.0,
                "Otsu output must be strictly binary (0.0 or 1.0), got {v}"
            );
        }
    });
}

// ── Positive: Otsu threshold is between the two modes ─────────────────────

/// For a bimodal image with modes at 20 and 200, the Otsu threshold must
/// lie strictly between 20 and 200.
#[test]
fn test_segment_otsu_threshold_between_modes() {
    let image = make_bimodal_image();
    let threshold = otsu_threshold::<Backend, 3>(&image);
    assert!(
        threshold > 20.0 && threshold < 200.0,
        "Otsu threshold {threshold} must lie between the two modes (20, 200)"
    );
}

// ── Positive: Otsu foreground count matches high-intensity voxels ─────────

/// In the bimodal image the high-intensity half (32 voxels at 200.0)
/// must become the foreground class.
#[test]
fn test_segment_otsu_foreground_count_equals_high_mode_voxels() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("mask.nii");

    ritk_io::write_nifti(&input, &make_bimodal_image()).unwrap();

    run(default_args(input.clone(), output.clone(), "otsu")).unwrap();

    let mask = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
    let foreground = count_foreground(&mask);
    assert_eq!(
        foreground, 32,
        "Otsu must label exactly 32 high-intensity voxels as foreground"
    );
}

// ── Positive: Multi-Otsu creates labeled output ────────────────────────────

/// Multi-Otsu with 3 classes must create an output file with the correct shape.
#[test]
fn test_segment_multi_otsu_creates_output_with_correct_shape() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("labels.nii");

    ritk_io::write_nifti(&input, &make_trimodal_image()).unwrap();

    run(default_args(input.clone(), output.clone(), "multi-otsu")).unwrap();

    assert!(output.exists(), "output label image must be created");
    let labels = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
    assert_eq!(labels.shape(), [6, 6, 6], "label shape must match input");
}

// ── Positive: Multi-Otsu labels are in valid set ───────────────────────────

/// For K=3 classes, every voxel label must be in {0.0, 1.0, 2.0}.
#[test]
fn test_segment_multi_otsu_labels_in_valid_set() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.mha");
    let output = dir.path().join("labels.mha");

    ritk_io::write_metaimage(&input, &make_trimodal_image()).unwrap();

    run(default_args(input.clone(), output.clone(), "multi-otsu")).unwrap();

    let labels = ritk_io::read_metaimage::<Backend, _>(&output, &Default::default()).unwrap();
    labels.with_data_slice(|values| {
        let valid = [0.0_f32, 1.0_f32, 2.0_f32];
        for &v in values {
            assert!(
                valid.iter().any(|&vv| (v - vv).abs() < 1e-4),
                "label value {v} is not in the valid set {{0, 1, 2}} for K=3"
            );
        }
    });
}

// ── Positive: Multi-Otsu returns K-1 thresholds ───────────────────────────

/// For K=3 classes, `multi_otsu_threshold` must return exactly 2 thresholds,
/// both lying within the image's intensity range.
#[test]
fn test_segment_multi_otsu_returns_k_minus_1_thresholds() {
    let image = make_trimodal_image();
    let thresholds = multi_otsu_threshold::<Backend, 3>(&image, 3);
    assert_eq!(
        thresholds.len(),
        2,
        "K=3 must produce exactly 2 thresholds, got {:?}",
        thresholds
    );
    for &t in &thresholds {
        assert!(
            (30.0..=230.0).contains(&t),
            "threshold {t} must lie within the image intensity range [30, 230]"
        );
    }
    assert!(
        thresholds[0] < thresholds[1],
        "thresholds must be strictly increasing: {:?}",
        thresholds
    );
}

// ── Binary threshold tests ────────────────────────────────────────────────

#[test]
fn test_segment_binary_threshold_creates_output_with_correct_shape() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("out.nii");
    ritk_io::write_nifti(&input, &make_bimodal_image()).unwrap();

    let mut args = default_args(input.clone(), output.clone(), "binary");
    args.lower = Some(100.0);
    args.upper = Some(255.0);
    run(args).unwrap();

    assert!(output.exists(), "output file must be created");
    let out_image = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
    assert_eq!(
        out_image.shape(),
        [4, 4, 4],
        "output shape must match input 4×4×4"
    );
}

#[test]
fn test_segment_binary_threshold_output_is_strictly_binary() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("out.nii");
    ritk_io::write_nifti(&input, &make_bimodal_image()).unwrap();

    let mut args = default_args(input.clone(), output.clone(), "binary");
    args.lower = Some(100.0);
    args.upper = Some(255.0);
    run(args).unwrap();

    let out_image = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
    out_image.with_data_slice(|slice| {
        for &v in slice {
            assert!(
                v == 0.0 || v == 1.0,
                "binary threshold output must be in {{0,1}}, got {v}"
            );
        }
    });
}

#[test]
fn test_segment_binary_threshold_no_bounds_all_inside() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("out.nii");
    ritk_io::write_nifti(&input, &make_bimodal_image()).unwrap();

    let args = default_args(input.clone(), output.clone(), "binary");
    run(args).unwrap();

    let out_image = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
    out_image.with_data_slice(|slice| {
        assert!(
            slice.iter().all(|&v| v == 1.0),
            "no bounds → all voxels inside → all 1.0"
        );
    });
}
