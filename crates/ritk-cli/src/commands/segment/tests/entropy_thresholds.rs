use super::*; // ── Positive: Li threshold creates binary output ────────────────────────── /// Li thresholding on a bimodal image must produce a binary mask with the
/// threshold between the two modes.
#[test]
fn test_segment_li_creates_output_and_threshold_between_modes() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("mask.nii");
    ritk_io::write_nifti(&input, &make_bimodal_image()).unwrap();
    run(default_args(
        input.clone(),
        output.clone(),
        SegmentMethod::Li,
    ))
    .unwrap();
    assert!(output.exists(), "li output mask must be created");
    let mask = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
    assert_eq!(mask.shape(), [4, 4, 4], "output shape must match input");
    mask.with_data_slice(|values| {
        for &v in values {
            assert!(
                v == 0.0 || v == 1.0,
                "Li output must be strictly binary, got {v}"
            );
        }
    });
    let threshold = LiThreshold::new().compute(&make_bimodal_image());
    assert!(
        threshold > 20.0 && threshold < 200.0,
        "Li threshold {threshold} must lie between modes (20, 200)"
    );
}

// ── Positive: Yen threshold creates binary output ─────────────────────────
#[test]
fn test_segment_yen_creates_output_and_threshold_between_modes() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("mask.nii");
    ritk_io::write_nifti(&input, &make_bimodal_image()).unwrap();
    run(default_args(
        input.clone(),
        output.clone(),
        SegmentMethod::Yen,
    ))
    .unwrap();
    assert!(output.exists(), "yen output mask must be created");
    let mask = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
    assert_eq!(mask.shape(), [4, 4, 4]);
    mask.with_data_slice(|values| {
        for &v in values {
            assert!(
                v == 0.0 || v == 1.0,
                "Yen output must be strictly binary, got {v}"
            );
        }
    });
    let threshold = YenThreshold::new().compute(&make_bimodal_image());
    assert!(
        threshold > 20.0 && threshold < 200.0,
        "Yen threshold {threshold} must lie between modes (20, 200)"
    );
}

// ── Positive: Kapur threshold creates binary output ───────────────────────
#[test]
fn test_segment_kapur_creates_output_and_threshold_between_modes() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("mask.nii");
    ritk_io::write_nifti(&input, &make_bimodal_image()).unwrap();
    run(default_args(
        input.clone(),
        output.clone(),
        SegmentMethod::Kapur,
    ))
    .unwrap();
    assert!(output.exists(), "kapur output mask must be created");
    let mask = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
    assert_eq!(mask.shape(), [4, 4, 4]);
    mask.with_data_slice(|values| {
        for &v in values {
            assert!(
                v == 0.0 || v == 1.0,
                "Kapur output must be strictly binary, got {v}"
            );
        }
    });
    let threshold = KapurThreshold::new().compute(&make_bimodal_image());
    assert!(
        (20.0..=200.0).contains(&threshold),
        "Kapur threshold {threshold} must lie within mode range [20, 200]"
    );
}

// ── Positive: Triangle threshold creates binary output ────────────────────
#[test]
fn test_segment_triangle_creates_output_and_threshold_between_modes() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("mask.nii");
    ritk_io::write_nifti(&input, &make_bimodal_image()).unwrap();
    run(default_args(
        input.clone(),
        output.clone(),
        SegmentMethod::Triangle,
    ))
    .unwrap();
    assert!(output.exists(), "triangle output mask must be created");
    let mask = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
    assert_eq!(mask.shape(), [4, 4, 4]);
    mask.with_data_slice(|values| {
        for &v in values {
            assert!(
                v == 0.0 || v == 1.0,
                "Triangle output must be strictly binary, got {v}"
            );
        }
    });
    let threshold = TriangleThreshold::new().compute(&make_bimodal_image());
    assert!(
        threshold > 20.0 && threshold < 200.0,
        "Triangle threshold {threshold} must lie between modes (20, 200)"
    );
}

// ── Positive: foreground count tests ──────────────────────────────────────
#[test]
fn test_segment_li_foreground_count() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("mask.nii");
    ritk_io::write_nifti(&input, &make_bimodal_image()).unwrap();
    run(default_args(
        input.clone(),
        output.clone(),
        SegmentMethod::Li,
    ))
    .unwrap();
    let mask = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
    let foreground = count_foreground(&mask);
    assert_eq!(
        foreground, 32,
        "Li must label exactly 32 high-intensity voxels as foreground"
    );
}

#[test]
fn test_segment_yen_foreground_count() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("mask.nii");
    ritk_io::write_nifti(&input, &make_bimodal_image()).unwrap();
    run(default_args(
        input.clone(),
        output.clone(),
        SegmentMethod::Yen,
    ))
    .unwrap();
    let mask = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
    let foreground = count_foreground(&mask);
    assert_eq!(
        foreground, 32,
        "Yen must label exactly 32 high-intensity voxels as foreground"
    );
}

#[test]
fn test_segment_kapur_foreground_count() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("mask.nii");
    ritk_io::write_nifti(&input, &make_bimodal_image()).unwrap();
    run(default_args(
        input.clone(),
        output.clone(),
        SegmentMethod::Kapur,
    ))
    .unwrap();
    let mask = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
    let foreground = count_foreground(&mask);
    assert!(
        foreground == 32 || foreground == 64,
        "Kapur must label either 32 or 64 voxels as foreground, got {foreground}"
    );
}

#[test]
fn test_segment_triangle_foreground_count() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("mask.nii");
    ritk_io::write_nifti(&input, &make_bimodal_image()).unwrap();
    run(default_args(
        input.clone(),
        output.clone(),
        SegmentMethod::Triangle,
    ))
    .unwrap();
    let mask = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
    let foreground = count_foreground(&mask);
    assert_eq!(
        foreground, 32,
        "Triangle must label exactly 32 high-intensity voxels as foreground"
    );
}

#[test]
fn automatic_threshold_cli_family_matches_legacy_public_boundaries_exactly() {
    use ritk_segmentation::{
        KapurThreshold, LiThreshold, OtsuThreshold, TriangleThreshold, YenThreshold,
    };

    let dir = tempdir().unwrap();
    for method in [
        SegmentMethod::Otsu,
        SegmentMethod::Li,
        SegmentMethod::Yen,
        SegmentMethod::Kapur,
        SegmentMethod::Triangle,
    ] {
        let label = format!("{method:?}");
        let input = dir.path().join(format!("{label}-input.nii"));
        let output = dir.path().join(format!("{label}-mask.nii"));
        let fixture = make_bimodal_image();
        let expected = match method {
            SegmentMethod::Otsu => OtsuThreshold::new().apply(&fixture),
            SegmentMethod::Li => LiThreshold::new().apply(&fixture),
            SegmentMethod::Yen => YenThreshold::new().apply(&fixture),
            SegmentMethod::Kapur => KapurThreshold::new().apply(&fixture),
            SegmentMethod::Triangle => TriangleThreshold::new().apply(&fixture),
            _ => unreachable!("test enumerates only automatic threshold methods"),
        };
        ritk_io::write_nifti(&input, &fixture).unwrap();

        run(default_args(input, output.clone(), method)).unwrap();
        let actual = crate::commands::read_image_native(&output)
            .expect("automatic threshold output is natively readable");

        assert_eq!(actual.shape(), expected.shape());
        assert_eq!(*actual.origin(), *expected.origin());
        assert_eq!(*actual.spacing(), *expected.spacing());
        assert_eq!(*actual.direction(), *expected.direction());
        assert_eq!(
            actual.data_slice().expect("contiguous native output"),
            expected.data_slice().as_ref(),
            "{label} native CLI output diverged from its public legacy boundary"
        );
    }
}
