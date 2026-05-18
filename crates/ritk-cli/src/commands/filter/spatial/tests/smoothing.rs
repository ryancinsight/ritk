use super::*;

// ── Median ───────────────────────────────────────────────────────────────────

#[test]
fn test_filter_median_creates_output_with_correct_shape() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("out.nii");
    ritk_io::write_nifti(&input, &make_test_image()).unwrap();

    let result = run_median(&default_args(input, output.clone(), "median"));
    assert!(result.is_ok(), "median must succeed: {:?}", result.err());
    assert!(output.exists(), "median must write output file");

    let filtered = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
    assert_eq!(
        filtered.shape(),
        [5, 5, 5],
        "median output shape must match input"
    );
}

// ── Bilateral ────────────────────────────────────────────────────────────────

#[test]
fn test_filter_bilateral_creates_output_with_correct_shape() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("out.nii");
    ritk_io::write_nifti(&input, &make_test_image()).unwrap();

    let result = run_bilateral(&default_args(input, output.clone(), "bilateral"));
    assert!(result.is_ok(), "bilateral must succeed: {:?}", result.err());
    assert!(output.exists(), "bilateral must write output file");

    let filtered = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
    assert_eq!(
        filtered.shape(),
        [5, 5, 5],
        "bilateral output shape must match input"
    );
}

// ── Canny ────────────────────────────────────────────────────────────────────

#[test]
fn test_filter_canny_creates_output_with_binary_values() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("out.nii");
    ritk_io::write_nifti(&input, &make_test_image()).unwrap();

    let result = run_canny(&default_args(input, output.clone(), "canny"));
    assert!(result.is_ok(), "canny must succeed: {:?}", result.err());
    assert!(output.exists(), "canny must write output file");

    let filtered = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
    assert_eq!(
        filtered.shape(),
        [5, 5, 5],
        "canny output shape must match input"
    );

    // Canny output is binary: every voxel must be 0.0 or 1.0.
    filtered.with_data_slice(|values| {
        for &v in values {
            assert!(
                v == 0.0 || v == 1.0,
                "Canny output must be strictly binary (0.0 or 1.0), got {v}"
            );
        }
    });
}

// ── Sobel ────────────────────────────────────────────────────────────────────

#[test]
fn test_filter_sobel_creates_output_with_correct_shape() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("out.nii");
    ritk_io::write_nifti(&input, &make_test_image()).unwrap();

    let result = run_sobel(&default_args(input, output.clone(), "sobel"));
    assert!(result.is_ok(), "sobel must succeed: {:?}", result.err());
    assert!(output.exists(), "sobel must write output file");

    let filtered = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
    assert_eq!(
        filtered.shape(),
        [5, 5, 5],
        "sobel output shape must match input"
    );
}

// ── Laplacian of Gaussian (LoG) ──────────────────────────────────────────────

#[test]
fn test_filter_log_creates_output_with_correct_shape() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("out.nii");
    ritk_io::write_nifti(&input, &make_test_image()).unwrap();

    let result = run_log(&default_args(input, output.clone(), "log"));
    assert!(result.is_ok(), "log must succeed: {:?}", result.err());
    assert!(output.exists(), "log must write output file");

    let filtered = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
    assert_eq!(
        filtered.shape(),
        [5, 5, 5],
        "log output shape must match input"
    );
}
