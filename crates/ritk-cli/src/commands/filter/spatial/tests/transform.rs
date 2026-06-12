use super::*;

// ── Frangi vesselness ────────────────────────────────────────────────────────

#[test]
fn test_filter_frangi_creates_output() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("out.nii");
    ritk_io::write_nifti(&input, &make_test_image()).unwrap();

    let mut args = default_args(input, output.clone(), "frangi");
    args.scales = vec![1.0, 2.0];

    let result = run_frangi(&args);
    assert!(result.is_ok(), "frangi must succeed: {:?}", result.err());
    assert!(output.exists(), "frangi must write output file");
}

// ── Gradient magnitude ───────────────────────────────────────────────────────

#[test]
fn test_filter_gradient_magnitude_creates_output() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("out.nii");
    ritk_io::write_nifti(&input, &make_test_image()).unwrap();

    let result = run_gradient_magnitude(&default_args(input, output.clone(), "gradient-magnitude"));
    assert!(
        result.is_ok(),
        "gradient-magnitude must succeed: {:?}",
        result.err()
    );
    assert!(output.exists(), "gradient-magnitude must write output file");
}

// ── Laplacian ────────────────────────────────────────────────────────────────

#[test]
fn test_filter_laplacian_creates_output() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("out.nii");
    ritk_io::write_nifti(&input, &make_test_image()).unwrap();

    let result = run_laplacian(&default_args(input, output.clone(), "laplacian"));
    assert!(result.is_ok(), "laplacian must succeed: {:?}", result.err());
    assert!(output.exists(), "laplacian must write output file");
}

// ── Recursive Gaussian ───────────────────────────────────────────────────────

#[test]
fn test_filter_recursive_gaussian_creates_output_with_correct_shape() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("out.nii");
    ritk_io::write_nifti(&input, &make_test_image()).unwrap();

    let result = run_recursive_gaussian(&default_args(input, output.clone(), "recursive-gaussian"));
    assert!(
        result.is_ok(),
        "recursive-gaussian must succeed: {:?}",
        result.err()
    );
    assert!(output.exists(), "recursive-gaussian must write output file");

    let filtered = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
    assert_eq!(
        filtered.shape(),
        [5, 5, 5],
        "recursive-gaussian output shape must match input"
    );
}

#[test]
fn test_filter_recursive_gaussian_order_1_creates_output() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("out.nii");
    ritk_io::write_nifti(&input, &make_test_image()).unwrap();

    let mut args = default_args(input, output.clone(), "recursive-gaussian");
    args.order = CliDerivativeOrder::First;

    let result = run_recursive_gaussian(&args);
    assert!(
        result.is_ok(),
        "recursive-gaussian order=1 must succeed: {:?}",
        result.err()
    );
    assert!(
        output.exists(),
        "recursive-gaussian order=1 must write output file"
    );
}
