use super::*;
use tempfile::tempdir;

// ── Negative: unknown filter name returns error ───────────────────────────
#[test]
fn test_filter_unknown_name_returns_error() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("out.nii");

    ritk_io::write_nifti(&input, &make_test_image()).unwrap();

    let result = run(default_args(input, output, "nonexistent"));
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("Unknown filter 'nonexistent'"),
        "error must name the unknown filter, got: {msg}"
    );
}

// ── Boundary: missing input file returns error ────────────────────────────
#[test]
fn test_filter_missing_input_returns_error() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("does_not_exist.nii");
    let output = dir.path().join("out.nii");

    let result = run(default_args(input, output, "gaussian"));
    assert!(result.is_err(), "missing input must yield an error");
}
