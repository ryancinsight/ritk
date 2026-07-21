use super::*;
use tempfile::tempdir;

// ── Boundary: missing input file returns error ────────────────────────────
#[test]
fn test_filter_missing_input_returns_error() {
    let dir = tempdir().expect("infallible: validated precondition");
    let input = dir.path().join("does_not_exist.nii");
    let output = dir.path().join("out.nii");

    let result = run(default_args(input, output, FilterKind::Gaussian));
    assert!(result.is_err(), "missing input must yield an error");
}
