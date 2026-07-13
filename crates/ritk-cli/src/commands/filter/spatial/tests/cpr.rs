use super::*;

// ── CPR (Curved Planar Reformation) ─────────────────────────────────────────

#[test]
fn test_filter_cpr_creates_output() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("out.nii");
    ritk_io::write_nifti(&input, &make_test_image()).unwrap();

    let mut args = default_args(input, output.clone(), FilterKind::Cpr);
    args.cpr.cpr_points = vec!["0,0,0".to_string(), "4,4,4".to_string()];
    args.cpr.cpr_path_samples = 32;
    args.cpr.cpr_cross_samples = 16;
    args.cpr.cpr_half_width = 2.0;

    run_cpr(&args).expect("CPR must succeed");
    let output =
        crate::commands::read_image_native(&output).expect("CPR output must be natively readable");
    assert_eq!(
        output.shape(),
        [1, 16, 32],
        "CPR output shape must match configuration"
    );
}

#[test]
fn test_filter_cpr_insufficient_points_errors() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("out.nii");
    ritk_io::write_nifti(&input, &make_test_image()).unwrap();

    let mut args = default_args(input, output, FilterKind::Cpr);
    args.cpr.cpr_points = vec!["0,0,0".to_string()];

    let result = run_cpr(&args);
    assert!(result.is_err(), "CPR with 1 point must fail");
    let msg = format!("{:#}", result.unwrap_err());
    assert!(
        msg.contains("at least 2"),
        "error must mention minimum point count, got: {msg}"
    );
}

#[test]
fn test_filter_cpr_malformed_point_errors() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("out.nii");
    ritk_io::write_nifti(&input, &make_test_image()).unwrap();

    let mut args = default_args(input, output, FilterKind::Cpr);
    args.cpr.cpr_points = vec!["0,0".to_string(), "1,1,1".to_string()];

    let result = run_cpr(&args);
    assert!(result.is_err(), "CPR with malformed point must fail");
    let msg = format!("{:#}", result.unwrap_err());
    assert!(
        msg.contains("--cpr-point"),
        "error must reference --cpr-point flag, got: {msg}"
    );
}
