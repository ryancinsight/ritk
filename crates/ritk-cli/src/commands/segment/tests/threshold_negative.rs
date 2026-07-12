use super::*; // ── Negative: multi-otsu classes < 2 returns error ───────────────────────

#[test]
fn binary_threshold_rejects_nan_lower_bound() {
    let mut args = default_args(
        "unused-input.nii".into(),
        "unused-output.nii".into(),
        SegmentMethod::Binary,
    );
    args.lower = Some(f32::NAN);
    let error = run(args).expect_err("NaN lower bound must be rejected before I/O");
    assert!(error.to_string().contains("must not be NaN"));
}

#[test]
fn binary_threshold_rejects_nan_upper_bound() {
    let mut args = default_args(
        "unused-input.nii".into(),
        "unused-output.nii".into(),
        SegmentMethod::Binary,
    );
    args.upper = Some(f32::NAN);
    let error = run(args).expect_err("NaN upper bound must be rejected before I/O");
    assert!(error.to_string().contains("must not be NaN"));
}
#[test]
fn test_segment_multi_otsu_classes_lt_2_returns_error() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("out.nii");
    ritk_io::write_nifti(&input, &make_trimodal_image()).unwrap();
    let result = run(SegmentArgs {
        input,
        output,
        method: SegmentMethod::MultiOtsu,
        classes: 1,
        lower: None,
        upper: None,
        seed: None,
        multiplier: 2.5,
        ..Default::default()
    });
    assert!(result.is_err(), "classes < 2 must yield an error");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("\u{2265} 2"),
        "error must state the minimum class count, got: {msg}"
    );
}

// ── Boundary: parse_seed correct output ───────────────────────────────────
#[test]
fn test_parse_seed_valid_input() {
    let seed = parse_seed("4,5,6").unwrap();
    assert_eq!(seed, [4, 5, 6]);
}

#[test]
fn test_parse_seed_with_spaces() {
    let seed = parse_seed("1, 2, 3").unwrap();
    assert_eq!(seed, [1, 2, 3]);
}

#[test]
fn test_parse_seed_too_few_components_returns_error() {
    assert!(parse_seed("1,2").is_err());
}

#[test]
fn test_parse_seed_non_numeric_component_returns_error() {
    assert!(parse_seed("1,two,3").is_err());
}

#[test]
fn test_parse_seed_negative_component_returns_error() {
    assert!(parse_seed("1,-2,3").is_err());
}
