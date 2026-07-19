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
fn automatic_threshold_rejects_unknown_input_format_before_io() {
    let dir = tempdir().unwrap();
    let output = dir.path().join("output.nii");
    let error = run(default_args(
        dir.path().join("input.unknown"),
        output.clone(),
        SegmentMethod::Otsu,
    ))
    .expect_err("unknown input format must be rejected");
    assert!(error.to_string().contains("Cannot infer input format"));
    assert!(!output.exists());
}

#[test]
fn automatic_threshold_rejects_known_nonnative_output_before_io() {
    let dir = tempdir().unwrap();
    let output = dir.path().join("output.png");
    let error = run(default_args(
        dir.path().join("input.nii"),
        output.clone(),
        SegmentMethod::Otsu,
    ))
    .expect_err("PNG output lacks native threshold support");
    assert_eq!(
        error.to_string(),
        "automatic thresholding requires native input/output formats"
    );
    assert!(!output.exists());
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
        msg.contains(">= 2"),
        "error must state the minimum class count, got: {msg}"
    );
}

#[test]
fn multi_otsu_rejects_classes_above_default_bin_count_before_io() {
    let dir = tempdir().unwrap();
    let output = dir.path().join("output.nii");
    let mut args = default_args(
        dir.path().join("input.nii"),
        output.clone(),
        SegmentMethod::MultiOtsu,
    );
    args.classes = 257;
    let error = run(args).expect_err("classes above bins must be rejected before I/O");
    assert!(error.to_string().contains("must be <= 256"));
    assert!(!output.exists());
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
