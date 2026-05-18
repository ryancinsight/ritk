use super::*; // ── Negative: unknown method returns descriptive error ────────────────────
#[test]
fn test_segment_unknown_method_returns_error() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("out.nii");
    ritk_io::write_nifti(&input, &make_bimodal_image()).unwrap();
    let result = run(SegmentArgs {
        input,
        output,
        method: "nonexistent".to_string(),
        classes: 3,
        lower: None,
        upper: None,
        seed: None,
        multiplier: 2.5,
        ..Default::default()
    });
    assert!(result.is_err(), "unknown method must return Err");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("Unknown segmentation method 'nonexistent'"),
        "error must name the unsupported method, got: {msg}"
    );
}

// ── Negative: multi-otsu classes < 2 returns error ───────────────────────
#[test]
fn test_segment_multi_otsu_classes_lt_2_returns_error() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("out.nii");
    ritk_io::write_nifti(&input, &make_trimodal_image()).unwrap();
    let result = run(SegmentArgs {
        input,
        output,
        method: "multi-otsu".to_string(),
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
