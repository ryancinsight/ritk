#![expect(clippy::unwrap_used, reason = "ratchet RITK-UNWRAP-1")]
use super::*;
use std::f64::consts::PI;

#[test]
fn parse_spatial_values() {
    let s = format!("1.0 2.5 {} 4.0", -PI);
    let v = parse_f64_vec(&s, "test_field", 4).expect("valid spatial values");
    assert_eq!(v, vec![1.0, 2.5, -PI, 4.0]);
}

#[test]
fn parse_dimension_sizes() {
    let s = "10 20 30";
    let v = parse_usize_vec(s, "sizes", 3).expect("valid dimension sizes");
    assert_eq!(v, vec![10, 20, 30]);
}

#[test]
fn parser_rejects_missing_components() {
    let s = "1.0 2.0";
    let err = parse_f64_vec(s, "field", 3).unwrap_err();
    assert!(
        err.to_string().contains("must have exactly 3 components"),
        "expected length-mismatch error, got: {err:#}"
    );
}

#[test]
fn parser_rejects_surplus_components_before_collecting_them() {
    let err = parse_f64_vec("1.0 2.0 3.0 4.0", "field", 3).unwrap_err();
    assert!(err.to_string().contains("must have exactly 3 components"));
}

#[test]
fn parser_errors_do_not_echo_untrusted_header_tokens() {
    let err = parse_f64_vec("1.0 patient-name-secret 3.0", "spacing", 3).unwrap_err();
    let diagnostic = format!("{err:#}");
    assert!(diagnostic.contains("spacing"));
    assert!(!diagnostic.contains("patient-name-secret"));
}
