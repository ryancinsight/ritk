#![expect(clippy::unwrap_used, reason = "ratchet RITK-UNWRAP-1")]
use super::*;
use std::f64::consts::PI;

#[test]
fn parse_header_values_reads_real_components() {
    let s = format!("1.0 2.5 {} 4.0", -PI);
    let v: Vec<f64> =
        parse_header_values(&s, "test_field", 4).expect("infallible: validated precondition");
    assert_eq!(v, vec![1.0, 2.5, -PI, 4.0]);
}

#[test]
fn parse_header_values_reads_counts() {
    let s = "10 20 30";
    let v: Vec<usize> =
        parse_header_values(s, "sizes", 3).expect("infallible: validated precondition");
    assert_eq!(v, vec![10, 20, 30]);
}

#[test]
fn parse_header_values_rejects_a_component_count_mismatch() {
    let s = "1.0 2.0";
    let err = parse_header_values::<f64>(s, "field", 3).unwrap_err();
    assert!(
        err.to_string().contains("must have 3 components, got 2"),
        "expected length-mismatch error, got: {err:#}"
    );
}

#[test]
fn parse_header_values_names_an_unparsable_token() {
    let s = "1.0 not_a_number 3.0";
    let err = parse_header_values::<f64>(s, "field", 3).unwrap_err();
    assert!(
        err.to_string().contains("not_a_number"),
        "expected error to mention the offending token, got: {err:#}"
    );
}
