#![expect(clippy::unwrap_used, reason = "ratchet RITK-UNWRAP-1")]
use super::*;
use std::f64::consts::PI;

#[test]
fn parse_floats_f64_round_trip() {
    let s = format!("1.0 2.5 {} 4.0", -PI);
    let v: Vec<f64> =
        parse_floats(&s, "test_field", 4).expect("infallible: validated precondition");
    assert_eq!(v, vec![1.0, 2.5, -PI, 4.0]);
}

#[test]
fn parse_floats_usize_round_trip() {
    let s = "10 20 30";
    let v: Vec<usize> = parse_floats(s, "sizes", 3).expect("infallible: validated precondition");
    assert_eq!(v, vec![10, 20, 30]);
}

#[test]
fn parse_floats_length_mismatch_returns_error() {
    let s = "1.0 2.0";
    let err = parse_floats::<f64>(s, "field", 3).unwrap_err();
    assert!(
        err.to_string().contains("must have 3 components, got 2"),
        "expected length-mismatch error, got: {err:#}"
    );
}

#[test]
fn parse_floats_bad_token_returns_error() {
    let s = "1.0 not_a_number 3.0";
    let err = parse_floats::<f64>(s, "field", 3).unwrap_err();
    assert!(
        err.to_string().contains("not_a_number"),
        "expected error to mention the offending token, got: {err:#}"
    );
}
