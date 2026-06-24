use super::*;

#[test]
fn extract_scalar_float_from_float() {
    let val = AttributeValue::Float(std::f64::consts::PI);
    assert!((extract_scalar_float(&val).unwrap() - std::f64::consts::PI).abs() < 1e-10);
}

#[test]
fn extract_scalar_float_from_int() {
    let val = AttributeValue::Int(42);
    assert!((extract_scalar_float(&val).unwrap() - 42.0).abs() < 1e-10);
}

#[test]
fn extract_scalar_float_from_uint() {
    let val = AttributeValue::Uint(7);
    assert!((extract_scalar_float(&val).unwrap() - 7.0).abs() < 1e-10);
}

#[test]
fn extract_scalar_float_from_single_element_array() {
    let val = AttributeValue::FloatArray(vec![2.5]);
    assert!((extract_scalar_float(&val).unwrap() - 2.5).abs() < 1e-10);
}

#[test]
fn extract_scalar_float_rejects_multi_element_array() {
    let val = AttributeValue::FloatArray(vec![1.0, 2.0]);
    assert!(extract_scalar_float(&val).is_err());
}

#[test]
fn extract_i64_from_int() {
    let val = AttributeValue::Int(-5);
    assert_eq!(extract_i64(&val).unwrap(), -5);
}

#[test]
fn extract_i64_from_uint() {
    let val = AttributeValue::Uint(100);
    assert_eq!(extract_i64(&val).unwrap(), 100);
}

#[test]
fn extract_i64_rejects_float() {
    let val = AttributeValue::Float(3.9);
    let err = extract_i64(&val).unwrap_err().to_string();
    assert!(
        err.contains("Expected scalar integer"),
        "error must reject non-integer length values; got: {err}"
    );
}

#[test]
fn extract_i64_rejects_uint_overflow() {
    let val = AttributeValue::Uint(i64::MAX as u64 + 1);
    let err = extract_i64(&val).unwrap_err().to_string();
    assert!(
        err.contains("exceeds i64::MAX"),
        "error must name unsigned integer overflow; got: {err}"
    );
}

#[test]
fn extract_float_array_3_from_exact() {
    let val = AttributeValue::FloatArray(vec![0.5, 0.7, 0.3]);
    let arr = extract_float_array_3(&val).unwrap();
    assert!((arr[0] - 0.5).abs() < 1e-10);
    assert!((arr[1] - 0.7).abs() < 1e-10);
    assert!((arr[2] - 0.3).abs() < 1e-10);
}

#[test]
fn extract_float_array_3_rejects_longer() {
    let val = AttributeValue::FloatArray(vec![1.0, 2.0, 3.0, 4.0]);
    let err = extract_float_array_3(&val).unwrap_err().to_string();
    assert!(
        err.contains("exactly 3") && err.contains("4"),
        "error must reject extra direction_cosines components; got: {err}"
    );
}

#[test]
fn extract_float_array_3_too_short_errors() {
    let val = AttributeValue::FloatArray(vec![1.0, 2.0]);
    assert!(extract_float_array_3(&val).is_err());
}

#[test]
fn extract_float_array_3_rejects_scalar() {
    let val = AttributeValue::Float(0.707);
    let err = extract_float_array_3(&val).unwrap_err().to_string();
    assert!(
        err.contains("exactly 3"),
        "error must reject scalar direction_cosines replication; got: {err}"
    );
}

#[test]
fn extract_string_from_string() {
    let val = AttributeValue::String("zspace,yspace,xspace".to_string());
    assert_eq!(extract_string(&val).unwrap(), "zspace,yspace,xspace");
}

#[test]
fn extract_string_from_bytes_strips_null() {
    let val = AttributeValue::Bytes(b"hello\0\0\0".to_vec());
    assert_eq!(extract_string(&val).unwrap(), "hello");
}

#[test]
fn extract_dimorder_default_when_absent() {
    let attrs: Vec<consus_hdf5::attribute::Hdf5Attribute> = vec![];
    let order = extract_dimorder(&attrs).unwrap();
    assert_eq!(order, vec!["zspace", "yspace", "xspace"]);
}
