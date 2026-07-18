use super::*;

// Type aliases for testing
type Spacing3 = Spacing<3>;

#[test]
fn test_spacing_creation() {
    let s = Spacing3::new([1.0, 2.0, 3.0]);
    assert_eq!(s[0], 1.0);
    assert_eq!(s[1], 2.0);
    assert_eq!(s[2], 3.0);
}

#[test]
fn test_spacing_uniform() {
    let s = Spacing3::uniform(1.0);
    assert_eq!(s[0], 1.0);
    assert_eq!(s[1], 1.0);
    assert_eq!(s[2], 1.0);
}

#[test]
fn test_spacing_is_uniform() {
    let uniform = Spacing3::uniform(1.0);
    assert!(uniform.is_uniform());

    let non_uniform = Spacing3::new([1.0, 2.0, 3.0]);
    assert!(!non_uniform.is_uniform());
}

#[test]
fn test_spacing_min_max() {
    let s = Spacing3::new([1.0, 2.0, 3.0]);
    assert_eq!(s.min_spacing(), 1.0);
    assert_eq!(s.max_spacing(), 3.0);
}

#[test]
fn test_spacing_deref_vector_api() {
    let s = Spacing3::new([1.0, 2.0, 3.0]);
    // Deref provides Vector methods
    assert_eq!(s.to_array(), [1.0, 2.0, 3.0]);
}

#[test]
fn test_spacing_from_vector_roundtrip() {
    let v = Vector::new([1.0, 2.0, 3.0]);
    let s: Spacing3 = v.into();
    let v2: Vector<3> = s.into();
    assert_eq!(v2, Vector::new([1.0, 2.0, 3.0]));
}

#[test]
fn test_spacing_repr_transparent_size() {
    // #[repr(transparent)] guarantees same layout
    assert_eq!(
        std::mem::size_of::<Spacing<3>>(),
        std::mem::size_of::<Vector<3>>()
    );
}

#[test]
#[should_panic(expected = "must be positive")]
fn test_spacing_new_rejects_zero() {
    Spacing3::new([1.0, 0.0, 3.0]);
}

#[test]
#[should_panic(expected = "must be positive")]
fn test_spacing_new_rejects_negative() {
    Spacing3::new([1.0, -0.5, 3.0]);
}

#[test]
#[should_panic(expected = "must be positive")]
fn test_spacing_new_rejects_nan() {
    Spacing3::new([1.0, f64::NAN, 3.0]);
}

#[test]
#[should_panic(expected = "must be positive")]
fn test_spacing_uniform_rejects_zero() {
    Spacing3::uniform(0.0);
}

#[test]
fn test_spacing_try_new_rejects_invalid() {
    let err = Spacing3::try_new([1.0, 0.0, 3.0]).unwrap_err();
    assert_eq!(err.index, 1);
    assert_eq!(err.value, 0.0);

    let err = Spacing3::try_new([-1.0, 2.0, 3.0]).unwrap_err();
    assert_eq!(err.index, 0);

    assert!(Spacing3::try_new([1.0, 2.0, 3.0]).is_ok());
}

#[test]
fn test_spacing_new_unchecked_skips_validation() {
    // SAFETY: test-only, caller guarantees are not actually satisfied
    // but we verify the constructor does not panic.
    let s = unsafe { Spacing3::new_unchecked([1.0, 2.0, 3.0]) };
    assert_eq!(s[0], 1.0);
}

#[test]
fn test_invalid_spacing_display() {
    let err = InvalidSpacing {
        index: 2,
        value: -1.5 };
    assert_eq!(
        format!("{err}"),
        "spacing component [2] must be positive, got -1.5"
    );
}
