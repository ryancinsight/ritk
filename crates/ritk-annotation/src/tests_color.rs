use super::*;

#[test]
fn rgba_u8_default_is_opaque_black() {
    let c = RgbaBytes::default();
    assert_eq!(c.as_array(), &[0, 0, 0, 255]);
}

#[test]
fn rgba_f32_default_is_opaque_black() {
    let c = RgbaLinear::default();
    assert_eq!(c.as_array(), &[0.0, 0.0, 0.0, 1.0]);
}

#[test]
fn rgba_u8_accessors() {
    let c = RgbaBytes::new(10, 20, 30, 40);
    assert_eq!(c.r(), 10);
    assert_eq!(c.g(), 20);
    assert_eq!(c.b(), 30);
    assert_eq!(c.a(), 40);
}

#[test]
fn rgba_f32_accessors() {
    let c = RgbaLinear::new(0.1, 0.2, 0.3, 0.4);
    assert!((c.r() - 0.1).abs() < 1e-6);
    assert!((c.g() - 0.2).abs() < 1e-6);
    assert!((c.b() - 0.3).abs() < 1e-6);
    assert!((c.a() - 0.4).abs() < 1e-6);
}

#[test]
fn rgba_u8_from_array_roundtrip() {
    let arr = [255, 128, 0, 200];
    let c: RgbaBytes = arr.into();
    let back: [u8; 4] = c.into();
    assert_eq!(back, arr);
}

#[test]
fn rgba_f32_from_array_roundtrip() {
    let arr = [1.0, 0.5, 0.0, 0.8];
    let c: RgbaLinear = arr.into();
    let back: [f32; 4] = c.into();
    assert_eq!(back, arr);
}

#[test]
fn rgba_u8_to_f32_normalizes() {
    let c = RgbaBytes::new(255, 128, 0, 200);
    let f: RgbaLinear = c.into();
    assert!((f.r() - 1.0).abs() < 1e-6);
    assert!((f.g() - 128.0 / 255.0).abs() < 1e-6);
    assert!((f.b() - 0.0).abs() < 1e-6);
    assert!((f.a() - 200.0 / 255.0).abs() < 1e-6);
}

#[test]
fn rgba_f32_to_u8_clamps_and_denormalizes() {
    // Exact values: 0.0 -> 0, 1.0 -> 255
    let c = RgbaLinear::new(0.0, 1.0, 0.0, 1.0);
    let u: RgbaBytes = c.into();
    assert_eq!(u.r(), 0);
    assert_eq!(u.g(), 255);
    assert_eq!(u.b(), 0);
    assert_eq!(u.a(), 255);
}

#[test]
fn rgba_f32_to_u8_clamps_out_of_range() {
    let c = RgbaLinear::new(-0.5, 1.5, 2.0, -1.0);
    let u: RgbaBytes = c.into();
    assert_eq!(u.r(), 0);
    assert_eq!(u.g(), 255);
    assert_eq!(u.b(), 255);
    assert_eq!(u.a(), 0);
}

#[test]
fn rgba_u8_f32_roundtrip_within_tolerance() {
    let original = RgbaBytes::new(100, 200, 50, 255);
    let f: RgbaLinear = original.into();
    let back: RgbaBytes = f.into();
    assert_eq!(back, original);
}
