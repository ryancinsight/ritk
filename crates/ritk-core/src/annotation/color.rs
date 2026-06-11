//! RGBA color newtypes for annotation overlay color fields.
//!
//! [`RgbaU8`] stores RGBA as `[u8; 4]` in `[0, 255]`.
//! [`RgbaF32`] stores RGBA as `[f32; 4]` in `[0.0, 1.0]`.
//!
//! Both are `#[repr(transparent)]` over their inner array, guaranteeing
//! ABI compatibility with the raw array form.

use serde::{Deserialize, Serialize};

/// RGBA color stored as 8-bit unsigned integers: `[r, g, b, a]`.
///
/// Each channel is in `[0, 255]`. No validation is performed at construction;
/// the caller is responsible for supplying valid channel values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct RgbaU8(pub [u8; 4]);

impl RgbaU8 {
    /// Construct an RGBA color from four channel values.
    #[inline]
    pub fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self([r, g, b, a])
    }

    /// Red channel.
    #[inline]
    pub fn r(&self) -> u8 {
        self.0[0]
    }

    /// Green channel.
    #[inline]
    pub fn g(&self) -> u8 {
        self.0[1]
    }

    /// Blue channel.
    #[inline]
    pub fn b(&self) -> u8 {
        self.0[2]
    }

    /// Alpha channel.
    #[inline]
    pub fn a(&self) -> u8 {
        self.0[3]
    }

    /// Access the underlying `[u8; 4]` array by reference.
    #[inline]
    pub fn as_array(&self) -> &[u8; 4] {
        &self.0
    }
}

impl Default for RgbaU8 {
    /// Opaque black: `[0, 0, 0, 255]`.
    fn default() -> Self {
        Self([0, 0, 0, 255])
    }
}

impl From<[u8; 4]> for RgbaU8 {
    #[inline]
    fn from(arr: [u8; 4]) -> Self {
        Self(arr)
    }
}

impl From<RgbaU8> for [u8; 4] {
    #[inline]
    fn from(color: RgbaU8) -> Self {
        color.0
    }
}

/// RGBA color stored as 32-bit floats: `[r, g, b, a]`.
///
/// Each channel is nominally in `[0.0, 1.0]`. No validation is performed at
/// construction; the caller is responsible for supplying valid channel values.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[repr(transparent)]
pub struct RgbaF32(pub [f32; 4]);

impl RgbaF32 {
    /// Construct an RGBA color from four channel values.
    #[inline]
    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self([r, g, b, a])
    }

    /// Red channel.
    #[inline]
    pub fn r(&self) -> f32 {
        self.0[0]
    }

    /// Green channel.
    #[inline]
    pub fn g(&self) -> f32 {
        self.0[1]
    }

    /// Blue channel.
    #[inline]
    pub fn b(&self) -> f32 {
        self.0[2]
    }

    /// Alpha channel.
    #[inline]
    pub fn a(&self) -> f32 {
        self.0[3]
    }

    /// Access the underlying `[f32; 4]` array by reference.
    #[inline]
    pub fn as_array(&self) -> &[f32; 4] {
        &self.0
    }
}

impl Default for RgbaF32 {
    /// Opaque black: `[0.0, 0.0, 0.0, 1.0]`.
    fn default() -> Self {
        Self([0.0, 0.0, 0.0, 1.0])
    }
}

impl From<[f32; 4]> for RgbaF32 {
    #[inline]
    fn from(arr: [f32; 4]) -> Self {
        Self(arr)
    }
}

impl From<RgbaF32> for [f32; 4] {
    #[inline]
    fn from(color: RgbaF32) -> Self {
        color.0
    }
}

/// Normalize 0–255 to 0.0–1.0.
impl From<RgbaU8> for RgbaF32 {
    #[inline]
    fn from(c: RgbaU8) -> Self {
        Self([
            c.r() as f32 / 255.0,
            c.g() as f32 / 255.0,
            c.b() as f32 / 255.0,
            c.a() as f32 / 255.0,
        ])
    }
}

/// Clamp each channel to `[0.0, 1.0]` then denormalize to `[0, 255]`.
impl From<RgbaF32> for RgbaU8 {
    #[inline]
    fn from(c: RgbaF32) -> Self {
        fn clamp_denorm(v: f32) -> u8 {
            (v.clamp(0.0, 1.0) * 255.0).round() as u8
        }
        Self([
            clamp_denorm(c.r()),
            clamp_denorm(c.g()),
            clamp_denorm(c.b()),
            clamp_denorm(c.a()),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgba_u8_default_is_opaque_black() {
        let c = RgbaU8::default();
        assert_eq!(c.as_array(), &[0, 0, 0, 255]);
    }

    #[test]
    fn rgba_f32_default_is_opaque_black() {
        let c = RgbaF32::default();
        assert_eq!(c.as_array(), &[0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn rgba_u8_accessors() {
        let c = RgbaU8::new(10, 20, 30, 40);
        assert_eq!(c.r(), 10);
        assert_eq!(c.g(), 20);
        assert_eq!(c.b(), 30);
        assert_eq!(c.a(), 40);
    }

    #[test]
    fn rgba_f32_accessors() {
        let c = RgbaF32::new(0.1, 0.2, 0.3, 0.4);
        assert!((c.r() - 0.1).abs() < 1e-6);
        assert!((c.g() - 0.2).abs() < 1e-6);
        assert!((c.b() - 0.3).abs() < 1e-6);
        assert!((c.a() - 0.4).abs() < 1e-6);
    }

    #[test]
    fn rgba_u8_from_array_roundtrip() {
        let arr = [255, 128, 0, 200];
        let c: RgbaU8 = arr.into();
        let back: [u8; 4] = c.into();
        assert_eq!(back, arr);
    }

    #[test]
    fn rgba_f32_from_array_roundtrip() {
        let arr = [1.0, 0.5, 0.0, 0.8];
        let c: RgbaF32 = arr.into();
        let back: [f32; 4] = c.into();
        assert_eq!(back, arr);
    }

    #[test]
    fn rgba_u8_to_f32_normalizes() {
        let c = RgbaU8::new(255, 128, 0, 200);
        let f: RgbaF32 = c.into();
        assert!((f.r() - 1.0).abs() < 1e-6);
        assert!((f.g() - 128.0 / 255.0).abs() < 1e-6);
        assert!((f.b() - 0.0).abs() < 1e-6);
        assert!((f.a() - 200.0 / 255.0).abs() < 1e-6);
    }

    #[test]
    fn rgba_f32_to_u8_clamps_and_denormalizes() {
        // Exact values: 0.0 -> 0, 1.0 -> 255
        let c = RgbaF32::new(0.0, 1.0, 0.0, 1.0);
        let u: RgbaU8 = c.into();
        assert_eq!(u.r(), 0);
        assert_eq!(u.g(), 255);
        assert_eq!(u.b(), 0);
        assert_eq!(u.a(), 255);
    }

    #[test]
    fn rgba_f32_to_u8_clamps_out_of_range() {
        let c = RgbaF32::new(-0.5, 1.5, 2.0, -1.0);
        let u: RgbaU8 = c.into();
        assert_eq!(u.r(), 0);
        assert_eq!(u.g(), 255);
        assert_eq!(u.b(), 255);
        assert_eq!(u.a(), 0);
    }

    #[test]
    fn rgba_u8_f32_roundtrip_within_tolerance() {
        let original = RgbaU8::new(100, 200, 50, 255);
        let f: RgbaF32 = original.into();
        let back: RgbaU8 = f.into();
        assert_eq!(back, original);
    }
}
