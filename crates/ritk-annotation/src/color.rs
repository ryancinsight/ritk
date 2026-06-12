//! RGBA color newtypes for annotation overlay color fields.
//!
//! [`RgbaBytes`] stores RGBA as `[u8; 4]` in `[0, 255]`.
//! [`RgbaLinear`] stores RGBA as `[f32; 4]` in `[0.0, 1.0]`.
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
pub struct RgbaBytes(pub [u8; 4]);

impl RgbaBytes {
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

impl Default for RgbaBytes {
    /// Opaque black: `[0, 0, 0, 255]`.
    fn default() -> Self {
        Self([0, 0, 0, 255])
    }
}

impl From<[u8; 4]> for RgbaBytes {
    #[inline]
    fn from(arr: [u8; 4]) -> Self {
        Self(arr)
    }
}

impl From<RgbaBytes> for [u8; 4] {
    #[inline]
    fn from(color: RgbaBytes) -> Self {
        color.0
    }
}

/// RGBA color stored as 32-bit floats: `[r, g, b, a]`.
///
/// Each channel is nominally in `[0.0, 1.0]`. No validation is performed at
/// construction; the caller is responsible for supplying valid channel values.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[repr(transparent)]
pub struct RgbaLinear(pub [f32; 4]);

impl RgbaLinear {
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

impl Default for RgbaLinear {
    /// Opaque black: `[0.0, 0.0, 0.0, 1.0]`.
    fn default() -> Self {
        Self([0.0, 0.0, 0.0, 1.0])
    }
}

impl From<[f32; 4]> for RgbaLinear {
    #[inline]
    fn from(arr: [f32; 4]) -> Self {
        Self(arr)
    }
}

impl From<RgbaLinear> for [f32; 4] {
    #[inline]
    fn from(color: RgbaLinear) -> Self {
        color.0
    }
}

/// Normalize 0–255 to 0.0–1.0.
impl From<RgbaBytes> for RgbaLinear {
    #[inline]
    fn from(c: RgbaBytes) -> Self {
        Self([
            c.r() as f32 / 255.0,
            c.g() as f32 / 255.0,
            c.b() as f32 / 255.0,
            c.a() as f32 / 255.0,
        ])
    }
}

/// Clamp each channel to `[0.0, 1.0]` then denormalize to `[0, 255]`.
impl From<RgbaLinear> for RgbaBytes {
    #[inline]
    fn from(c: RgbaLinear) -> Self {
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
#[path = "tests_color.rs"]
mod tests;
