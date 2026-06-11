//! Domain newtypes for distance transform filters.

use serde::{Deserialize, Serialize};

/// Binarization threshold separating background from foreground in distance transforms.
///
/// Background voxels have intensity `\u2264 threshold`; foreground voxels have intensity `> threshold`.
/// The threshold must be non-negative.
///
/// Default: `0.5` (standard for binary images stored as `{0.0, 1.0}`).
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BinarizationThreshold(f32);

impl BinarizationThreshold {
    /// Default threshold of 0.5.
    pub const DEFAULT: Self = Self(0.5);

    /// Construct a validated `BinarizationThreshold`.
    ///
    /// Returns `Err` if `value < 0.0`.
    pub fn new(value: f32) -> Result<Self, &'static str> {
        if value < 0.0 {
            return Err("BinarizationThreshold must be >= 0");
        }
        Ok(Self(value))
    }
}

impl From<f32> for BinarizationThreshold {
    fn from(value: f32) -> Self {
        Self(value)
    }
}

impl From<BinarizationThreshold> for f32 {
    fn from(value: BinarizationThreshold) -> Self {
        value.0
    }
}

impl Default for BinarizationThreshold {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl PartialEq<f32> for BinarizationThreshold {
    fn eq(&self, other: &f32) -> bool {
        self.0 == *other
    }
}

impl PartialEq<BinarizationThreshold> for f32 {
    fn eq(&self, other: &BinarizationThreshold) -> bool {
        *self == other.0
    }
}
