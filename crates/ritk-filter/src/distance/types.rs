//! Domain newtypes for distance transform filters.

use serde::{Deserialize, Serialize};

/// Output measure for an unsigned Euclidean distance transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum DistanceMeasure {
    /// Physical Euclidean distance.
    #[default]
    Euclidean,
    /// Squared physical Euclidean distance.
    Squared,
}

/// Binarization threshold separating background from foreground in distance transforms.
///
/// Background voxels have intensity `\u2264 threshold`; foreground voxels have intensity `> threshold`.
/// The threshold must be non-negative.
///
/// Default: `0.5` (standard for binary images stored as `{0.0, 1.0}`).
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "f32", into = "f32")]
pub struct BinarizationThreshold(f32);

impl BinarizationThreshold {
    /// Default threshold of 0.5.
    pub const DEFAULT: Self = Self(0.5);

    /// Construct a validated `BinarizationThreshold`.
    ///
    /// Returns `Err` if `value` is non-finite or negative.
    pub fn new(value: f32) -> Result<Self, &'static str> {
        if !value.is_finite() || value < 0.0 {
            return Err("BinarizationThreshold must be finite and non-negative");
        }
        Ok(Self(value))
    }
}

impl TryFrom<f32> for BinarizationThreshold {
    type Error = &'static str;

    fn try_from(value: f32) -> Result<Self, Self::Error> {
        Self::new(value)
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
