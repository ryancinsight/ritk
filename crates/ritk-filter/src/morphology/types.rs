//! Domain newtypes for binary morphology filters.

use serde::{Deserialize, Serialize};

/// Foreground intensity value in binary morphology operations.
///
/// Replaces bare `f32` in struct fields and method signatures where the
/// semantic is "the voxel value treated as foreground" (ITK `SetForegroundValue`).
///
/// Default: `1.0` (ITK convention).
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ForegroundValue(pub f32);

impl ForegroundValue {
    /// Canonical foreground value of 1.0 (ITK default).
    pub const ONE: Self = Self(1.0);
}

impl From<f32> for ForegroundValue {
    fn from(value: f32) -> Self {
        Self(value)
    }
}

impl From<ForegroundValue> for f32 {
    fn from(value: ForegroundValue) -> Self {
        value.0
    }
}

impl Default for ForegroundValue {
    fn default() -> Self {
        Self::ONE
    }
}

impl PartialEq<f32> for ForegroundValue {
    fn eq(&self, other: &f32) -> bool {
        self.0 == *other
    }
}

impl PartialEq<ForegroundValue> for f32 {
    fn eq(&self, other: &ForegroundValue) -> bool {
        *self == other.0
    }
}
