//! Domain newtypes for annotation structures.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Unique identifier for a segmentation label.
///
/// Replaces bare `u32` in annotation fields where the semantic is "a label ID
/// in a segmentation label table or label map". Label 0 conventionally denotes
/// background.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LabelId(pub u32);

impl LabelId {
    /// Background label (convention: ID 0).
    pub const BACKGROUND: Self = Self(0);
}

impl fmt::Display for LabelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl From<u32> for LabelId {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

impl From<LabelId> for u32 {
    fn from(value: LabelId) -> Self {
        value.0
    }
}

impl PartialEq<u32> for LabelId {
    fn eq(&self, other: &u32) -> bool {
        self.0 == *other
    }
}

impl PartialEq<LabelId> for u32 {
    fn eq(&self, other: &LabelId) -> bool {
        *self == other.0
    }
}
