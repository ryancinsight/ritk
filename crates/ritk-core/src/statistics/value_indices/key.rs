//! Bit-pattern equality key for f32 HashMap entries.

use std::hash::{Hash, Hasher};

/// A `f32` keyed by its bit pattern, suitable for use as a `HashMap` key.
///
/// `f32` cannot implement `Eq` because `NaN != NaN`, but `HashMap`
/// requires `Eq + Hash` keys. This newtype uses the **bit pattern** of
/// the float as both the equality and hash source, which is the
/// canonical Rust solution (used throughout the standard library for
/// byte-oriented keys).
///
/// - **NaN**: all `NaN` payloads hash to the same value and compare
///   equal by bit pattern, so a `HashMap<F32Key, _>` treats all `NaN`
///   voxels as a single key. (Mathematical `NaN != NaN` semantics
///   cannot be represented in a `HashMap` without external tagging.)
/// - **±0.0**: distinct keys (`+0.0` is `0x00000000`, `-0.0` is
///   `0x80000000`). Categorical/segmentation inputs do not produce
///   signed zero, so this is observable only in pathological cases.
#[derive(Copy, Clone, Debug)]
pub struct F32Key(f32);

impl F32Key {
    /// Wrap a `f32` for use as a `HashMap` key.
    #[inline]
    pub const fn new(v: f32) -> Self {
        Self(v)
    }

    /// Recover the original `f32` value.
    #[inline]
    pub const fn get(self) -> f32 {
        self.0
    }
}

impl PartialEq for F32Key {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}

impl Eq for F32Key {}

impl Hash for F32Key {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}
