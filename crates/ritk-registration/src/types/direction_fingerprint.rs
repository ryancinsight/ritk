//! Direction fingerprint newtype for cache key comparisons.

/// Column-major 3×3 direction cosine matrix, flattened to `[f64; 9]`.
///
/// Used as a lightweight fingerprint in cache key comparisons
/// (`WFixedCache`, `HistogramCache`) where the full `Direction<3>` type
/// from `ritk-core` is unnecessary. Provides `PartialEq` and `Eq` for
/// exact structural comparison, and `From`/`Into` conversions to
/// bridge with `ritk_spatial::Direction<3>`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirectionFingerprint(pub [f64; 9]);

impl DirectionFingerprint {
    /// Construct from a raw column-major `[f64; 9]`.
    #[inline]
    pub fn new(dir: [f64; 9]) -> Self {
        Self(dir)
    }

    /// Return the inner `[f64; 9]`.
    #[inline]
    pub fn as_array(&self) -> [f64; 9] {
        self.0
    }
}

impl From<[f64; 9]> for DirectionFingerprint {
    #[inline]
    fn from(dir: [f64; 9]) -> Self {
        Self(dir)
    }
}

impl From<DirectionFingerprint> for [f64; 9] {
    #[inline]
    fn from(fp: DirectionFingerprint) -> Self {
        fp.0
    }
}
