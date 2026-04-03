//! Directions for Cross-Scan implementations
use burn::prelude::*;

/// Scan direction for cross-scan operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanDirection {
    /// Left to right
    HorizontalForward,
    /// Right to left
    HorizontalReverse,
    /// Top to bottom
    VerticalForward,
    /// Bottom to top
    VerticalReverse,
    /// Front to back (3D)
    DepthForward,
    /// Back to front (3D)
    DepthReverse,
}

impl ScanDirection {
    /// Get all 2D scan directions
    pub fn all_2d() -> &'static [ScanDirection] {
        &[
            ScanDirection::HorizontalForward,
            ScanDirection::HorizontalReverse,
            ScanDirection::VerticalForward,
            ScanDirection::VerticalReverse,
        ]
    }

    /// Get all 3D scan directions
    pub fn all_3d() -> &'static [ScanDirection] {
        &[
            ScanDirection::HorizontalForward,
            ScanDirection::HorizontalReverse,
            ScanDirection::VerticalForward,
            ScanDirection::VerticalReverse,
            ScanDirection::DepthForward,
            ScanDirection::DepthReverse,
        ]
    }
}
