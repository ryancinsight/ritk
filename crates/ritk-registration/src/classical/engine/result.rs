//! Registration result types for classical algorithms.

use super::super::spatial::SpatialTransform;
use crate::types::AffineTransform;
use crate::validation::RegistrationQualityMetrics;

/// Result of a classical registration operation.
#[derive(Debug, Clone)]
pub struct RegistrationResult {
    /// Final 4x4 homogeneous transformation matrix.
    pub transform: AffineTransform,
    /// Spatial transform classification.
    pub spatial: SpatialTransform,
    /// Registration quality metrics.
    pub quality: RegistrationQualityMetrics,
}
