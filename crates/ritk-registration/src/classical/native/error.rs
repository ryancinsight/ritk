//! Typed failures at the native classical-registration boundary.

use thiserror::Error;

/// Failures converting between native images, Leto volumes, and physical
/// affine frames.
#[derive(Debug, Error)]
pub enum NativeConversionError {
    /// Native image host extraction failed.
    #[error("native image data extraction failed: {0}")]
    ImageData(#[source] Box<dyn std::error::Error + Send + Sync>),
    /// Leto rejected the volume shape or extracted storage.
    #[error("Leto volume construction failed: {0}")]
    LetoVolume(#[source] leto::LetoError),
    /// Native image construction failed.
    #[error("native image construction failed: {0}")]
    ImageConstruction(#[source] Box<dyn std::error::Error + Send + Sync>),
    /// The fixed image's index-to-physical matrix is singular.
    #[error("fixed image index-to-physical matrix is singular")]
    SingularFixedPhysicalFrame,
    /// A physical affine component is outside the native `f32` contract.
    #[error("{role} contains non-representable f64 value {value}")]
    NonRepresentablePhysicalAffine {
        /// Matrix or translation component family.
        role: &'static str,
        /// Failing value before conversion to `f32`.
        value: f64,
    },
    /// Native affine construction rejected a checked shape.
    #[error("native physical affine construction failed: {0}")]
    PhysicalAffineConstruction(#[source] ritk_transform::transform::affine::AtlasAffineError),
}

/// Failure converting a classical physical rigid affine into the native frame.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum RigidPhysicalAffineError {
    /// The affine does not have the canonical homogeneous final row.
    #[error("physical affine homogeneous row must be [0, 0, 0, 1], got {actual:?}")]
    InvalidHomogeneousRow {
        /// Observed final row.
        actual: [f64; 4],
    },
    /// A component cannot be represented by the native `f32` transform.
    #[error("{role} contains non-representable f64 value {value}")]
    NonRepresentable {
        /// Matrix or translation component family.
        role: &'static str,
        /// Failing value before conversion to `f32`.
        value: f64,
    },
    /// Native affine construction rejected the checked shape.
    #[error("native rigid affine construction failed: {0}")]
    Construction(#[source] ritk_transform::transform::affine::AtlasAffineError),
}
