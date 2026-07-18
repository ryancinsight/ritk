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
        value: f64 },
    /// Native affine construction rejected a checked shape.
    #[error("native physical affine construction failed: {0}")]
    PhysicalAffineConstruction(#[source] ritk_transform::transform::affine::AtlasAffineError) }
