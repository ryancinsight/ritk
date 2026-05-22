//! Threshold constants used during DICOM series post-processing.
//!
//! Shared by directory, instance, and Part-10 byte scan paths.

/// Maximum deviation from axial identity IOP to treat as "effectively axial".
pub(crate) const AXIAL_IOP_THRESHOLD: f64 = 1e-4;

/// Minimum |GantryDetectorTilt| (degrees) to trigger IOP synthesis.
pub(crate) const GANTRY_TILT_MIN_DEGREES: f64 = 0.01;

/// Maximum component-wise IOP deviation before emitting a consistency warning.
pub(crate) const IOP_CONSISTENCY_THRESHOLD: f64 = 1e-4;

/// Maximum component-wise PixelSpacing deviation before emitting a consistency warning.
pub(crate) const PIXEL_SPACING_CONSISTENCY_THRESHOLD: f64 = 1e-4;
