//! Distance transform filters.
//!
//! # Filters
//!
//! - [`DistanceTransformImageFilter`] — unsigned Euclidean distance transform
//!   (ITK `DanielssonDistanceMapImageFilter` parity)
//! - [`SignedDistanceTransformImageFilter`] — signed Euclidean distance transform
//!   (ITK `SignedMaurerDistanceMapImageFilter` parity)

pub mod euclidean;

pub use euclidean::{DistanceTransformImageFilter, SignedDistanceTransformImageFilter};
