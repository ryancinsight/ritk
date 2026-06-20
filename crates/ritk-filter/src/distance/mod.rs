//! Distance transform filters.
//!
//! # Filters
//!
//! - [`DistanceTransformImageFilter`] — unsigned Euclidean distance transform
//!   (ITK `DanielssonDistanceMapImageFilter` parity)
//! - [`SignedDistanceTransformImageFilter`] — signed Euclidean distance transform
//!   (ITK `SignedMaurerDistanceMapImageFilter` parity)

pub mod chamfer;
pub mod euclidean;
pub mod fast_chamfer;
pub mod types;

pub use chamfer::{
    cdt, cdt_dispatch, chamfer_distance_transform, chamfer_distance_transform_generic,
    ChamferDistanceTransform, ChamferKernel, ChamferMetric, Chessboard, Taxicab,
};
pub(crate) use euclidean::signed_maurer_core;
pub use euclidean::{
    DistanceTransformImageFilter, SignedDistanceTransformImageFilter,
    SignedMaurerDistanceMapImageFilter,
};
pub use fast_chamfer::{ApproximateSignedDistanceMapFilter, FastChamferDistanceFilter};
pub use types::BinarizationThreshold;
