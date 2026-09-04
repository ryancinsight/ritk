/// Classical (non-ML) image registration algorithms.
///
/// This module provides deterministic, CPU-based registration algorithms
/// that do not require a deep-learning backend:
///
/// - **Rigid body**: Kabsch SVD landmark registration + mutual-information hill-climb
/// - **Affine**: 9-DOF MI optimisation (rotation + translation + anisotropic scale)
/// - **Temporal sync**: Cross-correlation phase estimation for multi-modal acquisitions
///
/// Registration quality metrics are re-exported from [`crate::validation`].
pub mod engine;
pub mod error;
pub mod native;
pub mod rigid_search;
mod robust_rigid;
pub mod spatial;
pub mod temporal;
pub mod translation;

// Re-export core types for convenience
pub use engine::{
    HistogramEstimator, ImageRegistration, IntensityRange, MutualInformationMetric,
    NmiNormalization, RegistrationResult, SpatiallyConditionedMutualInformationMetric,
};
pub use error::{RegistrationError, Result};
pub use native::{
    image_to_leto_volume, index_affine_to_physical, leto_volume_to_image,
    rigid_physical_affine_to_native, NativeConversionError, RigidPhysicalAffineError,
};
pub use rigid_search::{
    search_rigid_pose, RigidSearchAnchor, RigidSearchConfig, RigidSearchResult,
};
pub use robust_rigid::{fit_symmetric_trimmed_rigid, RigidCorrespondence, SymmetricRigidFit};
pub use spatial::SpatialTransform;
pub use temporal::{
    TemporalCorrelationSample, TemporalSignal, TemporalSync, TemporalSyncConfig, TemporalSyncError,
    TemporalSyncResult, TemporalSyncStatus,
};
pub use translation::{
    register_translation, MeanSquaredDifference, NormalizedCrossCorrelation, TranslationMetric,
    TranslationRegistrationError,
};

// Re-export quality metrics from validation (SSOT)
pub use crate::validation::RegistrationQualityMetrics;
