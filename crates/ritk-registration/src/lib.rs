#![allow(
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::doc_overindented_list_items,  // sub-list continuations (a., b., etc.) don't match heuristic
    clippy::field_reassign_with_default,  // stylistic; test code patterns
)]

//! ritk-registration: Unified image registration framework for the ritk toolkit.
//!
//! This crate provides differentiable Coeus operations and classical
//! Leto-based registration algorithms under a unified API:
//!
//! - **Differentiable registration**: Coeus autograd operations and transforms
//! - **Classical registration**: Deterministic CPU algorithms (Kabsch SVD, MI hill-climb)
//!
//! # Quick Start
//!
//! ```no_run
//! use leto::Array2;
//! use ritk_registration::classical::ImageRegistration;
//!
//! // Landmark-based rigid registration
//! let reg = ImageRegistration::default();
//! let fixed = Array2::from_vec([3, 3], vec![0., 0., 0., 1., 0., 0., 0., 1., 0.]).unwrap();
//! let moving = Array2::from_vec([3, 3], vec![1., 2., 3., 2., 2., 3., 1., 3., 3.]).unwrap();
//! let result = reg.rigid_registration_landmarks(&fixed, &moving).unwrap();
//! ```
//!
//! # Architecture
//!
//! ```text
//! ritk-registration (unified crate)
//! ├── classical/     - Non-ML algorithms
//! │   ├── engine/    - Registration orchestrator
//! │   ├── error.rs   - Error types
//! │   ├── spatial/   - Kabsch SVD, transforms
//! │   └── temporal/  - Temporal sync
//! ├── metric/        - Coeus differentiable operations
//! ├── regularization/ - Regularization terms
//! └── validation/   - Quality metrics
//! ```

pub mod atlas;
pub mod bspline_ffd;
pub mod classical;
pub(crate) mod deformable_field_ops;
pub mod demons;
pub mod diffeomorphic;
pub mod error;
pub mod label_transfer;
pub mod lddmm;
pub mod metric;
pub mod regularization;
pub mod types;
pub mod validation;

// ============================================================================
// Re-exports — SSOT for quality metrics
// ============================================================================
pub use error::{RegistrationError, Result};
pub use types::AffineTransform;
pub use validation::{NumericalCheck, ShapeValidation, ValidationConfig};

// ============================================================================
// Re-exports — Demons-family deformable registration
// ============================================================================
pub use demons::{
    DemonsConfig, DemonsResult, DiffeomorphicDemonsRegistration, InverseConsistentDemonsConfig,
    InverseConsistentDemonsResult, InverseConsistentDiffeomorphicDemonsRegistration,
    LevelSetMotionRegistration, MultiResDemonsConfig, MultiResDemonsRegistration,
    SymmetricDemonsRegistration, ThirionDemonsRegistration,
};

// ============================================================================
// Re-exports — B-Spline FFD registration
// ============================================================================
pub use bspline_ffd::{BSplineFFDConfig, BSplineFFDRegistration, BSplineFFDResult};

// ============================================================================
// Re-exports — SyN diffeomorphic registration
// ============================================================================
pub use deformable_field_ops::{
    warp_image, CpuFieldSmoother, CpuOrGpu, FieldSmoother, GpuFieldSmoother, VelocityField,
    WarpInterpolation,
};
pub use diffeomorphic::{SyNConfig, SyNRegistration, SyNResult};

// ============================================================================
// Re-exports — Multi-Resolution SyN and BSpline SyN
// ============================================================================
pub use diffeomorphic::bspline_syn::{BSplineSyNConfig, BSplineSyNRegistration, BSplineSyNResult};
pub use diffeomorphic::multires_syn::{MultiResSyNConfig, MultiResSyNRegistration};

// ============================================================================
// Re-exports — LDDMM registration
// ============================================================================
pub use lddmm::{LddmmConfig, LddmmRegistration, LddmmResult};

// ============================================================================
// Re-exports — Atlas / Groupwise registration + Label Fusion
// ============================================================================
pub use atlas::label_fusion::{
    joint_label_fusion, majority_vote, LabelFusionConfig, LabelFusionResult,
};
pub use atlas::{AtlasConfig, AtlasRegistration, AtlasResult, SubjectResult};

// ============================================================================
// Re-exports — Classical (non-ML) registration
// ============================================================================
pub use classical::{
    register_translation, ImageRegistration, MeanSquaredDifference, NormalizedCrossCorrelation,
    RegistrationQualityMetrics, RegistrationResult, SpatialTransform, TemporalQualityMetrics,
    TemporalSync, TranslationMetric, TranslationRegistrationError,
};

// ============================================================================
// Re-exports — atlas / label-map transfer (apply a transform to a label map)
// ============================================================================
pub use label_transfer::{label_centroids, warp_label_map};

// ============================================================================
// Re-exports — ANTs preprocessing pipeline
// ============================================================================
pub mod preprocessing;
pub use preprocessing::{
    ct_brain_mask, CtBrainMaskConfig, IntensityRescaleMode, PreprocessingPipeline,
    PreprocessingStep,
};
