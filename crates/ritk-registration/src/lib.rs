//! ritk-registration: Unified image registration framework for the ritk toolkit.
//!
//! This crate provides both ML-based (deep learning) and classical (non-ML)
//! registration algorithms under a unified API:
//!
//! - **ML-based registration**: Uses Burn autodiff for gradient-based optimization
//! - **Classical registration**: Deterministic CPU algorithms (Kabsch SVD, MI hill-climb)
//!
//! # Quick Start
//!
//! ```no_run
//! use ritk_registration::classical::ImageRegistration;
//! use ndarray::Array2;
//!
//! // Landmark-based rigid registration
//! let reg = ImageRegistration::default();
//! let fixed = Array2::from_shape_vec((3, 3), vec![0., 0., 0., 1., 0., 0., 0., 1., 0.]).unwrap();
//! let moving = Array2::from_shape_vec((3, 3), vec![1., 2., 3., 2., 2., 3., 1., 3., 3.]).unwrap();
//! let result = reg.rigid_registration_landmarks(&fixed, &moving).unwrap();
//! ```
//!
//! # Architecture
//!
//! ```text
//! ritk-registration (unified crate)
//! ├── classical/     - Non-ML algorithms (SSOT)
//! │   ├── engine.rs  - Registration orchestrator
//! │   ├── error.rs   - Error types
//! │   ├── intensity.rs - MI, NCC, correlation
//! │   ├── spatial.rs - Kabsch SVD, transforms
//! │   └── temporal.rs - Temporal sync
//! ├── metric/       - Similarity metrics (ML path)
//! ├── optimizer/    - Optimization algorithms
//! ├── registration/ - Registration workflow
//! ├── regularization/ - Regularization terms
//! └── validation/   - Quality metrics (SSOT)
//! ```

pub mod atlas;
pub mod bspline_ffd;
pub mod classical;
pub(crate) mod deformable_field_ops;
pub mod demons;
pub mod diffeomorphic;
pub mod error;
pub mod lddmm;
pub mod metric;
pub mod multires;
pub mod optimizer;
pub mod progress;
pub mod registration;
pub mod regularization;
pub mod validation;

// ============================================================================
// Re-exports — SSOT for quality metrics
// ============================================================================
pub use error::{RegistrationError, Result};
pub use progress::{
    ConsoleProgressCallback, ConvergenceChecker, EarlyStoppingCallback, HistoryCallback,
    ProgressCallback, ProgressInfo, ProgressTracker,
};
pub use validation::ValidationConfig;

// ============================================================================
// Re-exports — Demons-family deformable registration
// ============================================================================
pub use demons::{
    DemonsConfig, DemonsResult, DiffeomorphicDemonsRegistration, MultiResDemonsConfig,
    MultiResDemonsRegistration, SymmetricDemonsRegistration, ThirionDemonsRegistration,
};

// ============================================================================
// Re-exports — B-Spline FFD registration
// ============================================================================
pub use bspline_ffd::{BSplineFFDConfig, BSplineFFDRegistration, BSplineFFDResult};

// ============================================================================
// Re-exports — SyN diffeomorphic registration
// ============================================================================
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
    ImageRegistration, RegistrationQualityMetrics, RegistrationResult, SpatialTransform,
    TemporalQualityMetrics, TemporalSync,
};

// ============================================================================
// Re-exports — ML-based registration
// ============================================================================
pub use registration::{Registration, RegistrationConfig, RegistrationSummary};

// ============================================================================
// Re-exports — ANTs preprocessing pipeline
// ============================================================================
pub mod preprocessing;
pub use preprocessing::{NormalizationMode, PreprocessingPipeline, PreprocessingStep};
