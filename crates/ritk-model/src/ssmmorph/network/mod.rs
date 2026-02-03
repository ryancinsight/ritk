//! SSMMorph Network Module - Hierarchical Registration Architecture
//!
//! This module provides the complete SSMMorph network implementation
//! organized into focused submodules following separation of concerns.
//!
//! # Module Structure
//!
//! ```text
//! network/
//! ├── architecture.rs    - Main network (SSMMorph)
//! ├── integration.rs     - Diffeomorphic integration (scaling & squaring)
//! └── sampling.rs        - Grid sampling and flow composition
//! ```
//!
//! # Usage Example
//!
//! ```rust,no_run
//! use ritk_model::ssmmorph::network::{SSMMorph, SSMMorphConfig};
//! use burn::tensor::Tensor;
//! use burn_ndarray::NdArray;
//!
//! type B = NdArray;
//! let device = Default::default();
//!
//! let config = SSMMorphConfig::for_3d_registration();
//! let network = SSMMorph::new(&config, &device);
//!
//! let fixed = Tensor::<B, 5>::zeros([1, 1, 32, 32, 32], &device);
//! let moving = Tensor::<B, 5>::zeros([1, 1, 32, 32, 32], &device);
//! let output = network.forward(fixed, moving);
//! let displacement = output.displacement;
//! ```

pub mod architecture;
pub mod integration;
pub mod sampling;

// Re-export main types for convenience
pub use architecture::{
    SSMMorph,
    SSMMorphConfig,
    SSMMorphOutput,
    presets,
};

pub use integration::{
    IntegrationConfig,
    VelocityFieldIntegrator,
    TransformationComposer,
};

pub use sampling::{
    GridSampler,
    GridSamplerConfig,
    GridPaddingMode,
    InterpolationMode,
    FlowComposer,
};
