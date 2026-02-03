//! SSMMorph: State Space Model-based Medical Image Registration
//!
//! Implementation based on VMambaMorph and MambaBIR papers:
//! - Selective State Space (S6) with input-dependent parameters
//! - Cross-scan mechanism for spatial modeling
//! - U-shaped encoder-decoder with hierarchical feature extraction
//!
//! # Module Structure
//!
//! ```text
//! ssmmorph/
//! ├── state_space/       - Core SSM implementation (Selective State Space S6)
//! ├── cross_scan/        - 2D/3D cross-scan for spatial sequence modeling
//! ├── vmamba_block/      - VMamba block combining CNN and SSM
//! ├── encoder/           - Hierarchical feature encoder
//! ├── decoder/           - Hierarchical decoder with skip connections
//! ├── network/           - Complete registration network
//! │   ├── architecture/  - SSMMorph network definition
//! │   ├── integration/   - Diffeomorphic integration
//! │   └── sampling/      - Grid sampling and flow composition
//! └── integration/       - ritk framework integration
//! ```
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use ritk_model::ssmmorph::{
//!     SSMMorph, SSMMorphConfig,
//!     network::presets,
//! };
//! use burn::tensor::Tensor;
//! use burn_ndarray::NdArray;
//!
//! type B = NdArray;
//! let device = Default::default();
//!
//! // Create network with preset configuration
//! let config = presets::brain_mri();
//! let network = SSMMorph::new(&config, &device);
//!
//! // Register images
//! let fixed = Tensor::<B, 5>::zeros([1, 1, 32, 32, 32], &device);
//! let moving = Tensor::<B, 5>::zeros([1, 1, 32, 32, 32], &device);
//! let output = network.forward(fixed, moving);
//! let displacement = output.displacement;
//! ```
//!
//! References:
//! - "VMambaMorph: Multi-Modality Deformable Image Registration based on Visual State Space Model"
//! - "MambaBIR: Residual Pyramid Network for Brain Image Registration with State-Space Model"
//! - "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (Gu & Dao, 2023)

// Core modules
pub mod state_space;
pub mod cross_scan;
pub mod vmamba_block;
pub mod encoder;
pub mod decoder;

// Network module with hierarchical structure
pub mod network;

// Framework integration
pub mod integration;

// Re-export core types
pub use state_space::{
    SelectiveStateSpaceConfig,
    SelectiveStateSpace,
    StateSpaceParameters,
};

pub use cross_scan::{
    CrossScanConfig,
    CrossScan,
    ScanDirection,
    Scan2D,
    Scan3D,
};

pub use vmamba_block::{
    VMambaBlockConfig,
    VMambaBlock,
    VMambaStage,
};

pub use encoder::{
    SSMMorphEncoderConfig,
    SSMMorphEncoder,
    EncoderStage,
    EncoderStageConfig,
};

pub use decoder::{
    SSMMorphDecoderConfig,
    SSMMorphDecoder,
    DecoderStage,
    DecoderStageConfig,
};

// Network re-exports (primary API)
pub use network::architecture::{
    SSMMorphConfig,
    SSMMorph,
    SSMMorphOutput,
    presets as network_presets,
};

// Additional network components
pub use network::integration::{
    IntegrationConfig,
    VelocityFieldIntegrator,
    TransformationComposer,
};

pub use network::sampling::{
    GridSampler,
    GridSamplerConfig,
    GridPaddingMode,
    InterpolationMode,
    FlowComposer,
};

// Framework integration re-exports
pub use integration::{
    SSMMorphIntegration,
    DiffeomorphicSSMMorph,
    LossComponents,
    SSMMorphAnalysis,
};
