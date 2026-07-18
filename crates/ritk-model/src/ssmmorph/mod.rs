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
//! â”œâ”€â”€ state_space/       - Core SSM implementation (Selective State Space S6)
//! â”œâ”€â”€ cross_scan/        - 2D/3D cross-scan for spatial sequence modeling
//! â”œâ”€â”€ vmamba_block/      - VMamba block combining CNN and SSM
//! â”œâ”€â”€ encoder/           - Hierarchical feature encoder
//! â”œâ”€â”€ decoder/           - Hierarchical decoder with skip connections
//! â”œâ”€â”€ network/           - Complete registration network
//! â”‚   â””â”€â”€ architecture/  - SSMMorph network definition and presets
//! ```
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use ritk_model::ssmmorph::{
//!     SSMMorph, SSMMorphConfig,
//!     network::presets,
//! };
//! use coeus_autograd::Var;
//! use coeus_core::MoiraiBackend;
//! use coeus_tensor::Tensor;
//!
//! // Create network with preset configuration
//! let config = presets::brain_mri();
//! let network = SSMMorph::<MoiraiBackend>::new(&config);
//!
//! // Register images
//! let fixed = Var::new(Tensor::zeros_on([1, 1, 32, 32, 32], &MoiraiBackend::new()), false);
//! let moving = Var::new(Tensor::zeros_on([1, 1, 32, 32, 32], &MoiraiBackend::new()), false);
//! let displacement = network.forward(&fixed, &moving)?.displacement;
//! # Ok::<(), ritk_model::ModelError>(())
//! ```
//!
//! References:
//! - "VMambaMorph: Multi-Modality Deformable Image Registration based on Visual State Space Model"
//! - "MambaBIR: Residual Pyramid Network for Brain Image Registration with State-Space Model"
//! - "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (Gu & Dao, 2023)

// Shared policy enums
pub mod policy;

// Core modules
pub mod cross_scan;
pub mod decoder;
pub mod encoder;
pub mod state_space;
pub mod vmamba_block;

// Network module with hierarchical structure
pub mod network;

// Re-export core types
pub use state_space::{SelectiveStateSpace, SelectiveStateSpaceConfig, StateSpaceParameters};

pub use policy::ScanDimensionality;

pub use cross_scan::{CrossScan, CrossScanConfig, Scan2D, Scan3D, ScanDirection};

pub use vmamba_block::{VMambaBlock, VMambaBlockConfig};

pub use encoder::{
    DownsamplePolicy, DropPath, EncoderStage, EncoderStageConfig, SSMMorphEncoder,
    SSMMorphEncoderConfig };

pub use decoder::{
    DecoderStage, DecoderStageConfig, SSMMorphDecoder, SSMMorphDecoderConfig, SkipConnections };

// Network re-exports (primary API)
pub use network::architecture::{
    presets as network_presets, IntegrationMode, SSMMorph, SSMMorphConfig, SSMMorphOutput };
