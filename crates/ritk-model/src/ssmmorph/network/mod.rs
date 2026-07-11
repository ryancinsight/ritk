//! SSMMorph Network Module - Hierarchical Registration Architecture
//!
//! This module provides the complete SSMMorph network implementation
//! organized into focused submodules following separation of concerns.
//!
//! # Module Structure
//!
//! ```text
//! network/
//! └── architecture.rs    - Main network and presets
//! ```
//!
//! # Usage Example
//!
//! ```rust,no_run
//! use ritk_model::ssmmorph::network::{SSMMorph, SSMMorphConfig};
//! use coeus_autograd::Var;
//! use coeus_core::MoiraiBackend;
//! use coeus_tensor::Tensor;
//!
//! let config = SSMMorphConfig::for_3d_registration();
//! let network = SSMMorph::<MoiraiBackend>::new(&config);
//!
//! let fixed = Var::new(Tensor::zeros_on([1, 1, 32, 32, 32], &MoiraiBackend::new()), false);
//! let moving = Var::new(Tensor::zeros_on([1, 1, 32, 32, 32], &MoiraiBackend::new()), false);
//! let displacement = network.forward(&fixed, &moving)?.displacement;
//! # Ok::<(), ritk_model::ModelError>(())
//! ```

pub mod architecture;

// Re-export main types for convenience
pub use architecture::{presets, SSMMorph, SSMMorphConfig, SSMMorphOutput};
