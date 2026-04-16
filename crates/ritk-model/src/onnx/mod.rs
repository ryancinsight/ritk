//! ONNX model import for deep learning registration models.
//!
//! This module provides ONNX (Open Neural Network Exchange) model import
//! capabilities for loading pre-trained deep learning registration networks
//! (TransMorph, VoxelMorph, etc.) into the Burn framework.
//!
//! # Architecture
//!
//! The ONNX import pipeline consists of three stages:
//!
//! 1. **Protobuf Parsing**: Parse ONNX file format (`.onnx`) into an intermediate
//!    representation (IR) using the `onnx-ir` crate from the Burn ecosystem.
//!
//! 2. **Graph Conversion**: Convert ONNX computation graph to Burn's `Module<B>`
//!    representation, mapping ONNX operators to Burn tensor operations.
//!
//! 3. **Weight Loading**: Extract and initialize tensor weights from ONNX
//!    initializers into Burn tensor structures.
//!
//! # Supported Operators
//!
//! Current support focuses on registration-relevant operations:
//!
//! - **Convolution**: Conv1D, Conv2D, Conv3D (with padding, stride, dilation)
//! - **Normalization**: BatchNorm, InstanceNorm, LayerNorm
//! - **Activation**: ReLU, LeakyReLU, PReLU, Sigmoid, Tanh, ELU
//! - **Pooling**: MaxPool, AvgPool, GlobalAvgPool
//! - **Tensor Ops**: Concat, Split, Reshape, Transpose, Permute
//! - **Arithmetic**: Add, Sub, Mul, Div, MatMul
//! - **Interpolation**: Upsample, Resize (nearest, linear, trilinear)
//! - **Transform**: GridSample (spatial transformer networks)
//!
//! # Limitations
//!
//! - Dynamic axes are partially supported (batch dimension only)
//! - Control flow operators (If, Loop) are not supported
//! - Custom operators require manual registration
//! - RNN/LSTM operators are not yet supported
//!
//! # Example
//!
//! ```ignore
//! use ritk_model::onnx::{OnnxImporter, ImportConfig};
//! use burn::backend::Wgpu;
//! use burn_ndarray::NdArray;
//!
//! type Backend = Wgpu;
//! let device = burn::backend::wgpu::WgpuDevice::default();
//!
//! // Import ONNX model
//! let config = ImportConfig::default();
//! let importer = OnnxImporter::new(config);
//! let model = importer.import::<Backend>("transmorph_brain.onnx", &device)?;
//!
//! // Run inference
//! let input = Tensor::<Backend, 5>::zeros([1, 2, 64, 64, 64], &device);
//! let output = model.forward(input);
//! ```
//!
//! # References
//!
//! - ONNX Specification: https://github.com/onnx/onnx/blob/main/docs/IR.md
//! - ONNX Operator Schemas: https://github.com/onnx/onnx/blob/main/docs/Operators.md
//! - Burn ONNX Import: https://github.com/tracel-ai/burn-onnx

pub mod error;
pub mod graph;
pub mod importer;
pub mod ops;
pub mod tensor;

pub use error::{OnnxError, OnnxResult};
pub use graph::{OnnxGraph, OnnxNode, OnnxTensor, OnnxValue};
pub use importer::{ImportConfig, OnnxImporter};

use burn::tensor::backend::Backend;

/// Trait for ONNX-imported models.
///
/// Models imported from ONNX must implement this trait to provide
/// a common interface for inference and introspection.
pub trait OnnxModel<B: Backend> {
    /// Get the model's input names in order.
    fn input_names(&self) -> &[String];

    /// Get the model's output names in order.
    fn output_names(&self) -> &[String];

    /// Get the ONNX opset version used by this model.
    fn opset_version(&self) -> i64;

    /// Get the producer name (e.g., "pytorch", "tensorflow").
    fn producer(&self) -> Option<&str>;
}

/// Metadata extracted from ONNX model file.
#[derive(Debug, Clone)]
pub struct OnnxMetadata {
    /// IR version of the ONNX model
    pub ir_version: i64,
    /// ONNX operator set version
    pub opset_version: i64,
    /// Producer name (e.g., "pytorch", "tensorflow")
    pub producer_name: Option<String>,
    /// Producer version
    pub producer_version: Option<String>,
    /// Model domain (e.g., "org.pytorch")
    pub domain: Option<String>,
    /// Model version
    pub model_version: Option<i64>,
    /// Custom metadata as key-value pairs
    pub metadata_props: std::collections::HashMap<String, String>,
}

impl Default for OnnxMetadata {
    fn default() -> Self {
        Self {
            ir_version: 7,
            opset_version: 13,
            producer_name: None,
            producer_version: None,
            domain: None,
            model_version: None,
            metadata_props: std::collections::HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_default() {
        let meta = OnnxMetadata::default();
        assert_eq!(meta.ir_version, 7);
        assert_eq!(meta.opset_version, 13);
        assert!(meta.producer_name.is_none());
    }

    #[test]
    fn test_metadata_custom() {
        let mut meta = OnnxMetadata::default();
        meta.producer_name = Some("pytorch".to_string());
        meta.opset_version = 17;

        assert_eq!(meta.producer_name, Some("pytorch".to_string()));
        assert_eq!(meta.opset_version, 17);
    }
}
