//! ONNX model import for deep learning registration models.
//!
//! This module provides ONNX (Open Neural Network Exchange) model import
//! capabilities for inspecting pre-trained registration-network documents.
//!
//! # Architecture
//!
//! The ONNX parsing pipeline consists of two stages:
//!
//! 1. **Protobuf Parsing**: Parse ONNX file format (`.onnx`) into a bounded,
//!    borrowed document using the format-focused Consus reader.
//!
//! 2. **Graph Validation**: Map public graph metadata into the crate-local IR
//!    and validate its connectivity.
//!
//! # Limitations
//!
//! This module parses and validates document structure. It does not claim to
//! compile or execute ONNX operators. Executable models are authored directly
//! against Coeus so parameter and operator semantics remain explicit.
//!
//! # Example
//!
//! ```ignore
//! use ritk_model::onnx::OnnxParser;
//!
//! let parser = OnnxParser::new();
//! let document = parser.parse("transmorph_brain.onnx")?;
//! println!("{} nodes", document.graph().nodes.len());
//! ```
//!
//! # References
//!
//! - ONNX Specification: <https://github.com/onnx/onnx/blob/main/docs/IR.md>
//! - ONNX Operator Schemas: <https://github.com/onnx/onnx/blob/main/docs/Operators.md>

pub mod error;
pub mod graph;
pub mod parser;

pub use error::{OnnxError, OnnxResult};
pub use graph::{OnnxAttribute, OnnxElementType, OnnxGraph, OnnxNode, OnnxTensor, OnnxValue};
pub use parser::{OnnxDocument, OnnxParser};

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
    pub metadata_props: std::collections::HashMap<String, String> }

impl Default for OnnxMetadata {
    fn default() -> Self {
        Self {
            ir_version: 7,
            opset_version: 13,
            producer_name: None,
            producer_version: None,
            domain: None,
            model_version: None,
            metadata_props: std::collections::HashMap::new() }
    }
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
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
