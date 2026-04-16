//! Error types for ONNX import operations.
//!
//! This module defines a comprehensive error taxonomy for ONNX parsing,
//! graph conversion, and weight loading operations.

use thiserror::Error;

/// Result type alias for ONNX operations.
pub type OnnxResult<T> = Result<T, OnnxError>;

/// Error types for ONNX import pipeline.
#[derive(Debug, Error)]
pub enum OnnxError {
    /// File I/O error during ONNX file reading.
    #[error("IO error reading ONNX file '{path}': {source}")]
    IoError {
        path: String,
        #[source]
        source: std::io::Error,
    },

    /// Protobuf parsing error.
    #[error("Protobuf parsing error: {message}")]
    ProtobufParseError { message: String },

    /// Invalid ONNX model structure.
    #[error("Invalid ONNX model structure: {message}")]
    InvalidModel { message: String },

    /// Unsupported ONNX operator.
    #[error("Unsupported ONNX operator '{op_type}' at node '{node_name}'")]
    UnsupportedOperator { op_type: String, node_name: String },

    /// Missing required attribute on ONNX node.
    #[error("Missing required attribute '{attribute}' on node '{node_name}'")]
    MissingAttribute {
        attribute: String,
        node_name: String,
    },

    /// Invalid attribute value.
    #[error("Invalid attribute '{attribute}' on node '{node_name}': {message}")]
    InvalidAttribute {
        attribute: String,
        node_name: String,
        message: String,
    },

    /// Shape inference failure.
    #[error("Shape inference failed for tensor '{tensor_name}': {message}")]
    ShapeInferenceError {
        tensor_name: String,
        message: String,
    },

    /// Weight initialization error.
    #[error("Weight initialization failed for tensor '{tensor_name}': {message}")]
    WeightInitError {
        tensor_name: String,
        message: String,
    },

    /// Graph cycle detected.
    #[error("Cycle detected in ONNX computation graph involving nodes: {nodes:?}")]
    GraphCycle { nodes: Vec<String> },

    /// Missing initializer for expected weight.
    #[error("Missing initializer for weight '{name}'")]
    MissingInitializer { name: String },

    /// Data type mismatch.
    #[error("Data type mismatch for tensor '{tensor_name}': expected {expected}, got {actual}")]
    DataTypeMismatch {
        tensor_name: String,
        expected: String,
        actual: String,
    },

    /// Unsupported data type.
    #[error("Unsupported ONNX data type: {dtype}")]
    UnsupportedDataType { dtype: String },

    /// Invalid tensor shape.
    #[error("Invalid tensor shape for '{tensor_name}': {shape:?}. {message}")]
    InvalidShape {
        tensor_name: String,
        shape: Vec<i64>,
        message: String,
    },

    /// Opset version mismatch.
    #[error("Opset version mismatch: model uses version {model_version}, but maximum supported is {supported_version}")]
    OpsetVersionMismatch {
        model_version: i64,
        supported_version: i64,
    },

    /// Missing input tensor.
    #[error("Missing required input '{input_name}' for model inference")]
    MissingInput { input_name: String },

    /// Output tensor not found.
    #[error("Output tensor '{output_name}' not found in model outputs")]
    OutputNotFound { output_name: String },

    /// Conversion error from ONNX to Burn.
    #[error("Conversion error: {message}")]
    ConversionError { message: String },

    /// Generic error with context.
    #[error("{message}")]
    Generic { message: String },
}

impl From<String> for OnnxError {
    fn from(message: String) -> Self {
        OnnxError::Generic { message }
    }
}

impl OnnxError {
    /// Create an IO error with path context.
    pub fn io_error(path: impl Into<String>, source: std::io::Error) -> Self {
        OnnxError::IoError {
            path: path.into(),
            source,
        }
    }

    /// Create an unsupported operator error.
    pub fn unsupported_operator(op_type: impl Into<String>, node_name: impl Into<String>) -> Self {
        OnnxError::UnsupportedOperator {
            op_type: op_type.into(),
            node_name: node_name.into(),
        }
    }

    /// Create a missing attribute error.
    pub fn missing_attribute(attribute: impl Into<String>, node_name: impl Into<String>) -> Self {
        OnnxError::MissingAttribute {
            attribute: attribute.into(),
            node_name: node_name.into(),
        }
    }

    /// Create a shape inference error.
    pub fn shape_inference_error(
        tensor_name: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        OnnxError::ShapeInferenceError {
            tensor_name: tensor_name.into(),
            message: message.into(),
        }
    }

    /// Create a conversion error.
    pub fn conversion_error(message: impl Into<String>) -> Self {
        OnnxError::ConversionError {
            message: message.into(),
        }
    }

    /// Create a generic error.
    pub fn generic(message: impl Into<String>) -> Self {
        OnnxError::Generic {
            message: message.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = OnnxError::unsupported_operator("Conv", "conv1");
        let msg = format!("{}", err);
        assert!(msg.contains("Conv"));
        assert!(msg.contains("conv1"));
    }

    #[test]
    fn test_error_helpers() {
        let err = OnnxError::missing_attribute("kernel_size", "conv2");
        match err {
            OnnxError::MissingAttribute {
                attribute,
                node_name,
            } => {
                assert_eq!(attribute, "kernel_size");
                assert_eq!(node_name, "conv2");
            }
            _ => panic!("Expected MissingAttribute error"),
        }
    }

    #[test]
    fn test_io_error_creation() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err = OnnxError::io_error("model.onnx", io_err);

        match err {
            OnnxError::IoError { path, .. } => {
                assert_eq!(path, "model.onnx");
            }
            _ => panic!("Expected IoError"),
        }
    }
}
