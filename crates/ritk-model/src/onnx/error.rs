//! Typed ONNX document parsing failures.

use thiserror::Error;

/// Result returned by ONNX document operations.
pub type OnnxResult<T> = Result<T, OnnxError>;

/// Failure while parsing or validating an ONNX document.
#[derive(Debug, Error)]
pub enum OnnxError {
    /// The protobuf document could not be decoded.
    #[error("ONNX protobuf parse failed: {message}")]
    ProtobufParseError {
        /// Parser diagnostic with path context.
        message: String,
    },
    /// Parsed graph connectivity violates the ONNX graph contract.
    #[error("invalid ONNX model structure: {message}")]
    InvalidModel {
        /// Violated graph invariant.
        message: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validation_error_names_the_violated_contract() {
        let error = OnnxError::InvalidModel {
            message: "missing graph input x".to_string(),
        };
        assert_eq!(
            error.to_string(),
            "invalid ONNX model structure: missing graph input x"
        );
    }
}
