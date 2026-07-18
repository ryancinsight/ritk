//! ONNX tensor value and shape info types.

use super::OnnxElementType;

/// ONNX tensor value (input, output, or intermediate).
#[derive(Debug, Clone)]
pub struct OnnxValue {
    /// Tensor name
    pub name: String,
    /// Type and shape information
    pub value_info: OnnxValueInfo }

impl OnnxValue {
    /// Create a new ONNX value.
    pub fn new(name: String, elem_type: OnnxElementType, shape: Vec<i64>) -> Self {
        Self {
            name,
            value_info: OnnxValueInfo { elem_type, shape } }
    }
}

/// Type and shape information for an ONNX tensor.
#[derive(Debug, Clone)]
pub struct OnnxValueInfo {
    /// Element data type
    pub elem_type: OnnxElementType,
    /// Tensor shape (dimensions may be -1 for dynamic)
    pub shape: Vec<i64> }

impl OnnxValueInfo {
    /// Create new value info.
    pub fn new(elem_type: OnnxElementType, shape: Vec<i64>) -> Self {
        Self { elem_type, shape }
    }

    /// Check if shape is fully static (no dynamic dimensions).
    pub fn is_static(&self) -> bool {
        self.shape.iter().all(|&d| d > 0)
    }

    /// Get the number of elements (if shape is static).
    pub fn num_elements(&self) -> Option<usize> {
        if self.is_static() {
            Some(self.shape.iter().map(|&d| d as usize).product())
        } else {
            None
        }
    }
}
