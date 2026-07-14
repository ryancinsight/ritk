//! ONNX tensor (initializer or intermediate value).

use super::OnnxElementType;

/// ONNX tensor (initializer or intermediate value).
///
/// Holds actual tensor data for weights/biases.
#[derive(Debug, Clone)]
pub struct OnnxTensor {
    /// Tensor name
    pub name: String,
    /// Dimensions
    pub dims: Vec<i64>,
    /// Element data type
    pub data_type: OnnxElementType,
    /// Raw data as bytes (little-endian)
    pub raw_data: Vec<u8>,
}

impl OnnxTensor {
    /// Create a new ONNX tensor.
    pub fn new(name: String, dims: Vec<i64>, data_type: OnnxElementType) -> Self {
        let num_elements: usize = dims.iter().map(|&d| d as usize).product();
        let data_size = num_elements * data_type.element_size();
        Self {
            name,
            dims,
            data_type,
            raw_data: vec![0u8; data_size],
        }
    }

    /// Get the number of dimensions.
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Get the number of elements.
    pub fn num_elements(&self) -> usize {
        self.dims.iter().map(|&d| d as usize).product()
    }

    /// Get the total size in bytes.
    pub fn byte_size(&self) -> usize {
        self.raw_data.len()
    }

    /// Check if tensor is scalar (rank 0 or rank 1 with dim 1).
    pub fn is_scalar(&self) -> bool {
        self.dims.is_empty() || (self.dims.len() == 1 && self.dims[0] == 1)
    }

    /// Extract data as f32 slice (if type is compatible).
    pub fn as_f32_slice(&self) -> Result<&[f32], String> {
        if self.data_type != OnnxElementType::Float {
            return Err(format!("Expected Float, got {}", self.data_type));
        }
        if !self.raw_data.len().is_multiple_of(4) {
            return Err("Invalid data alignment".to_string());
        }
        Ok(bytemuck::cast_slice(&self.raw_data))
    }

    /// Extract data as i64 slice (if type is compatible).
    pub fn as_i64_slice(&self) -> Result<&[i64], String> {
        if self.data_type != OnnxElementType::Int64 {
            return Err(format!("Expected Int64, got {}", self.data_type));
        }
        if !self.raw_data.len().is_multiple_of(8) {
            return Err("Invalid data alignment".to_string());
        }
        Ok(bytemuck::cast_slice(&self.raw_data))
    }
}
