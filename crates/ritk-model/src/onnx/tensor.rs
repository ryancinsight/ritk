//! ONNX tensor utilities.
//!
//! This module provides helper functions for converting ONNX tensor data
//! to Burn tensor structures, handling various data types and layouts.

use crate::onnx::graph::{OnnxElementType, OnnxTensor};
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor};

/// Convert an ONNX tensor to a Burn tensor.
///
/// # Arguments
///
/// * `onnx_tensor` - The ONNX tensor to convert
/// * `device` - The device to create the tensor on
///
/// # Returns
///
/// A Burn tensor with the same data and shape.
pub fn onnx_tensor_to_burn<B: Backend, const D: usize>(
    onnx_tensor: &OnnxTensor,
    device: &B::Device,
) -> Result<Tensor<B, D>, String> {
    // Validate dimensionality
    if onnx_tensor.dims.len() != D {
        return Err(format!(
            "Dimension mismatch: ONNX tensor has {} dims, but requested {}",
            onnx_tensor.dims.len(),
            D
        ));
    }

    // Convert shape — direct array construction, no intermediate Vec
    let dims: [usize; D] = {
        let mut arr = [0usize; D];
        for (i, &d) in onnx_tensor.dims.iter().enumerate() {
            arr[i] = d as usize;
        }
        arr
    };
    let shape = Shape::new(dims);

    // Convert data based on element type
    let data = onnx_tensor.to_f32_vec()?;

    // Create Burn tensor
    let tensor_data = burn::tensor::TensorData::new(data, shape);
    Ok(Tensor::from_data(tensor_data, device))
}

/// Convert a Burn tensor to ONNX tensor format.
///
/// # Arguments
///
/// * `tensor` - The Burn tensor to convert
///
/// # Returns
///
/// An ONNX tensor with the same data and shape.
pub fn burn_tensor_to_onnx<B: Backend, const D: usize>(
    tensor: &Tensor<B, D>,
) -> Result<OnnxTensor, String> {
    let data = tensor.to_data();
    let shape: Vec<i64> = data.shape.iter().map(|&d| d as i64).collect();

    // Extract f32 values
    let mut values: Vec<f32> = data
        .as_slice::<f32>()
        .map_err(|_| "Failed to extract tensor data as f32")?
        .to_vec();

    // Create ONNX tensor
    let mut onnx_tensor = OnnxTensor::new("output".to_string(), shape, OnnxElementType::Float);

    // Reinterpret Vec<f32> allocation as Vec<u8> in-place (zero-copy).
    //
    // Safety:
    // 1. f32 has alignment 4, u8 has alignment 1 — the original pointer is
    //    valid for u8 because u8's alignment requirement is strictly weaker.
    // 2. byte_len = len * size_of::<f32>(), byte_cap = capacity * size_of::<f32>(),
    //    both correctly scaled for the new element type.
    // 3. mem::forget prevents the original Vec<f32> from deallocating.
    // 4. The new Vec<u8> takes ownership of the same heap allocation.
    // 5. The f32 values are valid bit-patterns for their original type; the
    //    u8 view is a raw byte representation consumed by ONNX protobuf.
    let byte_len = values.len() * std::mem::size_of::<f32>();
    let byte_cap = values.capacity() * std::mem::size_of::<f32>();
    let ptr = values.as_mut_ptr() as *mut u8;
    std::mem::forget(values);
    unsafe {
        onnx_tensor.raw_data = Vec::from_raw_parts(ptr, byte_len, byte_cap);
    }

    Ok(onnx_tensor)
}

/// Tensor data layout enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TensorLayout {
    /// Row-major (C-contiguous) layout
    #[default]
    RowMajor,
    /// Column-major (Fortran-contiguous) layout
    ColumnMajor,
    /// Strided layout with custom strides
    Strided,
}

/// Tensor shape utilities.
pub mod shape_utils {

    /// Calculate the number of elements from a shape.
    pub fn num_elements(shape: &[i64]) -> usize {
        shape
            .iter()
            .filter(|&&d| d > 0)
            .map(|&d| d as usize)
            .product()
    }

    /// Check if a shape is fully static.
    pub fn is_static(shape: &[i64]) -> bool {
        shape.iter().all(|&d| d > 0)
    }

    /// Calculate strides for a row-major tensor.
    pub fn row_major_strides(shape: &[i64]) -> Vec<i64> {
        let rank = shape.len();
        if rank == 0 {
            return vec![];
        }

        let mut strides = vec![1i64; rank];
        for i in (0..rank - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1].max(1);
        }
        strides
    }

    /// Infer broadcast output shape.
    pub fn broadcast_shapes(shapes: &[&[i64]]) -> Option<Vec<i64>> {
        if shapes.is_empty() {
            return None;
        }

        if shapes.len() == 1 {
            return Some(shapes[0].to_vec());
        }

        // Find maximum rank
        let max_rank = shapes.iter().map(|s| s.len()).max()?;

        // Pad shapes to same rank
        let padded: Vec<Vec<i64>> = shapes
            .iter()
            .map(|s| {
                let pad = max_rank - s.len();
                let mut padded = vec![1i64; pad];
                padded.extend_from_slice(s);
                padded
            })
            .collect();

        // Compute broadcast shape
        let mut result = vec![1i64; max_rank];
        for i in 0..max_rank {
            let mut dim = 1i64;
            for shape in &padded {
                if shape[i] == 1 {
                    continue;
                }
                if dim == 1 {
                    dim = shape[i];
                } else if dim != shape[i] {
                    return None; // Incompatible shapes
                }
            }
            result[i] = dim;
        }

        Some(result)
    }

    /// Compute squeeze shape (remove dimensions of size 1).
    pub fn squeeze_shape(shape: &[i64], axes: Option<&[i64]>) -> Vec<i64> {
        match axes {
            Some(axes) => shape
                .iter()
                .enumerate()
                .filter(|(i, _)| !axes.contains(&(*i as i64)))
                .map(|(_, &d)| d)
                .collect(),
            None => shape.iter().filter(|&&d| d != 1).cloned().collect(),
        }
    }

    /// Compute unsqueeze shape (insert dimensions of size 1).
    pub fn unsqueeze_shape(shape: &[i64], axes: &[i64]) -> Vec<i64> {
        let mut result = shape.to_vec();
        let mut sorted_axes: Vec<i64> = axes.to_vec();
        sorted_axes.sort();
        for &axis in sorted_axes.iter() {
            let axis = axis as usize;
            if axis <= result.len() {
                result.insert(axis, 1);
            }
        }
        result
    }

    /// Transpose shape by swapping two dimensions.
    pub fn transpose_shape(shape: &[i64], dim0: usize, dim1: usize) -> Vec<i64> {
        let mut result = shape.to_vec();
        if dim0 < result.len() && dim1 < result.len() {
            result.swap(dim0, dim1);
        }
        result
    }

    /// Permute shape according to permutation.
    pub fn permute_shape(shape: &[i64], perm: &[i64]) -> Vec<i64> {
        perm.iter()
            .map(|&i| shape.get(i as usize).copied().unwrap_or(1))
            .collect()
    }
}

impl std::fmt::Display for TensorLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorLayout::RowMajor => write!(f, "RowMajor"),
            TensorLayout::ColumnMajor => write!(f, "ColumnMajor"),
            TensorLayout::Strided => write!(f, "Strided"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_num_elements() {
        assert_eq!(
            shape_utils::num_elements(&[1, 3, 224, 224]),
            3 * 224 * 224
        );
        assert_eq!(shape_utils::num_elements(&[64, 3, 7, 7]), 64 * 3 * 7 * 7);
        assert_eq!(shape_utils::num_elements(&[]), 1);
    }

    #[test]
    fn test_is_static() {
        assert!(shape_utils::is_static(&[1, 3, 224, 224]));
        assert!(!shape_utils::is_static(&[-1, 3, 224, 224]));
        assert!(shape_utils::is_static(&[64]));
    }

    #[test]
    fn test_row_major_strides() {
        assert_eq!(shape_utils::row_major_strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(shape_utils::row_major_strides(&[64, 3]), vec![3, 1]);
    }

    #[test]
    fn test_broadcast_shapes() {
        let a = [1, 3, 224, 224];
        let b = [1, 3, 1, 1];
        let result = shape_utils::broadcast_shapes(&[&a[..], &b[..]]);
        assert_eq!(result, Some(vec![1, 3, 224, 224]));

        let c = [2, 3];
        let d = [3, 2];
        let result = shape_utils::broadcast_shapes(&[&c[..], &d[..]]);
        assert_eq!(result, None); // Incompatible
    }

    #[test]
    fn test_squeeze_shape() {
        let shape = [1, 3, 1, 224];
        let squeezed = shape_utils::squeeze_shape(&shape, None);
        assert_eq!(squeezed, vec![3, 224]);

        let squeezed_with_axes = shape_utils::squeeze_shape(&shape, Some(&[0, 2]));
        assert_eq!(squeezed_with_axes, vec![3, 224]);
    }

    #[test]
    fn test_unsqueeze_shape() {
        let shape = [3, 224];
        let unsqueezed = shape_utils::unsqueeze_shape(&shape, &[0, 2]);
        assert_eq!(unsqueezed, vec![1, 3, 1, 224]);
    }

    #[test]
    fn test_transpose_shape() {
        let shape = [1, 3, 224, 224];
        let transposed = shape_utils::transpose_shape(&shape, 0, 3);
        assert_eq!(transposed, vec![224, 3, 224, 1]);
    }

    #[test]
    fn test_permute_shape() {
        let shape = [1, 3, 224, 224];
        let permuted = shape_utils::permute_shape(&shape, &[0, 3, 1, 2]);
        assert_eq!(permuted, vec![1, 224, 3, 224]);
    }

    #[test]
    fn test_tensor_layout_display() {
        assert_eq!(format!("{}", TensorLayout::RowMajor), "RowMajor");
        assert_eq!(format!("{}", TensorLayout::ColumnMajor), "ColumnMajor");
    }
}
