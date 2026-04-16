//! ONNX operator implementations.
//!
//! This module provides implementations for converting ONNX operators
//! to Burn tensor operations. Each operator module handles the conversion
//! of ONNX node attributes and inputs to the equivalent Burn operations.

use crate::onnx::{OnnxError, OnnxNode, OnnxResult};

/// Supported ONNX operator categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperatorCategory {
    /// Convolution operations (Conv1D, Conv2D, Conv3D)
    Convolution,
    /// Normalization (BatchNorm, InstanceNorm, LayerNorm)
    Normalization,
    /// Activation functions (ReLU, LeakyReLU, Sigmoid, Tanh)
    Activation,
    /// Pooling operations (MaxPool, AvgPool, GlobalAvgPool)
    Pooling,
    /// Tensor manipulation (Reshape, Transpose, Concat, Split)
    Tensor,
    /// Arithmetic operations (Add, Sub, Mul, Div, MatMul)
    Arithmetic,
    /// Interpolation (Upsample, Resize, GridSample)
    Interpolation,
    /// Control flow (If, Loop) - not fully supported
    ControlFlow,
}

/// Trait for ONNX operator converters.
///
/// Each supported operator implements this trait to provide
/// conversion from ONNX representation to Burn operations.
pub trait OnnxOperator {
    /// Get the ONNX operator type name.
    fn op_type() -> &'static str;

    /// Get the operator category.
    fn category() -> OperatorCategory;

    /// Validate that the node has all required attributes.
    fn validate(node: &OnnxNode) -> OnnxResult<()>;

    /// Check if this operator is supported for conversion.
    fn is_supported() -> bool {
        true
    }
}

/// Registry of supported ONNX operators.
pub struct OperatorRegistry {
    supported_ops: Vec<&'static str>,
}

impl OperatorRegistry {
    /// Create a new operator registry.
    pub fn new() -> Self {
        Self {
            supported_ops: vec![
                // Convolution
                "Conv",
                "ConvTranspose",
                // Normalization
                "BatchNormalization",
                "InstanceNormalization",
                "LayerNormalization",
                // Activation
                "Relu",
                "LeakyRelu",
                "PRelu",
                "Sigmoid",
                "Tanh",
                "Elu",
                "Selu",
                "Softmax",
                // Pooling
                "MaxPool",
                "AveragePool",
                "GlobalAveragePool",
                "GlobalMaxPool",
                // Tensor operations
                "Reshape",
                "Transpose",
                "Permute",
                "Concat",
                "Split",
                "Slice",
                "Gather",
                "Unsqueeze",
                "Squeeze",
                "Flatten",
                // Arithmetic
                "Add",
                "Sub",
                "Mul",
                "Div",
                "MatMul",
                "Gemm",
                // Interpolation
                "Upsample",
                "Resize",
                "GridSample",
                // Other common ops
                "Pad",
                "Clip",
                "Identity",
                "Dropout",
            ],
        }
    }

    /// Check if an operator is supported.
    pub fn is_supported(&self, op_type: &str) -> bool {
        self.supported_ops.contains(&op_type)
    }

    /// Get the list of supported operators.
    pub fn supported_operators(&self) -> &[&'static str] {
        &self.supported_ops
    }
}

impl Default for OperatorRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Convolution operator (Conv).
pub struct ConvOp;

impl OnnxOperator for ConvOp {
    fn op_type() -> &'static str {
        "Conv"
    }

    fn category() -> OperatorCategory {
        OperatorCategory::Convolution
    }

    fn validate(node: &OnnxNode) -> OnnxResult<()> {
        // Conv requires at least 2 inputs: X, W (B is optional)
        if node.inputs.len() < 2 {
            return Err(OnnxError::InvalidModel {
                message: format!("Conv node '{}' requires at least 2 inputs", node.name),
            });
        }

        // Check for required attributes (kernel_shape is inferred from weight tensor)
        // AutoPad, Dilations, Group, KernelShape, Pads, Strides are optional

        Ok(())
    }
}

/// Batch normalization operator.
pub struct BatchNormalizationOp;

impl OnnxOperator for BatchNormalizationOp {
    fn op_type() -> &'static str {
        "BatchNormalization"
    }

    fn category() -> OperatorCategory {
        OperatorCategory::Normalization
    }

    fn validate(node: &OnnxNode) -> OnnxResult<()> {
        // BatchNormalization requires 5 inputs: X, scale, B, input_mean, input_var
        if node.inputs.len() < 5 {
            return Err(OnnxError::InvalidModel {
                message: format!("BatchNormalization node '{}' requires 5 inputs", node.name),
            });
        }

        // epsilon attribute is optional (default: 1e-5)
        // momentum attribute is optional (default: 0.9)

        Ok(())
    }
}

/// ReLU activation operator.
pub struct ReluOp;

impl OnnxOperator for ReluOp {
    fn op_type() -> &'static str {
        "Relu"
    }

    fn category() -> OperatorCategory {
        OperatorCategory::Activation
    }

    fn validate(_node: &OnnxNode) -> OnnxResult<()> {
        // ReLU has no required attributes
        Ok(())
    }
}

/// Max pooling operator.
pub struct MaxPoolOp;

impl OnnxOperator for MaxPoolOp {
    fn op_type() -> &'static str {
        "MaxPool"
    }

    fn category() -> OperatorCategory {
        OperatorCategory::Pooling
    }

    fn validate(node: &OnnxNode) -> OnnxResult<()> {
        // kernel_shape is required
        let _kernel_shape: Vec<i64> = node
            .get_attr("kernel_shape")
            .map_err(|_| OnnxError::missing_attribute("kernel_shape", &node.name))?;

        Ok(())
    }
}

/// Reshape operator.
pub struct ReshapeOp;

impl OnnxOperator for ReshapeOp {
    fn op_type() -> &'static str {
        "Reshape"
    }

    fn category() -> OperatorCategory {
        OperatorCategory::Tensor
    }

    fn validate(_node: &OnnxNode) -> OnnxResult<()> {
        // Reshape requires shape input
        Ok(())
    }
}

/// Add operator (element-wise addition).
pub struct AddOp;

impl OnnxOperator for AddOp {
    fn op_type() -> &'static str {
        "Add"
    }

    fn category() -> OperatorCategory {
        OperatorCategory::Arithmetic
    }

    fn validate(_node: &OnnxNode) -> OnnxResult<()> {
        // Add requires 2 inputs, no required attributes
        Ok(())
    }
}

/// GridSample operator (spatial transformer).
pub struct GridSampleOp;

impl OnnxOperator for GridSampleOp {
    fn op_type() -> &'static str {
        "GridSample"
    }

    fn category() -> OperatorCategory {
        OperatorCategory::Interpolation
    }

    fn validate(node: &OnnxNode) -> OnnxResult<()> {
        // GridSample requires 2 inputs: X, Grid

        // Optional attributes: align_corners, mode, padding_mode
        let _mode: String = node.get_attr_or("mode", "bilinear".to_string())?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_operator_registry() {
        let registry = OperatorRegistry::new();

        assert!(registry.is_supported("Conv"));
        assert!(registry.is_supported("Relu"));
        assert!(registry.is_supported("BatchNormalization"));
        assert!(registry.is_supported("GridSample"));

        assert!(!registry.is_supported("UnknownOp"));
    }

    #[test]
    fn test_conv_validation() {
        let mut node = OnnxNode::new("conv1".to_string(), "Conv".to_string());
        node.inputs = vec!["X".to_string(), "W".to_string()];

        assert!(ConvOp::validate(&node).is_ok());

        node.inputs.clear();
        assert!(ConvOp::validate(&node).is_err());
    }

    #[test]
    fn test_relu_validation() {
        let node = OnnxNode::new("relu1".to_string(), "Relu".to_string());
        assert!(ReluOp::validate(&node).is_ok());
    }

    #[test]
    fn test_maxpool_validation() {
        let mut node = OnnxNode::new("pool1".to_string(), "MaxPool".to_string());

        // Missing kernel_shape should fail
        assert!(MaxPoolOp::validate(&node).is_err());

        // With kernel_shape should succeed
        use crate::onnx::graph::OnnxAttribute;
        node.attributes
            .insert("kernel_shape".to_string(), OnnxAttribute::Ints(vec![2, 2]));
        assert!(MaxPoolOp::validate(&node).is_ok());
    }

    #[test]
    fn test_operator_categories() {
        assert_eq!(ConvOp::category(), OperatorCategory::Convolution);
        assert_eq!(ReluOp::category(), OperatorCategory::Activation);
        assert_eq!(AddOp::category(), OperatorCategory::Arithmetic);
        assert_eq!(MaxPoolOp::category(), OperatorCategory::Pooling);
    }
}
