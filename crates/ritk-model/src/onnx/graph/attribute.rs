//! ONNX node attribute types and extraction trait.

use super::{OnnxGraph, OnnxTensor};

/// ONNX node attribute.
#[derive(Debug, Clone)]
pub enum OnnxAttribute {
    /// Floating point scalar
    Float(f32),
    /// Integer scalar
    Int(i64),
    /// String value
    String(String),
    /// Tensor data
    Tensor(OnnxTensor),
    /// Graph (sub-network, for control flow)
    Graph(OnnxGraph),
    /// List of floats
    Floats(Vec<f32>),
    /// List of integers
    Ints(Vec<i64>),
    /// List of strings
    Strings(Vec<String>),
    /// List of tensors
    Tensors(Vec<OnnxTensor>),
    /// List of graphs
    Graphs(Vec<OnnxGraph>) }

/// Trait for extracting typed values from ONNX attributes.
pub trait FromOnnxAttribute: Sized {
    fn from_onnx_attr(attr: &OnnxAttribute, name: &str, node: &str) -> Result<Self, String>;
}

impl FromOnnxAttribute for i64 {
    fn from_onnx_attr(attr: &OnnxAttribute, name: &str, node: &str) -> Result<Self, String> {
        match attr {
            OnnxAttribute::Int(v) => Ok(*v),
            _ => Err(format!(
                "Attribute '{}' on node '{}' is not an integer",
                name, node
            )) }
    }
}

impl FromOnnxAttribute for f32 {
    fn from_onnx_attr(attr: &OnnxAttribute, name: &str, node: &str) -> Result<Self, String> {
        match attr {
            OnnxAttribute::Float(v) => Ok(*v),
            _ => Err(format!(
                "Attribute '{}' on node '{}' is not a float",
                name, node
            )) }
    }
}

impl FromOnnxAttribute for String {
    fn from_onnx_attr(attr: &OnnxAttribute, name: &str, node: &str) -> Result<Self, String> {
        match attr {
            OnnxAttribute::String(v) => Ok(v.clone()),
            _ => Err(format!(
                "Attribute '{}' on node '{}' is not a string",
                name, node
            )) }
    }
}

impl FromOnnxAttribute for Vec<i64> {
    fn from_onnx_attr(attr: &OnnxAttribute, name: &str, node: &str) -> Result<Self, String> {
        match attr {
            OnnxAttribute::Ints(v) => Ok(v.clone()),
            OnnxAttribute::Int(v) => Ok(vec![*v]),
            _ => Err(format!(
                "Attribute '{}' on node '{}' is not an integer list",
                name, node
            )) }
    }
}

impl FromOnnxAttribute for Vec<f32> {
    fn from_onnx_attr(attr: &OnnxAttribute, name: &str, node: &str) -> Result<Self, String> {
        match attr {
            OnnxAttribute::Floats(v) => Ok(v.clone()),
            OnnxAttribute::Float(v) => Ok(vec![*v]),
            _ => Err(format!(
                "Attribute '{}' on node '{}' is not a float list",
                name, node
            )) }
    }
}

impl FromOnnxAttribute for OnnxTensor {
    fn from_onnx_attr(attr: &OnnxAttribute, name: &str, node: &str) -> Result<Self, String> {
        match attr {
            OnnxAttribute::Tensor(t) => Ok(t.clone()),
            _ => Err(format!(
                "Attribute '{}' on node '{}' is not a tensor",
                name, node
            )) }
    }
}
