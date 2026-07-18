//! ONNX computation node.

use std::collections::HashMap;

use super::{FromOnnxAttribute, OnnxAttribute};

/// ONNX computation node.
///
/// Represents a single operation in the ONNX computation graph.
#[derive(Debug, Clone)]
pub struct OnnxNode {
    /// Node name (unique identifier)
    pub name: String,
    /// Operator type (e.g., "Conv", "Add", "Relu")
    pub op_type: String,
    /// Operator domain (e.g., "" for standard ONNX, "com.microsoft" for MS domain)
    pub domain: String,
    /// Input tensor names
    pub inputs: Vec<String>,
    /// Output tensor names
    pub outputs: Vec<String>,
    /// Named attributes for this operator
    pub attributes: HashMap<String, OnnxAttribute>,
    /// Documentation string
    pub doc_string: Option<String> }

impl OnnxNode {
    /// Create a new ONNX node.
    pub fn new(name: String, op_type: String) -> Self {
        Self {
            name,
            op_type,
            domain: String::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            attributes: HashMap::new(),
            doc_string: None }
    }

    /// Get a required attribute by name.
    pub fn get_attr<T: FromOnnxAttribute>(&self, name: &str) -> Result<T, String> {
        self.attributes
            .get(name)
            .ok_or_else(|| {
                format!(
                    "Missing required attribute '{}' on node '{}'",
                    name, self.name
                )
            })
            .and_then(|attr| T::from_onnx_attr(attr, name, &self.name))
    }

    /// Get an optional attribute by name.
    pub fn get_attr_opt<T: FromOnnxAttribute>(&self, name: &str) -> Result<Option<T>, String> {
        self.attributes
            .get(name)
            .map(|attr| T::from_onnx_attr(attr, name, &self.name))
            .transpose()
    }

    /// Get an attribute with a default value.
    pub fn get_attr_or<T: FromOnnxAttribute>(&self, name: &str, default: T) -> Result<T, String> {
        self.get_attr_opt(name)?.map(Ok).unwrap_or(Ok(default))
    }
}
