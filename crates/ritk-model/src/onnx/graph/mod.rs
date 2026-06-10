//! ONNX computation graph intermediate representation.
//!
//! This module defines data structures for representing ONNX computation
//! graphs in memory, enabling conversion to Burn modules.

mod attribute;
mod element_type;
mod node;
mod tensor;
mod value;

#[cfg(test)]
mod tests;

pub use attribute::{FromOnnxAttribute, OnnxAttribute};
pub use element_type::OnnxElementType;
pub use node::OnnxNode;
pub use tensor::OnnxTensor;
pub use value::{OnnxValue, OnnxValueInfo};

use std::collections::HashMap;

/// ONNX computation graph.
///
/// Represents the full model graph including inputs, outputs, nodes,
/// and initializers (weights).
#[derive(Debug, Clone)]
pub struct OnnxGraph {
    /// Graph name (from ONNX model metadata)
    pub name: String,
    /// Input tensors (model inputs)
    pub inputs: Vec<OnnxValue>,
    /// Output tensors (model outputs)
    pub outputs: Vec<OnnxValue>,
    /// Computation nodes in topological order
    pub nodes: Vec<OnnxNode>,
    /// Initializers (weights/biases) as name -> tensor mapping
    pub initializers: HashMap<String, OnnxTensor>,
    /// Value info for intermediate tensors (name -> shape/type)
    pub value_info: HashMap<String, OnnxValueInfo>,
}

impl OnnxGraph {
    /// Create a new empty ONNX graph.
    pub fn new(name: String) -> Self {
        Self {
            name,
            inputs: Vec::new(),
            outputs: Vec::new(),
            nodes: Vec::new(),
            initializers: HashMap::new(),
            value_info: HashMap::new(),
        }
    }

    /// Get an input by name.
    pub fn get_input(&self, name: &str) -> Option<&OnnxValue> {
        self.inputs.iter().find(|i| i.name == name)
    }

    /// Get an output by name.
    pub fn get_output(&self, name: &str) -> Option<&OnnxValue> {
        self.outputs.iter().find(|o| o.name == name)
    }

    /// Get a node by name.
    pub fn get_node(&self, name: &str) -> Option<&OnnxNode> {
        self.nodes.iter().find(|n| n.name == name)
    }

    /// Get an initializer tensor by name.
    pub fn get_initializer(&self, name: &str) -> Option<&OnnxTensor> {
        self.initializers.get(name)
    }

    /// Check if a name refers to an initializer.
    pub fn is_initializer(&self, name: &str) -> bool {
        self.initializers.contains_key(name)
    }

    /// Get the shape of a tensor by name (from value_info or initializers).
    pub fn get_shape(&self, name: &str) -> Option<&[i64]> {
        if let Some(info) = self.value_info.get(name) {
            Some(&info.shape)
        } else if let Some(tensor) = self.initializers.get(name) {
            Some(&tensor.dims)
        } else {
            None
        }
    }

    /// Get all tensor names in the graph.
    pub fn tensor_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = Vec::new();

        for input in &self.inputs {
            names.push(&input.name);
        }
        for output in &self.outputs {
            names.push(&output.name);
        }
        for name in self.initializers.keys() {
            names.push(name.as_str());
        }
        for info in self.value_info.keys() {
            names.push(info.as_str());
        }

        names.sort();
        names.dedup();
        names
    }

    /// Validate the graph structure.
    ///
    /// Checks for:
    /// - All node inputs exist (either as inputs, initializers, or intermediate values)
    /// - All node outputs are unique
    /// - No cycles in the graph
    /// - All model outputs are produced by some node
    pub fn validate(&self) -> Result<(), String> {
        let mut available: std::collections::HashSet<&str> = std::collections::HashSet::new();

        // Add model inputs and initializers
        for input in &self.inputs {
            available.insert(&input.name);
        }
        for name in self.initializers.keys() {
            available.insert(name.as_str());
        }

        // Track produced outputs
        let mut produced: std::collections::HashSet<&str> = std::collections::HashSet::new();

        // Check each node
        for node in &self.nodes {
            // Check all inputs are available
            for input in &node.inputs {
                // Skip empty input names — these are Constant nodes whose values
                // are embedded and referenced via the Constant node itself, not a
                // separate name in the available set.
                if input.is_empty() {
                    continue;
                }
                if !available.contains(input.as_str()) {
                    return Err(format!(
                        "Node '{}' requires input '{}' which is not available",
                        node.name, input
                    ));
                }
            }

            // Add outputs to available set
            for output in &node.outputs {
                if produced.contains(output.as_str()) {
                    return Err(format!(
                        "Duplicate output '{}' from node '{}'",
                        output, node.name
                    ));
                }
                available.insert(output.as_str());
                produced.insert(output.as_str());
            }
        }

        // Check all model outputs are produced
        for output in &self.outputs {
            if !available.contains(output.name.as_str()) {
                return Err(format!(
                    "Model output '{}' is not produced by any node",
                    output.name
                ));
            }
        }

        Ok(())
    }
}
