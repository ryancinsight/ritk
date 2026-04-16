//! ONNX computation graph intermediate representation.
//!
//! This module defines data structures for representing ONNX computation
//! graphs in memory, enabling conversion to Burn modules.

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
        let mut available: std::collections::HashSet<String> = std::collections::HashSet::new();

        // Add model inputs and initializers
        for input in &self.inputs {
            available.insert(input.name.clone());
        }
        for name in self.initializers.keys() {
            available.insert(name.clone());
        }

        // Track produced outputs
        let mut produced: std::collections::HashSet<String> = std::collections::HashSet::new();

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
                if !available.contains(input) {
                    return Err(format!(
                        "Node '{}' requires input '{}' which is not available",
                        node.name, input
                    ));
                }
            }

            // Add outputs to available set
            for output in &node.outputs {
                if produced.contains(output) {
                    return Err(format!(
                        "Duplicate output '{}' from node '{}'",
                        output, node.name
                    ));
                }
                available.insert(output.clone());
                produced.insert(output.clone());
            }
        }

        // Check all model outputs are produced
        for output in &self.outputs {
            if !available.contains(&output.name) {
                return Err(format!(
                    "Model output '{}' is not produced by any node",
                    output.name
                ));
            }
        }

        Ok(())
    }
}

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
    pub doc_string: Option<String>,
}

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
            doc_string: None,
        }
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

/// ONNX tensor value (input, output, or intermediate).
#[derive(Debug, Clone)]
pub struct OnnxValue {
    /// Tensor name
    pub name: String,
    /// Type and shape information
    pub value_info: OnnxValueInfo,
}

impl OnnxValue {
    /// Create a new ONNX value.
    pub fn new(name: String, elem_type: OnnxElementType, shape: Vec<i64>) -> Self {
        Self {
            name,
            value_info: OnnxValueInfo { elem_type, shape },
        }
    }
}

/// Type and shape information for an ONNX tensor.
#[derive(Debug, Clone)]
pub struct OnnxValueInfo {
    /// Element data type
    pub elem_type: OnnxElementType,
    /// Tensor shape (dimensions may be -1 for dynamic)
    pub shape: Vec<i64>,
}

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

/// ONNX element data types.
///
/// Reference: https://github.com/onnx/onnx/blob/main/onnx/onnx.proto#L484
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnnxElementType {
    /// 32-bit float
    Float,
    /// Unsigned 8-bit integer
    Uint8,
    /// Signed 8-bit integer
    Int8,
    /// Unsigned 16-bit integer
    Uint16,
    /// Signed 16-bit integer
    Int16,
    /// Signed 32-bit integer
    Int32,
    /// Signed 64-bit integer
    Int64,
    /// String (not supported for tensors)
    String,
    /// Boolean
    Bool,
    /// 16-bit float (IEEE 754-2008 binary16)
    Float16,
    /// 64-bit float
    Float64,
    /// Unsigned 32-bit integer
    Uint32,
    /// Unsigned 64-bit integer
    Uint64,
    /// Brain floating point (bfloat16)
    Bfloat16,
    /// Complex with 32-bit float real and imaginary parts
    Complex64,
    /// Complex with 64-bit float real and imaginary parts
    Complex128,
}

impl OnnxElementType {
    /// Get the size of each element in bytes.
    pub fn element_size(&self) -> usize {
        match self {
            OnnxElementType::Float => 4,
            OnnxElementType::Uint8 => 1,
            OnnxElementType::Int8 => 1,
            OnnxElementType::Uint16 => 2,
            OnnxElementType::Int16 => 2,
            OnnxElementType::Int32 => 4,
            OnnxElementType::Int64 => 8,
            OnnxElementType::String => 0, // Variable size
            OnnxElementType::Bool => 1,
            OnnxElementType::Float16 => 2,
            OnnxElementType::Float64 => 8,
            OnnxElementType::Uint32 => 4,
            OnnxElementType::Uint64 => 8,
            OnnxElementType::Bfloat16 => 2,
            OnnxElementType::Complex64 => 8,
            OnnxElementType::Complex128 => 16,
        }
    }

    /// Check if this is a floating-point type.
    pub fn is_float(&self) -> bool {
        matches!(
            self,
            OnnxElementType::Float
                | OnnxElementType::Float16
                | OnnxElementType::Float64
                | OnnxElementType::Bfloat16
        )
    }

    /// Check if this is an integer type.
    pub fn is_int(&self) -> bool {
        matches!(
            self,
            OnnxElementType::Uint8
                | OnnxElementType::Int8
                | OnnxElementType::Uint16
                | OnnxElementType::Int16
                | OnnxElementType::Int32
                | OnnxElementType::Int64
                | OnnxElementType::Uint32
                | OnnxElementType::Uint64
        )
    }
}

impl std::fmt::Display for OnnxElementType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OnnxElementType::Float => write!(f, "float32"),
            OnnxElementType::Uint8 => write!(f, "uint8"),
            OnnxElementType::Int8 => write!(f, "int8"),
            OnnxElementType::Uint16 => write!(f, "uint16"),
            OnnxElementType::Int16 => write!(f, "int16"),
            OnnxElementType::Int32 => write!(f, "int32"),
            OnnxElementType::Int64 => write!(f, "int64"),
            OnnxElementType::String => write!(f, "string"),
            OnnxElementType::Bool => write!(f, "bool"),
            OnnxElementType::Float16 => write!(f, "float16"),
            OnnxElementType::Float64 => write!(f, "float64"),
            OnnxElementType::Uint32 => write!(f, "uint32"),
            OnnxElementType::Uint64 => write!(f, "uint64"),
            OnnxElementType::Bfloat16 => write!(f, "bfloat16"),
            OnnxElementType::Complex64 => write!(f, "complex64"),
            OnnxElementType::Complex128 => write!(f, "complex128"),
        }
    }
}

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
        if self.raw_data.len() % 4 != 0 {
            return Err("Invalid data alignment".to_string());
        }
        Ok(bytemuck::cast_slice(&self.raw_data))
    }

    /// Extract data as i64 slice (if type is compatible).
    pub fn as_i64_slice(&self) -> Result<&[i64], String> {
        if self.data_type != OnnxElementType::Int64 {
            return Err(format!("Expected Int64, got {}", self.data_type));
        }
        if self.raw_data.len() % 8 != 0 {
            return Err("Invalid data alignment".to_string());
        }
        Ok(bytemuck::cast_slice(&self.raw_data))
    }

    /// Convert to f32 vector (handles type conversion).
    pub fn to_f32_vec(&self) -> Result<Vec<f32>, String> {
        match self.data_type {
            OnnxElementType::Float => {
                let slice: &[f32] = bytemuck::cast_slice(&self.raw_data);
                Ok(slice.to_vec())
            }
            OnnxElementType::Float16 => {
                // Float16 conversion not implemented yet
                Err("Float16 to f32 conversion not implemented".to_string())
            }
            OnnxElementType::Bfloat16 => {
                // BFloat16 conversion not implemented yet
                Err("Bfloat16 to f32 conversion not implemented".to_string())
            }
            OnnxElementType::Int32 => {
                let slice: &[i32] = bytemuck::cast_slice(&self.raw_data);
                Ok(slice.iter().map(|&x| x as f32).collect())
            }
            OnnxElementType::Int64 => {
                let slice: &[i64] = bytemuck::cast_slice(&self.raw_data);
                Ok(slice.iter().map(|&x| x as f32).collect())
            }
            _ => Err(format!("Cannot convert {} to f32", self.data_type)),
        }
    }
}

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
    Graphs(Vec<OnnxGraph>),
}

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
            )),
        }
    }
}

impl FromOnnxAttribute for f32 {
    fn from_onnx_attr(attr: &OnnxAttribute, name: &str, node: &str) -> Result<Self, String> {
        match attr {
            OnnxAttribute::Float(v) => Ok(*v),
            _ => Err(format!(
                "Attribute '{}' on node '{}' is not a float",
                name, node
            )),
        }
    }
}

impl FromOnnxAttribute for String {
    fn from_onnx_attr(attr: &OnnxAttribute, name: &str, node: &str) -> Result<Self, String> {
        match attr {
            OnnxAttribute::String(v) => Ok(v.clone()),
            _ => Err(format!(
                "Attribute '{}' on node '{}' is not a string",
                name, node
            )),
        }
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
            )),
        }
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
            )),
        }
    }
}

impl FromOnnxAttribute for OnnxTensor {
    fn from_onnx_attr(attr: &OnnxAttribute, name: &str, node: &str) -> Result<Self, String> {
        match attr {
            OnnxAttribute::Tensor(t) => Ok(t.clone()),
            _ => Err(format!(
                "Attribute '{}' on node '{}' is not a tensor",
                name, node
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_creation() {
        let graph = OnnxGraph::new("test_model".to_string());
        assert_eq!(graph.name, "test_model");
        assert!(graph.inputs.is_empty());
        assert!(graph.outputs.is_empty());
        assert!(graph.nodes.is_empty());
    }

    #[test]
    fn test_node_attribute_extraction() {
        let mut node = OnnxNode::new("conv1".to_string(), "Conv".to_string());
        node.attributes
            .insert("kernel_size".to_string(), OnnxAttribute::Ints(vec![3, 3]));
        node.attributes
            .insert("stride".to_string(), OnnxAttribute::Int(1));
        node.attributes
            .insert("epsilon".to_string(), OnnxAttribute::Float(1e-5));

        let kernel_size: Vec<i64> = node.get_attr("kernel_size").unwrap();
        assert_eq!(kernel_size, vec![3, 3]);

        let stride: i64 = node.get_attr("stride").unwrap();
        assert_eq!(stride, 1);

        let epsilon: f32 = node.get_attr("epsilon").unwrap();
        assert!((epsilon - 1e-5).abs() < 1e-10);
    }

    #[test]
    fn test_optional_attribute() {
        let node = OnnxNode::new("relu1".to_string(), "Relu".to_string());

        let alpha: Option<f32> = node.get_attr_opt("alpha").unwrap();
        assert!(alpha.is_none());

        let default_alpha = node.get_attr_or("alpha", 0.0f32).unwrap();
        assert!((default_alpha - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_tensor_creation() {
        let tensor = OnnxTensor::new(
            "weight".to_string(),
            vec![64, 3, 7, 7],
            OnnxElementType::Float,
        );

        assert_eq!(tensor.rank(), 4);
        assert_eq!(tensor.num_elements(), 64 * 3 * 7 * 7);
        assert_eq!(tensor.byte_size(), 64 * 3 * 7 * 7 * 4);
    }

    #[test]
    fn test_value_info() {
        let info = OnnxValueInfo::new(OnnxElementType::Float, vec![1, 3, 224, 224]);

        assert!(info.is_static());
        assert_eq!(info.num_elements(), Some(1 * 3 * 224 * 224));

        let dynamic_info = OnnxValueInfo::new(OnnxElementType::Float, vec![-1, 3, 224, 224]);
        assert!(!dynamic_info.is_static());
        assert!(dynamic_info.num_elements().is_none());
    }

    #[test]
    fn test_graph_validation() {
        let mut graph = OnnxGraph::new("test".to_string());

        // Add input
        graph.inputs.push(OnnxValue::new(
            "input".to_string(),
            OnnxElementType::Float,
            vec![1, 3, 32, 32],
        ));

        // Add initializer
        let weight = OnnxTensor::new(
            "weight".to_string(),
            vec![16, 3, 3, 3],
            OnnxElementType::Float,
        );
        graph.initializers.insert("weight".to_string(), weight);

        // Add node
        let mut conv = OnnxNode::new("conv1".to_string(), "Conv".to_string());
        conv.inputs = vec!["input".to_string(), "weight".to_string()];
        conv.outputs = vec!["output".to_string()];
        graph.nodes.push(conv);

        // Add output
        graph.outputs.push(OnnxValue::new(
            "output".to_string(),
            OnnxElementType::Float,
            vec![1, 16, 30, 30],
        ));

        assert!(graph.validate().is_ok());
    }

    #[test]
    fn test_element_type_properties() {
        assert!(OnnxElementType::Float.is_float());
        assert!(!OnnxElementType::Float.is_int());

        assert!(OnnxElementType::Int32.is_int());
        assert!(!OnnxElementType::Int32.is_float());

        assert_eq!(OnnxElementType::Float.element_size(), 4);
        assert_eq!(OnnxElementType::Int64.element_size(), 8);
        assert_eq!(OnnxElementType::Float16.element_size(), 2);
    }
}
