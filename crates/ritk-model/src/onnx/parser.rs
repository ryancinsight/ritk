//! ONNX document parser.
//!
//! This module provides validated ONNX document parsing and metadata extraction.
use crate::onnx::{OnnxError, OnnxGraph, OnnxMetadata, OnnxResult};
use onnx_ir::OnnxGraphBuilder;
use std::collections::HashMap;
use std::path::Path;

/// Validating ONNX document parser.
#[derive(Debug, Clone, Copy)]
pub struct OnnxParser;

impl OnnxParser {
    /// Create a validating parser.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Parse an ONNX document from a file path.
    ///
    /// This implementation uses `onnx-ir` to parse the model into a real ONNX
    /// graph, then maps the parsed graph into the crate-local IR for validation.
    pub fn parse<P: AsRef<Path>>(&self, path: P) -> OnnxResult<OnnxDocument> {
        let path = path.as_ref();
        let graph = self.parse_onnx_file(path)?;

        graph
            .validate()
            .map_err(|message| OnnxError::InvalidModel { message })?;

        let metadata = self.extract_metadata(&graph)?;
        Ok(OnnxDocument { graph, metadata })
    }

    /// Parse an ONNX file into an intermediate representation.
    fn parse_onnx_file(&self, path: &Path) -> OnnxResult<OnnxGraph> {
        let parsed = OnnxGraphBuilder::new().parse_file(path).map_err(|e| {
            OnnxError::ProtobufParseError {
                message: format!("Failed to parse ONNX file '{}': {e}", path.display()),
            }
        })?;

        let mut graph = OnnxGraph::new(
            path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("onnx_model")
                .to_string(),
        );

        // Map dynamic (non-static) inputs — static inputs are initializers handled separately.
        graph.inputs = parsed
            .inputs
            .iter()
            .filter(|arg| !arg.is_static())
            .map(|argument| {
                crate::onnx::OnnxValue::new(
                    argument.name.clone(),
                    Self::map_dtype(argument.ty.elem_type()),
                    argument
                        .ty
                        .static_shape()
                        .cloned()
                        .unwrap_or_default()
                        .into_iter()
                        .map(|dim| dim as i64)
                        .collect(),
                )
            })
            .collect();

        graph.outputs = parsed
            .outputs
            .iter()
            .map(|argument| {
                crate::onnx::OnnxValue::new(
                    argument.name.clone(),
                    Self::map_dtype(argument.ty.elem_type()),
                    argument
                        .ty
                        .static_shape()
                        .cloned()
                        .unwrap_or_default()
                        .into_iter()
                        .map(|dim| dim as i64)
                        .collect(),
                )
            })
            .collect();

        graph.nodes = parsed
            .nodes
            .iter()
            .map(|node| {
                crate::onnx::OnnxNode {
                    name: node.name().to_string(),
                    op_type: format!("{:?}", node),
                    domain: String::new(),
                    inputs: node.inputs().iter().map(|i| i.name.clone()).collect(),
                    outputs: node.outputs().iter().map(|o| o.name.clone()).collect(),
                    // The parser dependency exposes graph connectivity but not a
                    // public type-erased attribute iterator.
                    attributes: HashMap::new(),
                    doc_string: None,
                }
            })
            .collect();

        // Record only the initializer metadata exposed by the parser dependency.
        // No executable weight tensor is fabricated from unavailable payload data.
        for arg in &parsed.inputs {
            if arg.is_static() {
                let elem_type = Self::map_dtype(arg.ty.elem_type());
                let dims: Vec<i64> = arg
                    .ty
                    .static_shape()
                    .cloned()
                    .unwrap_or_default()
                    .into_iter()
                    .map(|dim| dim as i64)
                    .collect();
                let metadata = crate::onnx::OnnxValue::new(arg.name.clone(), elem_type, dims);
                graph.initializers.insert(arg.name.clone(), metadata);
            }
        }

        Ok(graph)
    }

    fn map_dtype(dtype: onnx_ir::ir::DType) -> crate::onnx::graph::OnnxElementType {
        use crate::onnx::graph::OnnxElementType as T;
        match dtype {
            onnx_ir::ir::DType::F32 => T::Float,
            onnx_ir::ir::DType::F64 => T::Float64,
            onnx_ir::ir::DType::F16 => T::Float16,
            onnx_ir::ir::DType::BF16 => T::Bfloat16,
            onnx_ir::ir::DType::I8 => T::Int8,
            onnx_ir::ir::DType::U8 => T::Uint8,
            onnx_ir::ir::DType::I16 => T::Int16,
            onnx_ir::ir::DType::U16 => T::Uint16,
            onnx_ir::ir::DType::I32 => T::Int32,
            onnx_ir::ir::DType::I64 => T::Int64,
            onnx_ir::ir::DType::U32 => T::Uint32,
            onnx_ir::ir::DType::U64 => T::Uint64,
            onnx_ir::ir::DType::Bool => T::Bool,
            _ => T::Float,
        }
    }

    /// Extract metadata from the ONNX graph.
    fn extract_metadata(&self, graph: &OnnxGraph) -> OnnxResult<OnnxMetadata> {
        let metadata = OnnxMetadata {
            producer_name: Some(graph.name.clone()),
            metadata_props: HashMap::new(),
            ..OnnxMetadata::default()
        };
        Ok(metadata)
    }
}

impl Default for OnnxParser {
    fn default() -> Self {
        Self::new()
    }
}

/// A validated ONNX document and its metadata.
pub struct OnnxDocument {
    /// The ONNX computation graph
    graph: OnnxGraph,
    /// Model metadata
    metadata: OnnxMetadata,
}

impl OnnxDocument {
    /// Get the model metadata.
    pub fn metadata(&self) -> &OnnxMetadata {
        &self.metadata
    }

    /// Get the computation graph.
    pub fn graph(&self) -> &OnnxGraph {
        &self.graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Integration test: parse basic_model.onnx from onnx-ir test fixtures.
    /// Verifies the parser maps inputs, outputs, nodes, and initializer metadata.
    #[test]
    fn test_parse_basic_model_onnx() {
        // Path to onnx-ir's basic_model.onnx fixture (copied to test_data/).
        let fixture_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("test_data")
            .join("basic_model.onnx");

        let parser = OnnxParser::new();
        let document = parser
            .parse(&fixture_path)
            .expect("public parser should accept basic_model.onnx");
        let graph = document.graph();

        // basic_model.onnx has 1 dynamic input, 1 output, and nodes:
        // Relu, PRelu, Add (see onnx-ir tests/basic.rs)
        assert!(
            !graph.inputs.is_empty(),
            "graph should have at least one dynamic (non-static) input"
        );
        assert!(
            !graph.outputs.is_empty(),
            "graph should have at least one output"
        );
        assert!(
            !graph.nodes.is_empty(),
            "graph should have computation nodes"
        );

        // Verify node names and op_types are non-empty
        for node in &graph.nodes {
            assert!(!node.name.is_empty(), "node name should be non-empty");
            assert!(
                !node.op_type.is_empty(),
                "node op_type should be non-empty for node '{}'",
                node.name
            );
            assert!(
                !node.inputs.is_empty() || node.op_type.contains("Constant"),
                "non-constant node '{}' should have inputs",
                node.name
            );
        }

        // Verify no input name collision between dynamic inputs and initializers
        let input_names: std::collections::HashSet<_> =
            graph.inputs.iter().map(|i| i.name.clone()).collect();
        for init_name in graph.initializers.keys() {
            assert!(
                !input_names.contains(init_name),
                "initializer '{}' should not also be a dynamic input",
                init_name
            );
        }

        // Verify initializers have correct element types (static args from the fixture)
        for (name, tensor) in &graph.initializers {
            assert!(
                matches!(
                    tensor.value_info.elem_type,
                    crate::onnx::graph::OnnxElementType::Float
                        | crate::onnx::graph::OnnxElementType::Int64
                        | crate::onnx::graph::OnnxElementType::Int32
                        | crate::onnx::graph::OnnxElementType::Bool,
                ),
                "initializer '{}' has unexpected dtype {:?}",
                name,
                tensor.value_info.elem_type
            );
        }

        // Verify graph validation passes
        assert_eq!(
            document.metadata().producer_name.as_deref(),
            Some("basic_model")
        );
    }

    /// Verify parse_onnx_file returns an error for a non-existent path.
    #[test]
    fn test_parse_nonexistent_file() {
        let parser = OnnxParser::new();
        let path = std::path::Path::new("/nonexistent/path/model.onnx");
        let result = parser.parse_onnx_file(path);
        assert!(
            result.is_err(),
            "parse_onnx_file should fail for nonexistent file"
        );
    }
}
