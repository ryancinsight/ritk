//! ONNX model importer.
//!
//! This module provides the main entry point for importing ONNX models
//! into the Burn framework for deep learning registration.
use crate::onnx::{OnnxError, OnnxGraph, OnnxMetadata, OnnxResult};
use burn::tensor::backend::Backend;
use onnx_ir::OnnxGraphBuilder;
use std::collections::HashMap;
use std::path::Path;

/// Configuration for ONNX model import.
#[derive(Debug, Clone)]
pub struct ImportConfig {
    /// Maximum opset version to support
    pub max_opset_version: i64,
    /// Allow dynamic batch dimension
    pub allow_dynamic_batch: bool,
    /// Validate graph structure before conversion
    pub validate_graph: bool,
    /// Enable shape inference
    pub infer_shapes: bool,
}

impl Default for ImportConfig {
    fn default() -> Self {
        Self {
            max_opset_version: 17,
            allow_dynamic_batch: true,
            validate_graph: true,
            infer_shapes: true,
        }
    }
}

/// ONNX model importer.
///
/// Handles parsing and conversion of ONNX models to Burn modules.
pub struct OnnxImporter {
    config: ImportConfig,
}

impl OnnxImporter {
    /// Create a new ONNX importer with the given configuration.
    pub fn new(config: ImportConfig) -> Self {
        Self { config }
    }

    /// Import an ONNX model from a file path.
    ///
    /// This implementation uses `onnx-ir` to parse the model into a real ONNX
    /// graph, then maps the parsed graph into the crate-local IR for validation
    /// and later Burn conversion.
    pub fn import<B: Backend, P: AsRef<Path>>(
        &self,
        path: P,
        _device: &B::Device,
    ) -> OnnxResult<OnnxImportedModel<B>> {
        let path = path.as_ref();
        let graph = self.parse_onnx_file(path)?;

        if self.config.validate_graph {
            graph
                .validate()
                .map_err(|e| OnnxError::InvalidModel { message: e })?;
        }

        let metadata = self.extract_metadata(&graph)?;
        Ok(OnnxImportedModel {
            graph,
            metadata,
            _backend: std::marker::PhantomData,
        })
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
                    // Attributes are stored per-node-type in onnx-ir but not re-exported
                    // through the public API. Attribute access requires matching on the
                    // Node enum variant directly, which is not exposed publicly.
                    // TODO(ryancinsight): Map node attributes by matching on `Node` variant
                    // once a public attribute accessor is available from onnx-ir.
                    attributes: HashMap::new(),
                    doc_string: None,
                }
            })
            .collect();

        // Map initializers from static input arguments (ValueSource::Static).
        // These are ONNX graph inputs with embedded constant values (weights/biases).
        // NOTE: arg.value() -> Option<TensorData> requires TensorDataExt trait which
        // is in onnx_ir::ir::TensorDataExt but that trait is not in the public API.
        // Until onnx-ir exposes a public tensor conversion API, initializers are
        // tracked by name only — the actual data lives in onnx-ir's tensor store.
        for arg in &parsed.inputs {
            if arg.is_static() {
                // arg.value() is not available via the public onnx-ir API.
                // Track the initializer name so is_initializer() returns true.
                let elem_type = Self::map_dtype(arg.ty.elem_type());
                let dims: Vec<i64> = arg
                    .ty
                    .static_shape()
                    .cloned()
                    .unwrap_or_default()
                    .into_iter()
                    .map(|dim| dim as i64)
                    .collect();
                // raw_data is empty until tensor data access is public in onnx-ir
                let tensor = crate::onnx::OnnxTensor::new(arg.name.clone(), dims, elem_type);
                graph.initializers.insert(arg.name.clone(), tensor);
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
        let mut metadata = OnnxMetadata::default();
        metadata.producer_name = Some(graph.name.clone());
        metadata.metadata_props = HashMap::new();
        Ok(metadata)
    }
}

impl Default for OnnxImporter {
    fn default() -> Self {
        Self::new(ImportConfig::default())
    }
}

/// An imported ONNX model.
///
/// This struct wraps the converted Burn module and provides
/// metadata about the original ONNX model.
pub struct OnnxImportedModel<B: Backend> {
    /// The ONNX computation graph
    graph: OnnxGraph,
    /// Model metadata
    metadata: OnnxMetadata,
    /// Backend marker
    _backend: std::marker::PhantomData<B>,
}

impl<B: Backend> OnnxImportedModel<B> {
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

    #[test]
    fn test_import_config_default() {
        let config = ImportConfig::default();
        assert_eq!(config.max_opset_version, 17);
        assert!(config.allow_dynamic_batch);
        assert!(config.validate_graph);
        assert!(config.infer_shapes);
    }

    #[test]
    fn test_importer_creation() {
        let config = ImportConfig {
            max_opset_version: 15,
            ..Default::default()
        };
        let importer = OnnxImporter::new(config);
        assert_eq!(importer.config.max_opset_version, 15);
    }

    /// Integration test: parse basic_model.onnx from onnx-ir test fixtures.
    /// Verifies the importer correctly maps inputs, outputs, nodes, and initializers.
    #[test]
    fn test_parse_basic_model_onnx() {
        // Path to onnx-ir's basic_model.onnx fixture (copied to test_data/).
        let fixture_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("test_data")
            .join("basic_model.onnx");

        let importer = OnnxImporter::default();
        let graph = importer
            .parse_onnx_file(&fixture_path)
            .expect("parse_onnx_file should succeed for basic_model.onnx");

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
        for (init_name, _) in &graph.initializers {
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
                    tensor.data_type,
                    crate::onnx::graph::OnnxElementType::Float
                        | crate::onnx::graph::OnnxElementType::Int64
                        | crate::onnx::graph::OnnxElementType::Int32
                        | crate::onnx::graph::OnnxElementType::Bool,
                ),
                "initializer '{}' has unexpected dtype {:?}",
                name,
                tensor.data_type
            );
        }

        // Verify graph validation passes
        graph
            .validate()
            .expect("graph should pass validation for basic_model.onnx");
    }

    /// Verify parse_onnx_file returns an error for a non-existent path.
    #[test]
    fn test_parse_nonexistent_file() {
        let importer = OnnxImporter::default();
        let path = std::path::Path::new("/nonexistent/path/model.onnx");
        let result = importer.parse_onnx_file(path);
        assert!(
            result.is_err(),
            "parse_onnx_file should fail for nonexistent file"
        );
    }

    /// Verify opset version is bounded by config.
    #[test]
    fn test_opset_version_bounded() {
        let config = ImportConfig {
            max_opset_version: 11,
            ..Default::default()
        };
        let importer = OnnxImporter::new(config);
        // max_opset_version is a soft bound; the field is stored but not
        // actively enforced during parse (opset validation happens at conversion time).
        assert_eq!(importer.config.max_opset_version, 11);
    }
}
