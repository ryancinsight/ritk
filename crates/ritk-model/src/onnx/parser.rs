//! Bounded ONNX document parsing and crate-local graph validation.

use crate::onnx::{OnnxError, OnnxGraph, OnnxMetadata, OnnxResult};
use consus_onnx::{ElementType, ModelDocument, ParseLimits, TensorInfo, ValueInfo};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::path::Path;

/// Validating ONNX document parser with explicit hostile-input bounds.
#[derive(Debug, Clone, Copy)]
pub struct OnnxParser {
    limits: ParseLimits }

impl OnnxParser {
    /// Create a parser using the Consus resource limits.
    #[must_use]
    pub fn new() -> Self {
        Self {
            limits: ParseLimits::default() }
    }

    /// Create a parser with caller-selected resource limits.
    #[must_use]
    pub const fn with_limits(limits: ParseLimits) -> Self {
        Self { limits }
    }

    /// Parse and validate an ONNX document from a file path.
    ///
    /// Names and initializer payloads remain borrowed while Consus decodes the
    /// document. Only graph metadata required by RITK's owned validation IR is
    /// copied across the boundary; weight payloads are not fabricated or copied.
    pub fn parse<P: AsRef<Path>>(&self, path: P) -> OnnxResult<OnnxDocument> {
        let path = path.as_ref();
        let mut file = File::open(path).map_err(|error| Self::file_error(path, error))?;
        let length = file
            .metadata()
            .map_err(|error| Self::file_error(path, error))?
            .len();
        let length = usize::try_from(length).map_err(|error| Self::parse_error(path, error))?;
        if length > self.limits.document_bytes {
            return Err(Self::parse_error(
                path,
                format!(
                    "document bytes exceed limit: limit {}, actual {length}",
                    self.limits.document_bytes
                ),
            ));
        }
        let bytes = consus_io::read_exact_bounded(&mut file, length)
            .map_err(|error| Self::file_error(path, error))?;
        let parsed = consus_onnx::parse_model(&bytes, self.limits)
            .map_err(|error| Self::parse_error(path, error))?;
        let graph = Self::map_graph(&parsed)?;
        graph
            .validate()
            .map_err(|message| OnnxError::InvalidModel { message })?;

        Ok(OnnxDocument {
            graph,
            metadata: Self::map_metadata(&parsed) })
    }

    fn map_graph(document: &ModelDocument<'_>) -> OnnxResult<OnnxGraph> {
        let mut graph = OnnxGraph::new(document.graph_name.to_owned());
        let initializer_names: HashSet<_> = document
            .initializers
            .iter()
            .map(|initializer| initializer.name)
            .collect();
        graph.inputs = document
            .inputs
            .iter()
            .filter(|value| !initializer_names.contains(value.name))
            .map(Self::map_value)
            .collect::<OnnxResult<_>>()?;
        graph.outputs = document
            .outputs
            .iter()
            .map(Self::map_value)
            .collect::<OnnxResult<_>>()?;
        graph.nodes = document
            .nodes
            .iter()
            .map(|node| crate::onnx::OnnxNode {
                name: node.name.to_owned(),
                op_type: node.operation.to_owned(),
                domain: node.domain.to_owned(),
                inputs: node.inputs.iter().map(|name| (*name).to_owned()).collect(),
                outputs: node.outputs.iter().map(|name| (*name).to_owned()).collect(),
                attributes: HashMap::new(),
                doc_string: None })
            .collect();
        for initializer in &document.initializers {
            let value = crate::onnx::OnnxValue::new(
                initializer.name.to_owned(),
                Self::map_element_type(initializer.element_type)?,
                initializer
                    .dimensions
                    .iter()
                    .map(|dimension| i64::try_from(*dimension))
                    .collect::<Result<_, _>>()
                    .map_err(|error| Self::model_error(initializer.name, error))?,
            );
            graph
                .initializers
                .insert(initializer.name.to_owned(), value);
        }
        Ok(graph)
    }

    fn map_value(value: &ValueInfo<'_>) -> OnnxResult<crate::onnx::OnnxValue> {
        let TensorInfo {
            element_type,
            dimensions } = value
            .tensor
            .as_ref()
            .ok_or_else(|| OnnxError::InvalidModel {
                message: format!("tensor metadata is absent for value '{}'", value.name) })?;
        let shape = dimensions
            .iter()
            .map(|dimension| match dimension {
                Some(dimension) => {
                    i64::try_from(*dimension).map_err(|error| Self::model_error(value.name, error))
                }
                None => Ok(-1) })
            .collect::<OnnxResult<_>>()?;
        Ok(crate::onnx::OnnxValue::new(
            value.name.to_owned(),
            Self::map_element_type(*element_type)?,
            shape,
        ))
    }

    fn map_element_type(value: ElementType) -> OnnxResult<crate::onnx::OnnxElementType> {
        use crate::onnx::OnnxElementType as Target;
        Ok(match value {
            ElementType::Float => Target::Float,
            ElementType::Uint8 => Target::Uint8,
            ElementType::Int8 => Target::Int8,
            ElementType::Uint16 => Target::Uint16,
            ElementType::Int16 => Target::Int16,
            ElementType::Int32 => Target::Int32,
            ElementType::Int64 => Target::Int64,
            ElementType::String => Target::String,
            ElementType::Bool => Target::Bool,
            ElementType::Float16 => Target::Float16,
            ElementType::Double => Target::Float64,
            ElementType::Uint32 => Target::Uint32,
            ElementType::Uint64 => Target::Uint64,
            ElementType::Complex64 => Target::Complex64,
            ElementType::Complex128 => Target::Complex128,
            ElementType::Bfloat16 => Target::Bfloat16,
            ElementType::Unknown(code) => {
                return Err(OnnxError::InvalidModel {
                    message: format!("unsupported tensor element type code {code}") });
            }
            _ => {
                return Err(OnnxError::InvalidModel {
                    message: "unsupported future tensor element type".to_owned() });
            }
        })
    }

    fn map_metadata(document: &ModelDocument<'_>) -> OnnxMetadata {
        let opset_version = document
            .operator_sets
            .iter()
            .filter(|opset| opset.domain.is_empty() || opset.domain == "ai.onnx")
            .map(|opset| opset.version)
            .max()
            .unwrap_or_default();
        OnnxMetadata {
            ir_version: document.ir_version,
            opset_version,
            producer_name: (!document.producer_name.is_empty())
                .then(|| document.producer_name.to_owned()),
            producer_version: (!document.producer_version.is_empty())
                .then(|| document.producer_version.to_owned()),
            domain: (!document.domain.is_empty()).then(|| document.domain.to_owned()),
            model_version: Some(document.model_version),
            metadata_props: HashMap::new() }
    }

    fn file_error(path: &Path, error: std::io::Error) -> OnnxError {
        Self::parse_error(path, error)
    }

    fn parse_error(path: &Path, error: impl std::fmt::Display) -> OnnxError {
        OnnxError::ProtobufParseError {
            message: format!("failed to parse ONNX file '{}': {error}", path.display()) }
    }

    fn model_error(name: &str, error: impl std::fmt::Display) -> OnnxError {
        OnnxError::InvalidModel {
            message: format!("tensor '{name}' has an unrepresentable dimension: {error}") }
    }
}

impl Default for OnnxParser {
    fn default() -> Self {
        Self::new()
    }
}

/// A validated ONNX graph and its model metadata.
#[derive(Debug)]
pub struct OnnxDocument {
    graph: OnnxGraph,
    metadata: OnnxMetadata }

impl OnnxDocument {
    /// Get the model metadata.
    #[must_use]
    pub const fn metadata(&self) -> &OnnxMetadata {
        &self.metadata
    }

    /// Get the computation graph.
    #[must_use]
    pub const fn graph(&self) -> &OnnxGraph {
        &self.graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_path() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("test_data")
            .join("basic_model.onnx")
    }

    #[test]
    fn basic_model_maps_exact_graph_semantics() {
        let document = OnnxParser::new()
            .parse(fixture_path())
            .expect("public parser should accept the committed ONNX fixture");
        let graph = document.graph();

        assert_eq!(
            graph
                .inputs
                .iter()
                .map(|value| value.name.as_str())
                .collect::<Vec<_>>(),
            ["input"]
        );
        assert_eq!(
            graph
                .outputs
                .iter()
                .map(|value| value.name.as_str())
                .collect::<Vec<_>>(),
            ["output"]
        );
        assert_eq!(
            graph
                .nodes
                .iter()
                .map(|node| node.op_type.as_str())
                .collect::<Vec<_>>(),
            ["Relu", "PRelu", "Add"]
        );
        let mut initializer_names = graph
            .initializers
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>();
        initializer_names.sort_unstable();
        assert_eq!(initializer_names, ["add_bias", "prelu_slope"]);
        assert_eq!(graph.inputs[0].value_info.shape, [1, 3, 4, 4]);
        assert_eq!(graph.outputs[0].value_info.shape, [1, 3, 4, 4]);
        assert_eq!(
            graph.initializers["prelu_slope"].value_info.shape,
            [3, 1, 1]
        );
        assert_eq!(
            graph.initializers["add_bias"].value_info.shape,
            [1, 3, 1, 1]
        );
        assert_eq!(document.metadata().ir_version, 12);
        assert_eq!(document.metadata().opset_version, 16);
    }

    #[test]
    fn missing_file_reports_path_context() {
        let path = std::path::Path::new("/nonexistent/path/model.onnx");
        let error = OnnxParser::new().parse(path).unwrap_err();
        assert!(error.to_string().contains("/nonexistent/path/model.onnx"));
    }

    #[test]
    fn document_limit_rejects_fixture_before_allocation() {
        let limits = ParseLimits {
            document_bytes: 1,
            ..ParseLimits::default()
        };
        let error = OnnxParser::with_limits(limits)
            .parse(fixture_path())
            .unwrap_err();
        assert!(error.to_string().contains("document bytes exceed limit"));
    }
}
