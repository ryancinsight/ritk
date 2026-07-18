//! DicomObjectModel â€” ordered collection of DICOM object nodes.

use std::collections::BTreeMap;
use std::path::PathBuf;

use super::tag::DicomTag;
use super::types::DicomObjectNode;

/// A DICOM object consisting of ordered nodes.
#[derive(Debug, Clone, PartialEq)]
pub struct DicomObjectModel {
    /// Object identifier when available.
    pub source: Option<PathBuf>,
    /// Ordered nodes keyed by tag for replacement semantics.
    pub nodes: Vec<DicomObjectNode>,
}

impl DicomObjectModel {
    /// Create an empty object.
    #[inline]
    pub fn new() -> Self {
        Self {
            source: None,
            nodes: Vec::new(),
        }
    }

    /// Create an empty object with a source path.
    #[inline]
    pub fn with_source(source: impl Into<PathBuf>) -> Self {
        Self {
            source: Some(source.into()),
            nodes: Vec::new(),
        }
    }

    /// Insert or replace a node.
    pub fn insert(&mut self, node: DicomObjectNode) {
        if let Some(existing) = self
            .nodes
            .iter_mut()
            .find(|existing| existing.tag() == node.tag())
        {
            *existing = node;
        } else {
            self.nodes.push(node);
        }
    }

    /// Get a node by tag.
    pub fn get(&self, tag: DicomTag) -> Option<&DicomObjectNode> {
        self.nodes.iter().find(|node| node.tag() == tag)
    }

    /// Get a mutable node by tag.
    pub fn get_mut(&mut self, tag: DicomTag) -> Option<&mut DicomObjectNode> {
        self.nodes.iter_mut().find(|node| node.tag() == tag)
    }

    /// Return true when the object contains no nodes.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Number of nodes in the object.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Preserve all tags from a series metadata map as scalar text nodes.
    pub fn from_scalar_tags(tags: &BTreeMap<DicomTag, String>) -> Self {
        let mut model = Self::new();
        for (tag, value) in tags {
            model.insert(DicomObjectNode::text(*tag, "LO", value.clone()));
        }
        model
    }
}

impl Default for DicomObjectModel {
    fn default() -> Self {
        Self::new()
    }
}
