//! Recursive DICOM object model.
//!
//! This module defines a lossless, typed representation for DICOM metadata
//! elements that can be preserved across series-oriented read/write passes.
//! It is intentionally conservative: it models scalar values, byte payloads,
//! nested sequences, and unknown elements without attempting to reinterpret
//! vendor-specific semantics.
//!
//! ## Stage 1 scope
//!
//! - Preserve unknown elements as typed data nodes
//! - Preserve private tags
//! - Preserve nested sequence item structure
//! - Provide a canonical in-memory object model for later writer integration
//!
//! ## Invariants
//!
//! - A node's tag uniquely identifies the element within its container.
//! - Sequence items preserve ordering.
//! - Raw byte payloads remain byte-for-byte stable when round-tripped.
//! - Scalar values may be rendered to text, but raw representation is retained
//!   when available.
//!
//! The model is designed to be used as the SSOT for DICOM object preservation
//! inside `ritk-io::format::dicom`.

use std::collections::BTreeMap;
use std::fmt;
use std::path::PathBuf;

/// DICOM tag in `(group, element)` form.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DicomTag {
    /// Group number.
    pub group: u16,
    /// Element number.
    pub element: u16,
}

impl DicomTag {
    /// Create a new DICOM tag.
    #[inline]
    pub const fn new(group: u16, element: u16) -> Self {
        Self { group, element }
    }

    /// Return the canonical `(gggg,eeee)` text form.
    #[inline]
    pub fn canonical(self) -> String {
        format!("{:04X},{:04X}", self.group, self.element)
    }
}

impl fmt::Display for DicomTag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:04X},{:04X})", self.group, self.element)
    }
}

/// DICOM value multiplicity container.
#[derive(Debug, Clone, PartialEq)]
pub enum DicomValue {
    /// UTF-8 / text scalar value.
    Text(String),
    /// Raw byte payload.
    Bytes(Vec<u8>),
    /// Unsigned integer scalar.
    U16(u16),
    /// Signed integer scalar.
    I32(i32),
    /// Floating-point scalar.
    F64(f64),
    /// Sequence of nested items.
    Sequence(Vec<DicomSequenceItem>),
    /// Empty value.
    Empty,
}

impl DicomValue {
    /// Render a stable textual form when one exists.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(value) => Some(value.as_str()),
            _ => None,
        }
    }

    /// Return true when the value stores nested sequence items.
    pub fn is_sequence(&self) -> bool {
        match self {
            DicomValue::Sequence(_) => true,
            _ => false,
        }
    }

    /// Return true when the value stores raw bytes.
    pub fn is_bytes(&self) -> bool {
        match self {
            DicomValue::Bytes(_) => true,
            _ => false,
        }
    }
}

/// A single DICOM sequence item.
#[derive(Debug, Clone, PartialEq)]
pub struct DicomSequenceItem {
    /// Ordered elements contained in this item.
    pub elements: Vec<DicomObjectNode>,
}

impl DicomSequenceItem {
    /// Create an empty sequence item.
    #[inline]
    pub fn new() -> Self {
        Self {
            elements: Vec::new(),
        }
    }

    /// Create a sequence item from a set of elements.
    #[inline]
    pub fn from_elements(elements: Vec<DicomObjectNode>) -> Self {
        Self { elements }
    }

    /// Insert or replace an element.
    pub fn insert(&mut self, node: DicomObjectNode) {
        if let Some(existing) = self
            .elements
            .iter_mut()
            .find(|existing| existing.tag() == node.tag())
        {
            *existing = node;
        } else {
            self.elements.push(node);
        }
    }

    /// Get an element by tag.
    pub fn get(&self, tag: DicomTag) -> Option<&DicomObjectNode> {
        self.elements.iter().find(|node| node.tag() == tag)
    }

    /// Get a mutable element by tag.
    pub fn get_mut(&mut self, tag: DicomTag) -> Option<&mut DicomObjectNode> {
        self.elements.iter_mut().find(|node| node.tag() == tag)
    }

    /// Return true when the sequence item contains no elements.
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    /// Number of elements in the item.
    pub fn len(&self) -> usize {
        self.elements.len()
    }
}

impl Default for DicomSequenceItem {
    fn default() -> Self {
        Self::new()
    }
}

/// A recursive DICOM object node.
#[derive(Debug, Clone, PartialEq)]
pub struct DicomObjectNode {
    /// Element tag.
    pub tag: DicomTag,
    /// Value representation string, when known.
    pub vr: Option<String>,
    /// Stored value.
    pub value: DicomValue,
    /// True when the element is private.
    pub private: bool,
    /// Free-form source note for preservation/debugging.
    pub source: Option<String>,
}

impl DicomObjectNode {
    /// Create a text node.
    #[inline]
    pub fn text(tag: DicomTag, vr: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            tag,
            vr: Some(vr.into()),
            value: DicomValue::Text(value.into()),
            private: is_private_tag(tag),
            source: None,
        }
    }

    /// Create a raw-byte node.
    #[inline]
    pub fn bytes(tag: DicomTag, vr: impl Into<String>, value: Vec<u8>) -> Self {
        Self {
            tag,
            vr: Some(vr.into()),
            value: DicomValue::Bytes(value),
            private: is_private_tag(tag),
            source: None,
        }
    }

    /// Create a sequence node.
    #[inline]
    pub fn sequence(tag: DicomTag, vr: impl Into<String>, items: Vec<DicomSequenceItem>) -> Self {
        Self {
            tag,
            vr: Some(vr.into()),
            value: DicomValue::Sequence(items),
            private: is_private_tag(tag),
            source: None,
        }
    }

    /// Create a numeric node from a 16-bit unsigned value.
    #[inline]
    pub fn u16(tag: DicomTag, vr: impl Into<String>, value: u16) -> Self {
        Self {
            tag,
            vr: Some(vr.into()),
            value: DicomValue::U16(value),
            private: is_private_tag(tag),
            source: None,
        }
    }

    /// Create a numeric node from a signed 32-bit value.
    #[inline]
    pub fn i32(tag: DicomTag, vr: impl Into<String>, value: i32) -> Self {
        Self {
            tag,
            vr: Some(vr.into()),
            value: DicomValue::I32(value),
            private: is_private_tag(tag),
            source: None,
        }
    }

    /// Create a numeric node from a 64-bit float.
    #[inline]
    pub fn f64(tag: DicomTag, vr: impl Into<String>, value: f64) -> Self {
        Self {
            tag,
            vr: Some(vr.into()),
            value: DicomValue::F64(value),
            private: is_private_tag(tag),
            source: None,
        }
    }

    /// Return the node tag.
    #[inline]
    pub fn tag(&self) -> DicomTag {
        self.tag
    }

    /// Return true when the node is private.
    #[inline]
    pub fn is_private(&self) -> bool {
        self.private
    }

    /// Return true when the node stores nested items.
    #[inline]
    pub fn is_sequence(&self) -> bool {
        self.value.is_sequence()
    }

    /// Attach a source note.
    #[inline]
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }
}

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

/// Determine whether a tag is private.
///
/// In DICOM, private tags occupy odd-numbered groups.
#[inline]
pub const fn is_private_tag(tag: DicomTag) -> bool {
    tag.group % 2 == 1
}

/// A shallow preservation record for unsupported elements.
///
/// This is intended as a bridge type when parsing a DICOM object with tags
/// that the series-oriented reader does not yet interpret semantically.
#[derive(Debug, Clone, PartialEq)]
pub struct DicomPreservedElement {
    /// Tag of the preserved element.
    pub tag: DicomTag,
    /// Raw VR if known.
    pub vr: Option<String>,
    /// Raw bytes for lossless retention.
    pub bytes: Vec<u8>,
}

impl DicomPreservedElement {
    /// Create a new preserved element.
    #[inline]
    pub fn new(tag: DicomTag, vr: Option<String>, bytes: Vec<u8>) -> Self {
        Self { tag, vr, bytes }
    }
}

/// A container for object-model preservation data.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DicomPreservationSet {
    /// Supported scalar nodes.
    pub object: DicomObjectModel,
    /// Unsupported or raw-retained elements.
    pub preserved: Vec<DicomPreservedElement>,
}

impl DicomPreservationSet {
    /// Create an empty preservation set.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a preserved raw element.
    pub fn preserve(&mut self, element: DicomPreservedElement) {
        self.preserved.push(element);
    }

    /// True when the set contains no preserved content.
    pub fn is_empty(&self) -> bool {
        self.object.is_empty() && self.preserved.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn private_tag_detection_uses_odd_groups() {
        assert!(is_private_tag(DicomTag::new(0x0019, 0x10AA)));
        assert!(is_private_tag(DicomTag::new(0x0029, 0x0010)));
        assert!(!is_private_tag(DicomTag::new(0x0028, 0x0010)));
    }

    #[test]
    fn canonical_tag_format_is_stable() {
        assert_eq!(DicomTag::new(0x0019, 0x10AA).canonical(), "0019,10AA");
        assert_eq!(DicomTag::new(0x7FE0, 0x0010).canonical(), "7FE0,0010");
    }

    #[test]
    fn sequence_items_preserve_order_and_replace_by_tag() {
        let mut item = DicomSequenceItem::new();
        item.insert(DicomObjectNode::text(
            DicomTag::new(0x0010, 0x0010),
            "PN",
            "Test^Patient",
        ));
        item.insert(DicomObjectNode::text(
            DicomTag::new(0x0010, 0x0020),
            "LO",
            "PAT001",
        ));
        item.insert(DicomObjectNode::text(
            DicomTag::new(0x0010, 0x0020),
            "LO",
            "PAT002",
        ));

        assert_eq!(item.len(), 2);
        assert_eq!(
            item.get(DicomTag::new(0x0010, 0x0020))
                .and_then(|n| n.value.as_text()),
            Some("PAT002")
        );
        assert_eq!(item.elements[0].tag(), DicomTag::new(0x0010, 0x0010));
    }

    #[test]
    fn object_model_replaces_nodes_by_tag() {
        let mut model = DicomObjectModel::new();
        model.insert(DicomObjectNode::text(
            DicomTag::new(0x0008, 0x0060),
            "CS",
            "OT",
        ));
        model.insert(DicomObjectNode::text(
            DicomTag::new(0x0008, 0x0060),
            "CS",
            "CT",
        ));
        assert_eq!(model.len(), 1);
        assert_eq!(
            model
                .get(DicomTag::new(0x0008, 0x0060))
                .and_then(|n| n.value.as_text()),
            Some("CT")
        );
    }

    #[test]
    fn preservation_set_tracks_object_and_raw_elements() {
        let mut set = DicomPreservationSet::new();
        set.object.insert(DicomObjectNode::text(
            DicomTag::new(0x0008, 0x103E),
            "LO",
            "Series Description",
        ));
        set.preserve(DicomPreservedElement::new(
            DicomTag::new(0x0009, 0x1001),
            Some("OB".to_string()),
            vec![1, 2, 3, 4],
        ));

        assert!(!set.is_empty());
        assert_eq!(set.object.len(), 1);
        assert_eq!(set.preserved.len(), 1);
        assert_eq!(set.preserved[0].bytes, vec![1, 2, 3, 4]);
    }
}
