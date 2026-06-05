//! Mutually-recursive DICOM value, sequence, and node types.
//!
//! These three types form a closed recursive family:
//! - `DicomValue::Sequence` holds `Vec<DicomSequenceItem>`
//! - `DicomSequenceItem::elements` holds `Vec<DicomObjectNode>`
//! - `DicomObjectNode::value` holds `DicomValue`
//!
//! They are co-located to satisfy Rust's type system without indirection.

use arrayvec::ArrayString;
use super::tag::{is_private_tag, DicomTag};

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
        matches!(self, DicomValue::Sequence(_))
    }

    /// Return true when the value stores raw bytes.
    pub fn is_bytes(&self) -> bool {
        matches!(self, DicomValue::Bytes(_))
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
    pub vr: Option<ArrayString<2>>,
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
    pub fn text(tag: DicomTag, vr: &str, value: impl Into<String>) -> Self {
        Self {
            tag,
            vr: Some(ArrayString::<2>::try_from(vr).unwrap_or_default()),
            value: DicomValue::Text(value.into()),
            private: is_private_tag(tag),
            source: None,
        }
    }

    /// Create a raw-byte node.
    #[inline]
    pub fn bytes(tag: DicomTag, vr: &str, value: Vec<u8>) -> Self {
        Self {
            tag,
            vr: Some(ArrayString::<2>::try_from(vr).unwrap_or_default()),
            value: DicomValue::Bytes(value),
            private: is_private_tag(tag),
            source: None,
        }
    }

    /// Create a sequence node.
    #[inline]
    pub fn sequence(tag: DicomTag, vr: &str, items: Vec<DicomSequenceItem>) -> Self {
        Self {
            tag,
            vr: Some(ArrayString::<2>::try_from(vr).unwrap_or_default()),
            value: DicomValue::Sequence(items),
            private: is_private_tag(tag),
            source: None,
        }
    }

    /// Create a numeric node from a 16-bit unsigned value.
    #[inline]
    pub fn u16(tag: DicomTag, vr: &str, value: u16) -> Self {
        Self {
            tag,
            vr: Some(ArrayString::<2>::try_from(vr).unwrap_or_default()),
            value: DicomValue::U16(value),
            private: is_private_tag(tag),
            source: None,
        }
    }

    /// Create a numeric node from a signed 32-bit value.
    #[inline]
    pub fn i32(tag: DicomTag, vr: &str, value: i32) -> Self {
        Self {
            tag,
            vr: Some(ArrayString::<2>::try_from(vr).unwrap_or_default()),
            value: DicomValue::I32(value),
            private: is_private_tag(tag),
            source: None,
        }
    }

    /// Create a numeric node from a 64-bit float.
    #[inline]
    pub fn f64(tag: DicomTag, vr: &str, value: f64) -> Self {
        Self {
            tag,
            vr: Some(ArrayString::<2>::try_from(vr).unwrap_or_default()),
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
