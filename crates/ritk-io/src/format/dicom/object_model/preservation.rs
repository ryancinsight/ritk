//! DicomPreservedElement and DicomPreservationSet.

use super::model::DicomObjectModel;
use super::tag::DicomTag;
use arrayvec::ArrayString;

/// A shallow preservation record for unsupported elements.
///
/// This is intended as a bridge type when parsing a DICOM object with tags
/// that the series-oriented reader does not yet interpret semantically.
#[derive(Debug, Clone, PartialEq)]
pub struct DicomPreservedElement {
    /// Tag of the preserved element.
    pub tag: DicomTag,
    /// Raw VR if known.
    pub vr: Option<ArrayString<2>>,
    /// Raw bytes for lossless retention.
    pub bytes: Vec<u8>,
}

impl DicomPreservedElement {
    /// Create a new preserved element.
    #[inline]
    pub fn new(tag: DicomTag, vr: Option<ArrayString<2>>, bytes: Vec<u8>) -> Self {
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
