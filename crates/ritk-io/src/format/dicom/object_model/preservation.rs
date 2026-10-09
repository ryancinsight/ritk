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

/// Why a DICOM element's value could not be added to the inventory.
///
/// A parsed element is always either interpreted, retained opaquely, or
/// recorded with one of these reasons. There is no fourth outcome: the reader
/// must never discard parser data silently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DicomRetentionReason {
    /// A sequence-typed element exposed no items to walk and could not be
    /// re-encoded as raw bytes.
    SequenceItemsUnavailable,
    /// The element value could not be re-encoded to raw bytes, so neither a
    /// typed node nor an opaque byte record could be produced.
    ValueBytesUnavailable,
    /// Recursion stopped at the nesting limit; the subtree below this element
    /// is unretained.
    NestingDepthExceeded,
}

impl DicomRetentionReason {
    /// A stable, human-readable description used in conversion reports.
    #[must_use]
    pub const fn describe(self) -> &'static str {
        match self {
            Self::SequenceItemsUnavailable => "sequence value exposed no items",
            Self::ValueBytesUnavailable => "value could not be re-encoded as bytes",
            Self::NestingDepthExceeded => "sequence nesting exceeded the retention limit",
        }
    }
}

/// A scoped record of an element that was neither interpreted nor retained.
///
/// One of these is recorded wherever the reader would otherwise drop an
/// element, so a conversion preflight can reject the source instead of
/// silently producing a volume with missing metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct DicomRetentionLoss {
    /// Tag of the element whose value was dropped.
    pub tag: DicomTag,
    /// Why the value could not be retained.
    pub reason: DicomRetentionReason,
}

impl DicomRetentionLoss {
    /// Create a new retention-loss record.
    #[inline]
    #[must_use]
    pub fn new(tag: DicomTag, reason: DicomRetentionReason) -> Self {
        Self { tag, reason }
    }
}

/// A container for object-model preservation data.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DicomPreservationSet {
    /// Supported scalar nodes.
    pub object: DicomObjectModel,
    /// Unsupported or raw-retained elements.
    pub preserved: Vec<DicomPreservedElement>,
    /// Elements that could be neither interpreted nor retained.
    ///
    /// Empty for a fully accounted source; a non-empty set means the parser
    /// dropped metadata and the conversion preflight must reject or report it.
    pub losses: Vec<DicomRetentionLoss>,
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

    /// Record that `tag`'s value could not be retained for `reason`.
    pub fn record_loss(&mut self, tag: DicomTag, reason: DicomRetentionReason) {
        self.losses.push(DicomRetentionLoss::new(tag, reason));
    }

    /// True when the set contains no preserved content and no recorded loss.
    pub fn is_empty(&self) -> bool {
        self.object.is_empty() && self.preserved.is_empty() && self.losses.is_empty()
    }

    /// True when the parser dropped metadata that the set does not retain.
    #[must_use]
    pub fn has_losses(&self) -> bool {
        !self.losses.is_empty()
    }
}
