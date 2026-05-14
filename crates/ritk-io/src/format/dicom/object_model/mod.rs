//! Recursive DICOM object model.
//!
//! This module defines a lossless, typed representation for DICOM metadata
//! elements that can be preserved across series-oriented read/write passes.
//! It is intentionally conservative: it models scalar values, byte payloads,
//! nested sequences, and unknown elements without attempting to reinterpret
//! vendor-specific semantics.
//!
//! ## Invariants
//!
//! - A node's tag uniquely identifies the element within its container.
//! - Sequence items preserve ordering.
//! - Raw byte payloads remain byte-for-byte stable when round-tripped.
//!
//! ## Module hierarchy
//!
//! ```text
//! object_model/
//! ├── tag.rs          — DicomTag, is_private_tag
//! ├── types.rs        — DicomValue, DicomSequenceItem, DicomObjectNode (co-located: mutual recursion)
//! ├── model.rs        — DicomObjectModel
//! └── preservation.rs — DicomPreservedElement, DicomPreservationSet
//! ```

mod model;
mod preservation;
mod tag;
mod types;

pub use model::DicomObjectModel;
pub use preservation::{DicomPreservationSet, DicomPreservedElement};
pub use tag::{is_private_tag, DicomTag};
pub use types::{DicomObjectNode, DicomSequenceItem, DicomValue};

#[cfg(test)]
mod tests;
