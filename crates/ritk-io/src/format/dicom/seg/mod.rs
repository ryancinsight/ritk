//! DICOM Segmentation Storage (SOP Class 1.2.840.10008.5.1.4.1.1.66.4) reader/writer.
//!
//! # Specification
//!
//! A DICOM-SEG file encodes N binary or fractional segmentation frames:
//! - (0028,0010) Rows, (0028,0011) Columns, (0028,0008) NumberOfFrames
//! - (0028,0100) BitsAllocated: 1 (BINARY) or 8 (FRACTIONAL)
//! - (0062,0001) SegmentationType: "BINARY" | "FRACTIONAL"
//! - (0062,0002) SegmentSequence: one item per segment label
//! - (5200,9230) Per-Frame Functional Groups: segment identification and plane position
//! - (5200,9229) Shared Functional Groups: orientation and pixel measures
//! - (7FE0,0010) PixelData: packed bits for BINARY, byte-per-pixel for FRACTIONAL
//!
//! ## BINARY pixel unpacking invariant
//!
//! For frame f, the stream index is `f * rows * cols + i`.
//! That index maps to:
//!   byte   = index / 8
//!   bit    = index % 8   (least significant bit first)
//!   value  = (raw_byte >> bit) & 1
//! Frames have no individual padding. This replaces the previous MSB-first,
//! byte-aligned interpretation, which contradicted
//! [PS3.5 D.1](https://dicom.nema.org/medical/dicom/current/output/chtml/part05/chapter_D.html)
//! and section 8.1.1. The asymmetric nine-sample test pins the specified bytes.
//!
//! FRACTIONAL frames: pixel i of frame f = raw_bytes[f * rows*cols + i].

mod converters;
mod reader;
mod types;
mod writer;

pub use converters::{dicom_seg_to_label_map, label_map_to_dicom_seg, SegEncoding};
pub use reader::read_dicom_seg;
pub use types::{DicomSegmentInfo, DicomSegmentation, SegmentAlgorithmType, SegmentationType};
pub use writer::write_dicom_seg;

#[cfg(test)]
mod tests;
