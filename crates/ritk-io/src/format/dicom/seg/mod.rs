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
//! For frame f with rows×cols pixels, frame_bytes = ⌈rows×cols / 8⌉.
//! The flat pixel index i ∈ [0, rows×cols) maps to:
//!   byte   = i / 8
//!   bit    = 7 - (i % 8)   (MSB-first within each byte)
//!   value  = (raw_byte >> bit) & 1
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
