//! Native codec implementations.
//!
//! All implementations live in `ritk-codecs` — the canonical SSOT crate.
//! This module re-exports the public codec API under the same paths that
//! existing `ritk-dicom` callers already use, preserving binary compatibility.
pub use ritk_codecs::{
    decode_jpeg2000_fragment, decode_jpeg_fragment, decode_jpeg_ls_fragment,
    decode_rle_lossless_fragment, packbits_decode,
};
