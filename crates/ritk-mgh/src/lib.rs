//! MGH / MGZ (FreeSurfer) reader and writer for 3-D medical images.
//!
//! # Format
//!
//! MGH (Massachusetts General Hospital) is FreeSurfer's native volumetric
//! image format.  The file consists of a 284-byte big-endian header followed
//! by voxel data in Fortran (column-major) order where the X axis varies
//! fastest, then Y, then Z.
//!
//! MGZ is the gzip-compressed variant. File extensions `.mgz` and `.mgh.gz`,
//! matched without ASCII case sensitivity, trigger gzip decompression on read
//! and gzip compression on write.
//!
//! # Frames
//!
//! The header's `nframes` field counts consecutive volumes of identical
//! geometry: one for an anatomical volume, many for a diffusion or time series.
//!
//! - [`read_mgh`] returns a single 3-D image and rejects multi-frame files.
//! - [`read_mgh_series`] returns one `Image` per frame, accepting both
//!   single-frame and multi-frame files.
//! - [`write_mgh`] writes a single frame (`nframes = 1`).
//! - [`write_mgh_series`] writes every image in the slice as one frame,
//!   preserving the shared spatial grid.
//!
//! # Spatial metadata
//!
//! When `goodRASFlag == 1` in the header, the reader extracts direction
//! cosines, voxel spacing, and the RAS center of the volume, converting
//! them to RITK's `origin` / `spacing` / `direction` representation via:
//!
//! ```text
//! Mdc    = [x_ras, y_ras, z_ras]          (3×3 direction cosine matrix)
//! D      = diag(spacing_x, spacing_y, spacing_z)
//! h      = [(width−1)/2, (height−1)/2, (depth−1)/2]^T
//! origin = c_ras − Mdc · D · h
//! ```
//!
//! When the flag is unset, default spatial metadata is used (identity
//! direction, unit spacing, zero origin).
//!
//! # Pixel types
//!
//! The reader handles all four MGH data types (`MRI_UCHAR` u8,
//! `MRI_SHORT` i16, `MRI_INT` i32, `MRI_FLOAT` f32), converting all
//! to f32 for the RITK tensor.  The writer always emits `MRI_FLOAT`.

mod binary;
mod reader;
mod spatial;
#[cfg(test)]
mod test_support;
mod types;
mod writer;

pub use reader::{read_mgh, read_mgh_series, MghReader};
pub use writer::{write_mgh, write_mgh_series, MghWriter};

use std::path::Path;

/// MGH header size in bytes.
const HEADER_SIZE: usize = 284;

/// Valid MGH format version number.
const VERSION: i32 = 1;

/// The `nframes` header value of a single-volume MGH file.
///
/// MGH stores `nframes` consecutive volumes of identical geometry — one for a
/// plain anatomical volume, many for a diffusion or time series. The reader and
/// writer in this crate carry the 3-D `Image` contract, which represents exactly
/// one frame, so this is the only value they read or emit.
const SINGLE_FRAME: i32 = 1;

/// Byte offset where the spatial metadata block (goodRASFlag through c_ras)
/// ends inside the 284-byte header.  Everything from this offset to byte 284
/// is zero-padding.
const SPATIAL_BLOCK_END: usize = 90;

/// Padding length in bytes: `HEADER_SIZE - SPATIAL_BLOCK_END`.
const PADDING_LEN: usize = HEADER_SIZE - SPATIAL_BLOCK_END;

/// `goodRASFlag` value marking the header's direction cosines, spacing, and
/// `c_ras` center as authoritative. Any other value means the spatial block is
/// unset and the geometry is synthesized.
const GOOD_RAS_VALID: i16 = 1;

/// `dof` header value for volumes carrying no degrees-of-freedom annotation.
///
/// The field records the DOF of a statistical map; a plain image has none. The
/// reader ignores it, so the writer emits the unset value.
const DOF_UNSET: i32 = 0;

/// MGH data type code: unsigned 8-bit integer.
const MRI_UCHAR: i32 = 0;

/// MGH data type code: signed 32-bit integer.
const MRI_INT: i32 = 1;

/// MGH data type code: 32-bit IEEE 754 float.
const MRI_FLOAT: i32 = 3;

/// MGH data type code: signed 16-bit integer.
const MRI_SHORT: i32 = 4;

/// Returns `true` when the file path extension indicates gzip compression.
///
/// Recognized patterns, without ASCII case sensitivity: `.mgz`, `.mgh.gz`.
fn is_gzip_path(path: &Path) -> bool {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    [".mgz", ".mgh.gz"].iter().any(|suffix| {
        name.as_bytes()
            .get(name.len().saturating_sub(suffix.len())..)
            .is_some_and(|tail| tail.eq_ignore_ascii_case(suffix.as_bytes()))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_gzip_detection_mgz() {
        assert!(is_gzip_path(Path::new("brain.mgz")));
    }

    #[test]
    fn test_gzip_detection_mgh_gz() {
        assert!(is_gzip_path(Path::new("brain.mgh.gz")));
        assert!(is_gzip_path(Path::new("BRAIN.MGH.GZ")));
        assert!(is_gzip_path(Path::new("BRAIN.MGZ")));
    }

    #[test]
    fn test_gzip_detection_plain_mgh() {
        assert!(!is_gzip_path(Path::new("brain.mgh")));
    }

    #[test]
    fn test_header_constants_consistency() {
        assert_eq!(SPATIAL_BLOCK_END + PADDING_LEN, HEADER_SIZE);
        assert_eq!(HEADER_SIZE, 284);
    }
}
