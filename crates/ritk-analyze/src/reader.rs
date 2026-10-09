//! Analyze 7.5 reader — parses a 348-byte `.hdr` header and raw `.img` voxel data.
//!
//! # Format Overview
//!
//! Analyze 7.5 (Mayo Clinic, 1989) stores a 3-D volume as two files sharing the
//! same base name:
//!
//! * `<name>.hdr` — 348-byte binary header (little-endian).
//! * `<name>.img` — raw voxel values (little-endian, type given by `datatype` field).
//!
//! A paired NIfTI-1 dataset can use the same extensions, but identifies itself
//! with `ni1\0` at bytes 344–347 and is not an Analyze 7.5 file. This reader
//! rejects that variant explicitly instead of interpreting NIfTI spatial fields
//! as Analyze history fields.
//!
//! # Header Layout
//!
//! The 348-byte field map is declared once in [`crate::header`], which this
//! reader, the writer, and the stored-sample conversion all share.
//!
//! # Axis Convention
//!
//! Analyze stores voxels with X varying fastest (column-major XYZ).
//! RITK stores tensors with shape `[nz, ny, nx]` (Z-major ZYX).
//! Because both produce the same flat byte sequence for identical (nx, ny, nz),
//! no in-memory permutation is required.
//!
//! # Spatial Metadata
//!
//! The file stores spacing in file-axis order `pixdim[1..3] = [sx, sy, sz]`.
//! RITK's core `Spacing` is per tensor axis `[z, y, x]` (matching the `[nz, ny,
//! nx]` tensor shape), so the file components are reversed to `[sz, sy, sx]` on
//! read — the same column reorder the MetaImage/NRRD readers apply. The core
//! `origin` is a world-space point `[x, y, z]` and is **not** reversed.
//!
//! The physical origin is reconstructed from `originator` voxel coordinates:
//!
//! ```text
//!   origin_x = originator[0] × sx
//!   origin_y = originator[1] × sy
//!   origin_z = originator[2] × sz
//! ```
//!
//! Note: the `originator` field is unreliable across writers (Analyze 7.5 is a
//! deprecated format; SimpleITK does not round-trip a physical origin through
//! it), so origin parity with foreign Analyze files is not guaranteed.

use anyhow::{anyhow, Context, Result};
use coeus_core::ComputeBackend;
use ritk_spatial::{Direction, Point, Spacing};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::mem::size_of;
use std::path::Path;

pub use crate::codec::{DT_DOUBLE, DT_FLOAT, DT_SIGNED_INT, DT_SIGNED_SHORT, DT_UNSIGNED_CHAR};
use crate::header::{self, AnalyzeDatatype, AnalyzeHeader};

const DECODE_CHUNK_BYTES: usize = 8 * 1024;

trait AnalyzeVoxel: Sized {
    fn decode(bytes: &[u8]) -> f32;
}

impl AnalyzeVoxel for u8 {
    fn decode(bytes: &[u8]) -> f32 {
        f32::from(bytes[0])
    }
}

impl AnalyzeVoxel for i16 {
    fn decode(bytes: &[u8]) -> f32 {
        f32::from(i16::from_le_bytes(
            bytes
                .try_into()
                .expect("invariant: i16 Analyze chunks contain two bytes"),
        ))
    }
}

impl AnalyzeVoxel for i32 {
    fn decode(bytes: &[u8]) -> f32 {
        i32::from_le_bytes(
            bytes
                .try_into()
                .expect("invariant: i32 Analyze chunks contain four bytes"),
        ) as f32
    }
}

impl AnalyzeVoxel for f32 {
    fn decode(bytes: &[u8]) -> f32 {
        Self::from_le_bytes(
            bytes
                .try_into()
                .expect("invariant: f32 Analyze chunks contain four bytes"),
        )
    }
}

impl AnalyzeVoxel for f64 {
    fn decode(bytes: &[u8]) -> f32 {
        Self::from_le_bytes(
            bytes
                .try_into()
                .expect("invariant: f64 Analyze chunks contain eight bytes"),
        ) as f32
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Read a 3-D image from an Analyze 7.5 `.hdr` / `.img` file pair.
///
/// `path` may point to either the `.hdr` or the `.img` file.  The sibling file
/// is located automatically by replacing the extension.
///
/// # Supported datatypes
/// `DT_UNSIGNED_CHAR` (2), `DT_SIGNED_SHORT` (4), `DT_SIGNED_INT` (8),
/// `DT_FLOAT` (16), `DT_DOUBLE` (64).  All are converted to `f32` in the
/// returned native image buffer.
///
/// # Errors
/// Returns an error when:
/// - Either file cannot be opened or read.
/// - The header is not a little-endian, 348-byte Analyze header, including
///   paired NIfTI data using the same extensions.
/// - The header does not describe exactly one 3-D volume with positive,
///   non-overflowing dimensions.
/// - `datatype` is not supported or `bitpix` does not match it.
/// - Spacing, scale, or offset metadata is non-finite, or the offset is not a
///   supported whole-byte position.
/// - The `.img` file length differs from the exact declared payload size.
/// - Output allocation, seeking, decoding, or image construction fails.
pub fn read_analyze<B: ComputeBackend, P: AsRef<Path>>(
    path: P,
    backend: &B,
) -> Result<ritk_image::Image<f32, B, 3>> {
    let DecodedAnalyze {
        data,
        dims,
        origin,
        spacing,
        direction,
    } = decode_analyze(path)?;

    ritk_image::Image::from_flat_on(data, dims, origin, spacing, direction, backend)
}

/// Substrate-agnostic decode of an Analyze `.hdr`/`.img` pair into flat
/// `[Z, Y, X]` voxels plus spatial metadata for the public reader.
struct DecodedAnalyze {
    data: Vec<f32>,
    dims: [usize; 3],
    origin: Point<3>,
    spacing: Spacing<3>,
    direction: Direction<3>,
}

fn decode_analyze<P: AsRef<Path>>(path: P) -> Result<DecodedAnalyze> {
    let path = path.as_ref();

    // Derive sibling paths regardless of which file the caller passed.
    let header = header::parse(&path.with_extension("hdr"))?;
    let [nz, ny, nx] = header.shape;
    let voxel_count = nx
        .checked_mul(ny)
        .and_then(|plane| plane.checked_mul(nz))
        .context("Analyze voxel count overflows usize")?;
    let mut img_file = open_payload(&path.with_extension("img"), &header, voxel_count)?;
    let vals = decode_voxels(header.datatype, &mut img_file, voxel_count, header.scale)?;

    tracing::debug!(
        nx,
        ny,
        nz,
        datatype = header.datatype.code(),
        "decode_analyze: complete"
    );

    // Spacing reverses file `[sx, sy, sz]` into core tensor-axis order
    // `[sz, sy, sx]`; origin stays a world-space `[x, y, z]` point.
    Ok(DecodedAnalyze {
        data: vals,
        dims: header.shape,
        origin: header.origin,
        spacing: header.spacing,
        direction: Direction::identity(),
    })
}

/// Opens the `.img` payload and validates it against the parsed header.
///
/// The declared length is `vox_offset` plus the exact voxel bytes, so a file
/// that is short, long, or offset differently is rejected before any voxel is
/// read. The returned handle is positioned at the first payload byte.
///
/// # Errors
///
/// Returns an error when the payload cannot be opened or inspected, or when its
/// length differs from the header's declared `vox_offset` plus payload size.
pub(crate) fn open_payload(
    img_path: &Path,
    header: &AnalyzeHeader,
    voxel_count: usize,
) -> Result<File> {
    let expected_bytes = voxel_count
        .checked_mul(header.datatype.width())
        .context("Analyze payload byte count overflows usize")?;
    let expected_bytes_u64 =
        u64::try_from(expected_bytes).context("Analyze payload byte count exceeds u64")?;
    let expected_file_len = header
        .vox_offset
        .checked_add(expected_bytes_u64)
        .context("Analyze payload end offset overflows u64")?;
    let mut img_file = File::open(img_path).context("Cannot open Analyze data file")?;
    let actual_file_len = img_file
        .metadata()
        .context("Cannot inspect Analyze data file")?
        .len();
    if actual_file_len != expected_file_len {
        return Err(anyhow!(
            "Analyze .img length mismatch: expected {expected_file_len} bytes ({} offset + {expected_bytes} payload), found {actual_file_len}",
            header.vox_offset
        ));
    }
    img_file
        .seek(SeekFrom::Start(header.vox_offset))
        .context("Cannot seek to Analyze voxel payload")?;
    Ok(img_file)
}

/// Decodes stored voxels into `f32`, applying the header's intensity scale.
fn decode_voxels(
    datatype: AnalyzeDatatype,
    reader: &mut File,
    voxel_count: usize,
    scale: f32,
) -> Result<Vec<f32>> {
    match datatype {
        AnalyzeDatatype::UnsignedChar => decode_payload::<u8>(reader, voxel_count, scale),
        AnalyzeDatatype::SignedShort => decode_payload::<i16>(reader, voxel_count, scale),
        AnalyzeDatatype::SignedInt => decode_payload::<i32>(reader, voxel_count, scale),
        AnalyzeDatatype::Float => decode_payload::<f32>(reader, voxel_count, scale),
        AnalyzeDatatype::Double => decode_payload::<f64>(reader, voxel_count, scale),
    }
}

fn decode_payload<T: AnalyzeVoxel>(
    reader: &mut File,
    voxel_count: usize,
    scale: f32,
) -> Result<Vec<f32>> {
    let voxel_width = size_of::<T>();
    let voxels_per_chunk = DECODE_CHUNK_BYTES / voxel_width;
    debug_assert!(voxels_per_chunk > 0);
    let mut values = Vec::new();
    values
        .try_reserve_exact(voxel_count)
        .context("Cannot allocate Analyze output volume")?;
    let mut bytes = [0u8; DECODE_CHUNK_BYTES];
    let mut remaining = voxel_count;

    while remaining > 0 {
        let chunk_voxels = remaining.min(voxels_per_chunk);
        let chunk_bytes = chunk_voxels
            .checked_mul(voxel_width)
            .expect("invariant: decode chunk byte count fits its fixed buffer");
        let input = &mut bytes[..chunk_bytes];
        reader
            .read_exact(input)
            .context("Cannot read validated Analyze voxel payload")?;
        values.extend(
            input
                .chunks_exact(voxel_width)
                .map(|voxel| T::decode(voxel) * scale),
        );
        remaining -= chunk_voxels;
    }

    Ok(values)
}

// ── Reader wrapper type ───────────────────────────────────────────────────────

/// Read-side wrapper type implementing the `ImageReader` domain trait.
pub struct AnalyzeReader<B: ComputeBackend> {
    pub(crate) backend: B,
}

impl<B: ComputeBackend> AnalyzeReader<B> {
    /// Construct a reader bound to `backend`.
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    /// Read an Analyze image through the bound backend.
    pub fn read<P: AsRef<Path>>(&self, path: P) -> Result<ritk_image::Image<f32, B, 3>> {
        read_analyze(path, &self.backend)
    }
}
