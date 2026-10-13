//! Analyze 7.5 header layout — one definition of the 348-byte field map.
//!
//! The reader, the writer, and the stored-sample conversion all parse or emit
//! the same 348-byte block. Keeping the offsets, the datatype codes, and the
//! endianness here means a field is declared once and every consumer agrees on
//! its meaning.
//!
//! # Field map (key fields)
//!
//! | Offset | Type  | Field           | Meaning                                 |
//! |--------|-------|-----------------|-----------------------------------------|
//! |      0 | i32   | `sizeof_hdr`    | Must equal 348                          |
//! |     32 | i32   | `extents`       | Must equal 16 384                       |
//! |     38 | u8    | `regular`       | `b'r'`                                  |
//! |     40 | i16   | `dim[0]`        | Number of dimensions (3 or 4)           |
//! |     42 | i16   | `dim[1..=3]`    | X, Y, Z sizes                           |
//! |     48 | i16   | `dim[4]`        | Volume count (1)                        |
//! |     70 | i16   | `datatype`      | 2=u8, 4=i16, 8=i32, 16=f32, 64=f64      |
//! |     72 | i16   | `bitpix`        | Bits per voxel                          |
//! |     76 | f32   | `pixdim[0]`     | Number of dimensions                    |
//! |     80 | f32   | `pixdim[1..=3]` | X, Y, Z spacing (mm)                    |
//! |    108 | f32   | `vox_offset`    | Byte offset to data in `.img`           |
//! |    112 | f32   | `funused1`      | Intensity scale factor (0 or 1 = no-op) |
//! |    148 | u8×80 | `descrip`       | Provenance string                       |
//! |    253 | i16×5 | `originator`    | Voxel-space origin (x, y, z, 0, 0)      |
//!
//! # Axis convention
//!
//! `pixdim[1..=3]` is file-axis `[sx, sy, sz]`, while RITK's `Spacing<3>` is
//! tensor-axis `[sz, sy, sx]`; the components are reversed in both directions.
//! `originator` holds voxel coordinates and is scaled into a world-space
//! `Point<3>` `[ox, oy, oz]` without reversal.

use anyhow::{anyhow, Context, Result};
use consus_core::{read_integer, ByteOrder};
use ritk_codecs::SampleType;
use ritk_spatial::{Point, Spacing};
use std::fs::File;
use std::io::Read;
use std::mem::size_of;
use std::path::Path;

use crate::codec::{
    read_le, write_le, DT_DOUBLE, DT_FLOAT, DT_SIGNED_INT, DT_SIGNED_SHORT, DT_UNSIGNED_CHAR,
    EXTENTS, HDR_SIZE,
};

/// Byte offset of the `descrip` provenance string inside the header block.
pub(crate) const DESCRIP_OFFSET: usize = 148;

/// Capacity of the `descrip` field in bytes.
pub(crate) const DESCRIP_LEN: usize = 80;

/// The stored voxel representation a header's `datatype` field declares.
///
/// Analyze 7.5 defines five scalar codes. Unsigned 16-bit, signed 8-bit,
/// unsigned 32-bit, and 64-bit integers have no code, so a series in one of
/// those representations is not representable rather than converted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AnalyzeDatatype {
    /// Code 2 — unsigned 8-bit.
    UnsignedChar,
    /// Code 4 — signed 16-bit.
    SignedShort,
    /// Code 8 — signed 32-bit.
    SignedInt,
    /// Code 16 — IEEE-754 single precision.
    Float,
    /// Code 64 — IEEE-754 double precision.
    Double,
}

impl AnalyzeDatatype {
    /// Maps a header `datatype` code, rejecting codes Analyze 7.5 does not define.
    pub(crate) fn parse(code: i16) -> Result<Self> {
        match code {
            DT_UNSIGNED_CHAR => Ok(Self::UnsignedChar),
            DT_SIGNED_SHORT => Ok(Self::SignedShort),
            DT_SIGNED_INT => Ok(Self::SignedInt),
            DT_FLOAT => Ok(Self::Float),
            DT_DOUBLE => Ok(Self::Double),
            other => Err(anyhow!(
                "Unsupported Analyze datatype {other}. Supported codes: 2 (u8), 4 (i16), 8 (i32), 16 (f32), 64 (f64)."
            )),
        }
    }

    /// Returns the header code this representation writes.
    pub(crate) const fn code(self) -> i16 {
        match self {
            Self::UnsignedChar => DT_UNSIGNED_CHAR,
            Self::SignedShort => DT_SIGNED_SHORT,
            Self::SignedInt => DT_SIGNED_INT,
            Self::Float => DT_FLOAT,
            Self::Double => DT_DOUBLE,
        }
    }

    /// Returns the bytes one voxel occupies in the `.img` payload.
    pub(crate) const fn width(self) -> usize {
        match self {
            Self::UnsignedChar => size_of::<u8>(),
            Self::SignedShort => size_of::<i16>(),
            Self::SignedInt => size_of::<i32>(),
            Self::Float => size_of::<f32>(),
            Self::Double => size_of::<f64>(),
        }
    }

    /// Returns the exact stored sample representation this code denotes.
    pub(crate) const fn sample_type(self) -> SampleType {
        match self {
            Self::UnsignedChar => SampleType::U8,
            Self::SignedShort => SampleType::I16,
            Self::SignedInt => SampleType::I32,
            Self::Float => SampleType::F32,
            Self::Double => SampleType::F64,
        }
    }

    /// Returns the header code for an exact stored representation, when one exists.
    pub(crate) const fn from_sample_type(sample_type: SampleType) -> Option<Self> {
        match sample_type {
            SampleType::U8 => Some(Self::UnsignedChar),
            SampleType::I16 => Some(Self::SignedShort),
            SampleType::I32 => Some(Self::SignedInt),
            SampleType::F32 => Some(Self::Float),
            SampleType::F64 => Some(Self::Double),
            _ => None,
        }
    }
}

/// A validated Analyze 7.5 header expressed in RITK's tensor-axis convention.
pub(crate) struct AnalyzeHeader {
    /// Depth, row, column shape in RITK's `[nz, ny, nx]` order.
    pub(crate) shape: [usize; 3],
    /// Stored voxel representation declared by `datatype`.
    pub(crate) datatype: AnalyzeDatatype,
    /// Per tensor axis spacing `[sz, sy, sx]`, reversed from file `[sx, sy, sz]`.
    pub(crate) spacing: Spacing<3>,
    /// World-space origin `[ox, oy, oz]`, reconstructed from `originator`.
    pub(crate) origin: Point<3>,
    /// `funused1` intensity scale, normalised so a stored zero means one.
    pub(crate) scale: f32,
    /// Byte offset of the payload inside the `.img` file.
    pub(crate) vox_offset: u64,
    /// `descrip` provenance string, trimmed of trailing NUL padding.
    pub(crate) description: Vec<u8>,
}

/// Reads and validates the 348-byte Analyze header stored at `hdr_path`.
///
/// The returned header carries no payload information beyond the geometry and
/// the datatype needed to size it; [`crate::reader::open_payload`] validates the
/// matching `.img` file against it.
///
/// # Errors
///
/// Returns an error when the header cannot be read, is not exactly 348 bytes,
/// identifies itself as paired NIfTI, is big-endian, declares a dimension
/// count, volume count, or datatype the format does not support, or carries a
/// non-finite spacing, scale, or offset.
pub(crate) fn parse(hdr_path: &Path) -> Result<AnalyzeHeader> {
    let mut hdr_file = File::open(hdr_path).context("Cannot open Analyze header")?;
    let header_len = hdr_file
        .metadata()
        .context("Cannot inspect Analyze header")?
        .len();
    if header_len < HDR_SIZE as u64 {
        return Err(anyhow!(
            "Invalid Analyze header length: expected {HDR_SIZE} bytes, found {header_len}"
        ));
    }
    let mut hdr = [0u8; HDR_SIZE];
    hdr_file
        .read_exact(&mut hdr)
        .with_context(|| "Cannot read 348-byte header".to_string())?;
    if hdr[344..348] == *b"ni1\0" {
        return Err(anyhow!(
            "Unsupported paired NIfTI-1 header (ni1 magic); use the NIfTI reader with a single-file .nii dataset"
        ));
    }
    if header_len != HDR_SIZE as u64 {
        return Err(anyhow!(
            "Invalid Analyze header length: expected {HDR_SIZE} bytes, found {header_len}"
        ));
    }

    // sizeof_hdr must be exactly 348. Identify the unsupported byte order so a
    // big-endian file is not reported as arbitrary header corruption.
    let sizeof_hdr = read_le::<i32>(&hdr, 0);
    if sizeof_hdr != HDR_SIZE as i32 {
        if read_integer::<i32>(&hdr, ByteOrder::BigEndian) == Some(HDR_SIZE as i32) {
            return Err(anyhow!(
                "Unsupported big-endian Analyze file; RITK currently accepts little-endian Analyze 7.5 only"
            ));
        }
        return Err(anyhow!(
            "Invalid Analyze file: sizeof_hdr={} (expected 348)",
            sizeof_hdr
        ));
    }

    let dimension_count = read_le::<i16>(&hdr, 40);
    if !(3..=4).contains(&dimension_count) {
        return Err(anyhow!(
            "Unsupported Analyze dimension count {dimension_count}; the RITK reader accepts one 3-D volume"
        ));
    }
    let nx = positive_dimension(read_le::<i16>(&hdr, 42), "nx")?;
    let ny = positive_dimension(read_le::<i16>(&hdr, 44), "ny")?;
    let nz = positive_dimension(read_le::<i16>(&hdr, 46), "nz")?;
    if dimension_count == 4 {
        let volume_count = read_le::<i16>(&hdr, 48);
        if volume_count != 1 {
            return Err(anyhow!(
                "Unsupported Analyze volume count {volume_count}; the RITK reader accepts exactly one 3-D volume"
            ));
        }
    }

    let datatype_code = read_le::<i16>(&hdr, 70);
    let datatype = AnalyzeDatatype::parse(datatype_code)?;
    let bytes_per_voxel = datatype.width();
    let bitpix = read_le::<i16>(&hdr, 72);
    let expected_bitpix = i16::try_from(bytes_per_voxel * 8)
        .expect("invariant: supported Analyze voxel widths fit in i16 bits");
    if bitpix != expected_bitpix {
        return Err(anyhow!(
            "Analyze bitpix {bitpix} does not match datatype {datatype_code}; expected {expected_bitpix}"
        ));
    }

    let sx_raw = f64::from(finite_header_value(read_le::<f32>(&hdr, 80), "pixdim[1]")?);
    let sy_raw = f64::from(finite_header_value(read_le::<f32>(&hdr, 84), "pixdim[2]")?);
    let sz_raw = f64::from(finite_header_value(read_le::<f32>(&hdr, 88), "pixdim[3]")?);
    // Fall back to unit spacing when the stored value is zero or negative.
    let sx = if sx_raw > 0.0 { sx_raw } else { 1.0 };
    let sy = if sy_raw > 0.0 { sy_raw } else { 1.0 };
    let sz = if sz_raw > 0.0 { sz_raw } else { 1.0 };

    let scale_raw = finite_header_value(read_le::<f32>(&hdr, 112), "funused1 scale")?;
    let scale = if scale_raw == 0.0 { 1.0_f32 } else { scale_raw };

    let vox_offset_raw = f64::from(finite_header_value(
        read_le::<f32>(&hdr, 108),
        "vox_offset",
    )?);
    if vox_offset_raw < 0.0 || vox_offset_raw.fract() != 0.0 || vox_offset_raw > u64::MAX as f64 {
        return Err(anyhow!(
            "Unsupported Analyze vox_offset {vox_offset_raw}; expected a non-negative whole-byte offset"
        ));
    }
    let vox_offset = vox_offset_raw as u64;

    let ox_vox = read_le::<i16>(&hdr, 253) as f64;
    let oy_vox = read_le::<i16>(&hdr, 255) as f64;
    let oz_vox = read_le::<i16>(&hdr, 257) as f64;

    Ok(AnalyzeHeader {
        shape: [nz, ny, nx],
        datatype,
        spacing: Spacing::new([sz, sy, sx]),
        origin: Point::new([ox_vox * sx, oy_vox * sy, oz_vox * sz]),
        scale,
        vox_offset,
        description: descrip_bytes(&hdr),
    })
}

/// Every value the Analyze 7.5 header encodes.
pub(crate) struct AnalyzeHeaderFields<'a> {
    /// Depth, row, column shape in RITK's `[nz, ny, nx]` order.
    pub(crate) shape: [usize; 3],
    /// Per tensor axis spacing `[sz, sy, sx]`.
    pub(crate) spacing: &'a Spacing<3>,
    /// World-space origin `[ox, oy, oz]`.
    pub(crate) origin: &'a Point<3>,
    /// Stored voxel representation to declare.
    pub(crate) datatype: AnalyzeDatatype,
    /// Intensity scale written to `funused1`; one means no scaling.
    pub(crate) scale: f32,
    /// Provenance string written to `descrip`, at most 80 ASCII bytes.
    pub(crate) description: &'a [u8],
}

/// Builds the 348-byte header for `fields`.
///
/// The complete logical input is validated before a byte is produced, so a
/// caller can treat a returned header as proof that the geometry, the scale,
/// and the description are representable.
///
/// # Errors
///
/// Returns an error when a dimension is zero or exceeds `i16::MAX`, a spacing
/// or origin component is non-finite or unrepresentable as a positive `f32`, an
/// origin coordinate falls outside the `originator` `i16` range, the scale is
/// non-finite or the format's no-scaling sentinel, or the description is longer
/// than the 80-byte `descrip` field.
pub(crate) fn encode(fields: &AnalyzeHeaderFields<'_>) -> Result<[u8; HDR_SIZE]> {
    let [nz, ny, nx] = fields.shape;
    for &(name, &val) in [("nx", &nx), ("ny", &ny), ("nz", &nz)].iter() {
        if val == 0 {
            anyhow::bail!("Analyze: dimension {name} must be positive");
        }
        if val > i16::MAX as usize {
            anyhow::bail!(
                "Analyze: dimension {name}={val} exceeds i16::MAX ({})",
                i16::MAX
            );
        }
    }

    // File-axis spacing [sx, sy, sz] is the reverse of core [sz, sy, sx].
    let (sx, sy, sz) = (fields.spacing[2], fields.spacing[1], fields.spacing[0]);
    let sx_header = header_spacing("x", sx)?;
    let sy_header = header_spacing("y", sy)?;
    let sz_header = header_spacing("z", sz)?;
    for (axis, value) in [
        ("x", fields.origin[0]),
        ("y", fields.origin[1]),
        ("z", fields.origin[2]),
    ] {
        if !value.is_finite() {
            anyhow::bail!("Analyze: origin[{axis}] must be finite, found {value}");
        }
    }
    if fields.description.len() > DESCRIP_LEN {
        anyhow::bail!(
            "Analyze: description is {} bytes; descrip holds at most {DESCRIP_LEN}",
            fields.description.len()
        );
    }
    let scale = header_scale(fields.scale)?;

    let mut hdr = [0u8; HDR_SIZE];
    write_le::<i32>(&mut hdr, 0, HDR_SIZE as i32); // sizeof_hdr
    write_le::<i32>(&mut hdr, 32, EXTENTS); // extents
    hdr[38] = b'r'; // regular

    write_le::<i16>(&mut hdr, 40, 4); // dim[0] = num dimensions
    write_le::<i16>(&mut hdr, 42, nx as i16); // dim[1] = X
    write_le::<i16>(&mut hdr, 44, ny as i16); // dim[2] = Y
    write_le::<i16>(&mut hdr, 46, nz as i16); // dim[3] = Z
    write_le::<i16>(&mut hdr, 48, 1); // dim[4] = time (1 volume)

    write_le::<i16>(&mut hdr, 70, fields.datatype.code());
    write_le::<i16>(&mut hdr, 72, bitpix(fields.datatype));

    write_le::<f32>(&mut hdr, 76, 4.0_f32); // pixdim[0] = number of dims
    write_le::<f32>(&mut hdr, 80, sx_header); // pixdim[1] = sx
    write_le::<f32>(&mut hdr, 84, sy_header); // pixdim[2] = sy
    write_le::<f32>(&mut hdr, 88, sz_header); // pixdim[3] = sz
    write_le::<f32>(&mut hdr, 92, 1.0_f32); // pixdim[4] = TR (unused)

    write_le::<f32>(&mut hdr, 108, 0.0_f32); // vox_offset
    write_le::<f32>(&mut hdr, 112, scale); // funused1 = scale factor

    hdr[DESCRIP_OFFSET..DESCRIP_OFFSET + fields.description.len()]
        .copy_from_slice(fields.description);

    let ox_vox = vox_coord("x", fields.origin[0], f64::from(sx_header))?;
    let oy_vox = vox_coord("y", fields.origin[1], f64::from(sy_header))?;
    let oz_vox = vox_coord("z", fields.origin[2], f64::from(sz_header))?;
    write_le::<i16>(&mut hdr, 253, ox_vox); // originator[0] = x voxel
    write_le::<i16>(&mut hdr, 255, oy_vox); // originator[1] = y voxel
    write_le::<i16>(&mut hdr, 257, oz_vox); // originator[2] = z voxel

    Ok(hdr)
}

/// Returns the `bitpix` value a datatype writes.
const fn bitpix(datatype: AnalyzeDatatype) -> i16 {
    (datatype.width() as i16) * 8
}

fn positive_dimension(raw: i16, name: &str) -> Result<usize> {
    usize::try_from(raw)
        .map_err(|_| anyhow!("Invalid Analyze dimension {name}={raw}; expected a positive value"))
        .and_then(|value| {
            if value == 0 {
                Err(anyhow!(
                    "Invalid Analyze dimension {name}=0; expected a positive value"
                ))
            } else {
                Ok(value)
            }
        })
}

fn finite_header_value(raw: f32, field: &str) -> Result<f32> {
    if raw.is_finite() {
        Ok(raw)
    } else {
        Err(anyhow!(
            "Invalid Analyze {field}: expected a finite value, found {raw}"
        ))
    }
}

/// Reads `descrip` as the writer's provenance string, dropping NUL padding.
fn descrip_bytes(hdr: &[u8; HDR_SIZE]) -> Vec<u8> {
    let field = &hdr[DESCRIP_OFFSET..DESCRIP_OFFSET + DESCRIP_LEN];
    let end = field
        .iter()
        .rposition(|byte| *byte != 0)
        .map_or(0, |last| last + 1);
    field[..end].to_vec()
}

fn header_spacing(axis: &str, spacing_mm: f64) -> Result<f32> {
    let encoded = spacing_mm as f32;
    if !encoded.is_finite() || encoded <= 0.0 {
        anyhow::bail!(
            "Analyze: spacing[{axis}]={spacing_mm} is not representable as a positive finite f32 header value"
        );
    }
    Ok(encoded)
}

/// Validates the `funused1` intensity scale.
///
/// A stored zero means "no scaling" on read, so a zero scale cannot round-trip;
/// a non-finite scale has no `f32` header form.
fn header_scale(scale: f32) -> Result<f32> {
    if !scale.is_finite() {
        anyhow::bail!("Analyze: scale {scale} must be finite");
    }
    if scale == 0.0 {
        anyhow::bail!("Analyze: scale 0 is the header's no-scaling sentinel and cannot round-trip");
    }
    Ok(scale)
}

/// Converts a physical origin coordinate to the format's rounded voxel index.
#[inline]
fn vox_coord(axis: &str, origin_mm: f64, spacing_mm: f64) -> Result<i16> {
    let voxel = (origin_mm / spacing_mm).round();
    if !voxel.is_finite() || voxel < f64::from(i16::MIN) || voxel > f64::from(i16::MAX) {
        anyhow::bail!(
            "Analyze: origin[{axis}]={origin_mm} maps to voxel coordinate {voxel}, outside the i16 header range"
        );
    }
    Ok(voxel as i16)
}
