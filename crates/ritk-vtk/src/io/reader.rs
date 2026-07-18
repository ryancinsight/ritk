//! VTK legacy structured points format reader.
//!
//! Parses the VTK legacy file format (version 1.0â€“5.1) restricted to
//! `DATASET STRUCTURED_POINTS` with scalar point data. Both ASCII and
//! BINARY encoding are supported.
//!
//! ## Coordinate Convention
//!
//! VTK header fields `DIMENSIONS`, `ORIGIN`, `SPACING` are in **[X, Y, Z]**
//! order. RITK spatial metadata (`Point`, `Spacing`) also uses **[X, Y, Z]**
//! order, so values transfer directly without permutation.
//!
//! RITK tensor shape is **[nz, ny, nx]** (Z varies slowest, X varies fastest).
//! VTK stores scalar data with X varying fastest, matching RITK's memory
//! layout. No data permutation is required.
//!
//! ## Supported Scalar Types
//!
//! `float`, `double`, `unsigned_char`, `short`, `unsigned_short`, `int`,
//! `unsigned_int`. All are converted to `f32` for the output tensor.

use anyhow::{bail, Context, Result};
use coeus_core::ComputeBackend;
use ritk_image::native::Image;
use ritk_spatial::{Direction, Point, Spacing};
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

/// Encoding declared in the VTK header (line 3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VtkEncoding {
    Ascii,
    Binary }

/// Scalar type declared by the `SCALARS` keyword.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VtkScalarType {
    Float,
    Double,
    UnsignedChar,
    Short,
    UnsignedShort,
    Int,
    UnsignedInt }

impl VtkScalarType {
    fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "float" => Ok(Self::Float),
            "double" => Ok(Self::Double),
            "unsigned_char" => Ok(Self::UnsignedChar),
            "short" => Ok(Self::Short),
            "unsigned_short" => Ok(Self::UnsignedShort),
            "int" => Ok(Self::Int),
            "unsigned_int" => Ok(Self::UnsignedInt),
            other => bail!("unsupported VTK scalar type: {}", other) }
    }

    /// Byte width of a single scalar element in binary encoding.
    fn byte_width(self) -> usize {
        match self {
            Self::Float => 4,
            Self::Double => 8,
            Self::UnsignedChar => 1,
            Self::Short => 2,
            Self::UnsignedShort => 2,
            Self::Int => 4,
            Self::UnsignedInt => 4 }
    }
}

/// Intermediate representation of parsed VTK header fields.
struct VtkHeader {
    encoding: VtkEncoding,
    dims: [usize; 3],  // [nx, ny, nz]
    origin: [f64; 3],  // [ox, oy, oz]
    spacing: [f64; 3], // [sx, sy, sz]
    point_data_n: usize,
    scalar_type: VtkScalarType }

/// Decode a VTK legacy structured-points file into substrate-free flat voxel
/// data plus geometry, without constructing any tensor or image carrier.
///
/// This is the shared core underlying both the burn-backed [`read_vtk`] and the
/// Coeus-backed `ritk_io` native reader: it performs the complete decode (header
/// parse, POINT_DATA validation, ASCII/binary scalar decode to `f32`) and
/// returns the raw ingredients so each caller can build its own carrier from an
/// identical byte-level decode.
///
/// ## Return convention
///
/// Returns `(data, dims, origin, spacing)` where:
/// - `data` is row-major scalar data with X varying fastest, Y next, Z slowest
///   (VTK's native storage order, matching RITK's `[nz, ny, nx]` tensor layout).
/// - `dims` is `[nx, ny, nz]` â€” VTK header `DIMENSIONS` **[X, Y, Z]** order, not
///   yet permuted to tensor `[nz, ny, nx]` order.
/// - `origin` / `spacing` are `[ox, oy, oz]` / `[sx, sy, sz]` in VTK **[X, Y, Z]**
///   order, transferring directly to RITK spatial metadata without permutation.
///
/// All scalar types (`float`, `double`, `unsigned_char`, `short`,
/// `unsigned_short`, `int`, `unsigned_int`) decode to `f32`. Binary payloads are
/// big-endian per the VTK legacy specification.
///
/// # Errors
///
/// Returns an error when:
/// - The file cannot be opened or read.
/// - The header does not conform to VTK legacy structured-points format.
/// - The declared scalar type is unsupported.
/// - The data section is truncated or malformed.
// The 4-tuple is a flat decode bundle (scalars, dims, origin, spacing) whose
// element roles are pinned in the "Return convention" doc section above; a
// wrapper struct would add a named type without improving call-site clarity,
// since both consumers destructure all four fields immediately.
#[allow(clippy::type_complexity)]
pub fn read_vtk_flat<P: AsRef<Path>>(
    path: P,
) -> Result<(Vec<f32>, [usize; 3], [f64; 3], [f64; 3])> {
    let path = path.as_ref();
    let file = std::fs::File::open(path)
        .with_context(|| format!("failed to open VTK file: {}", path.display()))?;
    let mut reader = BufReader::new(file);

    let header = parse_header(&mut reader).with_context(|| "failed to parse VTK header")?;

    let [nx, ny, nz] = header.dims;
    let expected_voxels = nx
        .checked_mul(ny)
        .and_then(|plane| plane.checked_mul(nz))
        .with_context(|| format!("VTK DIMENSIONS product overflows usize: {nx}Ã—{ny}Ã—{nz}"))?;

    if header.point_data_n != expected_voxels {
        bail!(
            "POINT_DATA count ({}) does not match DIMENSIONS product ({})",
            header.point_data_n,
            expected_voxels
        );
    }

    tracing::debug!(
        nx, ny, nz,
        ?header.encoding,
        "VTK structured points: reading {} voxels",
        expected_voxels
    );

    let data_f32 = match header.encoding {
        VtkEncoding::Binary => {
            read_binary_scalars(&mut reader, expected_voxels, header.scalar_type)
                .with_context(|| "failed to read VTK binary scalar data")?
        }
        VtkEncoding::Ascii => {
            crate::io::read_helpers::read_ascii::<f32>(&mut reader, expected_voxels, "f32")
                .with_context(|| "failed to read VTK ASCII scalar data")?
        }
    };

    Ok((data_f32, header.dims, header.origin, header.spacing))
}

/// Read a VTK legacy structured-points file into a native `Image`.
///
/// Decodes through [`read_vtk_flat`], then builds the Coeus-native carrier.
///
/// # Errors
///
/// Returns an error when:
/// - The file cannot be opened or read.
/// - The header does not conform to VTK legacy structured-points format.
/// - The declared scalar type is unsupported.
/// - The data section is truncated or malformed.
pub fn read_vtk<B: ComputeBackend, P: AsRef<Path>>(
    path: P,
    backend: &B,
) -> Result<Image<f32, B, 3>> {
    let (data_f32, [nx, ny, nz], origin_arr, spacing_arr) = read_vtk_flat(path)?;

    let origin = Point::new(origin_arr);
    let spacing = Spacing::new(spacing_arr);
    let direction = Direction::identity();

    tracing::debug!(
        ?origin,
        ?spacing,
        "VTK image constructed: shape=[{},{},{}]",
        nz,
        ny,
        nx
    );

    Image::from_flat_on(data_f32, [nz, ny, nx], origin, spacing, direction, backend)
}

// ---------------------------------------------------------------------------
// Header parsing
// ---------------------------------------------------------------------------

/// Read the next non-empty, non-comment line from the reader.
/// Returns `None` at EOF. Strips trailing `\r` / `\n`.
fn next_meaningful_line(reader: &mut impl BufRead) -> Result<Option<String>> {
    let mut buf = String::new();
    loop {
        buf.clear();
        let n = reader
            .read_line(&mut buf)
            .with_context(|| "I/O error reading VTK header line")?;
        if n == 0 {
            return Ok(None); // EOF
        }
        let trimmed = buf.trim();
        if trimmed.is_empty() {
            continue; // skip blank lines
        }
        return Ok(Some(trimmed.to_owned()));
    }
}

fn parse_header(reader: &mut BufReader<std::fs::File>) -> Result<VtkHeader> {
    // Line 1: magic / version
    let line1 =
        next_meaningful_line(reader)?.with_context(|| "unexpected EOF before VTK version line")?;
    if !line1.starts_with("# vtk DataFile Version") {
        bail!(
            "not a VTK legacy file (expected '# vtk DataFile Version ...', got '{}')",
            line1
        );
    }
    tracing::debug!(version_line = %line1, "VTK version line parsed");

    // Line 2: description (ignored, but must be present)
    let _description = next_meaningful_line(reader)?
        .with_context(|| "unexpected EOF before VTK description line")?;

    // Line 3: encoding
    let enc_line =
        next_meaningful_line(reader)?.with_context(|| "unexpected EOF before VTK encoding line")?;
    let encoding = match enc_line.to_ascii_uppercase().as_str() {
        "ASCII" => VtkEncoding::Ascii,
        "BINARY" => VtkEncoding::Binary,
        other => bail!("unsupported VTK encoding: {}", other) };
    tracing::debug!(?encoding, "VTK encoding parsed");

    // Line 4: dataset type
    let ds_line =
        next_meaningful_line(reader)?.with_context(|| "unexpected EOF before VTK DATASET line")?;
    if !ds_line
        .to_ascii_uppercase()
        .starts_with("DATASET STRUCTURED_POINTS")
    {
        bail!(
            "unsupported VTK dataset type (expected STRUCTURED_POINTS, got '{}')",
            ds_line
        );
    }

    // Remaining header fields (order is not strictly fixed by the spec, so
    // we parse them in any order until we have everything).
    let mut dims: Option<[usize; 3]> = None;
    let mut origin: Option<[f64; 3]> = None;
    let mut spacing: Option<[f64; 3]> = None;
    let mut point_data_n: Option<usize> = None;
    let mut scalar_type: Option<VtkScalarType> = None;

    loop {
        let line = match next_meaningful_line(reader)? {
            Some(l) => l,
            None => break };
        let upper = line.to_ascii_uppercase();
        let tokens: Vec<&str> = line.split_whitespace().collect();

        if upper.starts_with("DIMENSIONS") {
            if tokens.len() < 4 {
                bail!("DIMENSIONS line requires 3 values, got: '{}'", line);
            }
            let nx: usize = tokens[1].parse().with_context(|| "bad DIMENSIONS nx")?;
            let ny: usize = tokens[2].parse().with_context(|| "bad DIMENSIONS ny")?;
            let nz: usize = tokens[3].parse().with_context(|| "bad DIMENSIONS nz")?;
            dims = Some([nx, ny, nz]);
            tracing::debug!(nx, ny, nz, "VTK DIMENSIONS parsed");
        } else if upper.starts_with("ORIGIN") {
            if tokens.len() < 4 {
                bail!("ORIGIN line requires 3 values, got: '{}'", line);
            }
            let ox: f64 = tokens[1].parse().with_context(|| "bad ORIGIN ox")?;
            let oy: f64 = tokens[2].parse().with_context(|| "bad ORIGIN oy")?;
            let oz: f64 = tokens[3].parse().with_context(|| "bad ORIGIN oz")?;
            origin = Some([ox, oy, oz]);
            tracing::debug!(ox, oy, oz, "VTK ORIGIN parsed");
        } else if upper.starts_with("SPACING") || upper.starts_with("ASPECT_RATIO") {
            if tokens.len() < 4 {
                bail!("SPACING line requires 3 values, got: '{}'", line);
            }
            let sx: f64 = tokens[1].parse().with_context(|| "bad SPACING sx")?;
            let sy: f64 = tokens[2].parse().with_context(|| "bad SPACING sy")?;
            let sz: f64 = tokens[3].parse().with_context(|| "bad SPACING sz")?;
            spacing = Some([sx, sy, sz]);
            tracing::debug!(sx, sy, sz, "VTK SPACING parsed");
        } else if upper.starts_with("POINT_DATA") {
            if tokens.len() < 2 {
                bail!("POINT_DATA line requires a count, got: '{}'", line);
            }
            let n: usize = tokens[1].parse().with_context(|| "bad POINT_DATA count")?;
            point_data_n = Some(n);
            tracing::debug!(n, "VTK POINT_DATA parsed");
        } else if upper.starts_with("SCALARS") {
            // SCALARS name type [ncomp]
            if tokens.len() < 3 {
                bail!(
                    "SCALARS line requires at least name and type, got: '{}'",
                    line
                );
            }
            let stype = VtkScalarType::from_str(tokens[2])
                .with_context(|| format!("bad SCALARS type in line: '{}'", line))?;
            scalar_type = Some(stype);
            tracing::debug!(?stype, name = tokens[1], "VTK SCALARS parsed");
        } else if upper.starts_with("LOOKUP_TABLE") {
            // Marks the end of the header; data follows immediately.
            tracing::debug!("VTK LOOKUP_TABLE line reached; data follows");
            break;
        }
        // Unknown header lines are silently skipped (forward compatibility).
    }

    let dims = dims.with_context(|| "VTK header missing DIMENSIONS")?;
    let origin = origin.unwrap_or([0.0, 0.0, 0.0]);
    let spacing = spacing.unwrap_or([1.0, 1.0, 1.0]);
    let point_data_n = point_data_n.with_context(|| "VTK header missing POINT_DATA")?;
    let scalar_type = scalar_type.with_context(|| "VTK header missing SCALARS")?;

    Ok(VtkHeader {
        encoding,
        dims,
        origin,
        spacing,
        point_data_n,
        scalar_type })
}

// ---------------------------------------------------------------------------
// Binary data reading
// ---------------------------------------------------------------------------

/// Read `count` big-endian scalars of the given type and convert each to f32.
fn read_binary_scalars(
    reader: &mut impl Read,
    count: usize,
    scalar_type: VtkScalarType,
) -> Result<Vec<f32>> {
    let byte_width = scalar_type.byte_width();
    let total_bytes = count
        .checked_mul(byte_width)
        .with_context(|| "scalar data size overflow")?;

    // Bound the speculative allocation: `count` is a header field and may exceed
    // the bytes actually present. `read_exact_bounded` grows the buffer per
    // confirmed chunk and reports truncation rather than aborting on OOM.
    let raw = consus_io::read_exact_bounded(reader, total_bytes).with_context(|| {
        format!("failed to read {total_bytes} bytes of VTK binary data ({count} voxels Ã— {byte_width} bytes)")
    })?;

    let mut out = Vec::with_capacity(count);

    match scalar_type {
        VtkScalarType::Float => {
            for chunk in raw.chunks_exact(4) {
                let val = f32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                out.push(val);
            }
        }
        VtkScalarType::Double => {
            for chunk in raw.chunks_exact(8) {
                let val = f64::from_be_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]);
                out.push(val as f32);
            }
        }
        VtkScalarType::UnsignedChar => {
            for &b in &raw {
                out.push(b as f32);
            }
        }
        VtkScalarType::Short => {
            for chunk in raw.chunks_exact(2) {
                let val = i16::from_be_bytes([chunk[0], chunk[1]]);
                out.push(val as f32);
            }
        }
        VtkScalarType::UnsignedShort => {
            for chunk in raw.chunks_exact(2) {
                let val = u16::from_be_bytes([chunk[0], chunk[1]]);
                out.push(val as f32);
            }
        }
        VtkScalarType::Int => {
            for chunk in raw.chunks_exact(4) {
                let val = i32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                out.push(val as f32);
            }
        }
        VtkScalarType::UnsignedInt => {
            for chunk in raw.chunks_exact(4) {
                let val = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                out.push(val as f32);
            }
        }
    }

    if out.len() != count {
        bail!(
            "binary scalar parse produced {} values, expected {}",
            out.len(),
            count
        );
    }

    Ok(out)
}
