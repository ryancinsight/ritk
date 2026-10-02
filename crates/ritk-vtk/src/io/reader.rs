//! VTK legacy structured points format reader.
//!
//! Parses the VTK legacy file format (version 1.0–5.1) restricted to
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
//! ## Stored Scalar Types
//!
//! The reader keeps the type the `SCALARS` line declares (`unsigned_char`,
//! `char`, `unsigned_short`, `short`, `unsigned_int`, `int`, `vtktypeuint64`,
//! `vtktypeint64`, `float`, `double`; see the `scalar_type` module and
//! converts it to the caller's sample type under the caller's
//! [`Conversion`]. `bit` and `long`/`unsigned_long` are refused by name.

use crate::io::read_helpers::read_ascii_samples;
use crate::io::scalar_type::{sample_type_from_name, VtkEncoding};
use anyhow::{anyhow, bail, Context, Result};
use coeus_core::ComputeBackend;
use consus_core::ByteOrder;
use ritk_codecs::sample::{Conversion, Sample, SampleBuffer, SampleType};
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};
use std::io::{self, BufRead, BufReader};
use std::path::Path;

/// Intermediate representation of parsed VTK header fields.
struct VtkHeader {
    encoding: VtkEncoding,
    dims: [usize; 3],  // [nx, ny, nz]
    origin: [f64; 3],  // [ox, oy, oz]
    spacing: [f64; 3], // [sx, sy, sz]
    point_data_n: usize,
    sample_type: SampleType,
}

/// Decode a VTK legacy structured-points file into substrate-free flat voxel
/// data plus geometry, without constructing any tensor or image carrier.
///
/// This is the shared core underlying [`read_vtk`]: it performs the complete
/// decode (header parse, POINT_DATA validation, ASCII/binary scalar decode in
/// the stored type) and converts the samples to `T` under `conversion`.
///
/// ## Return convention
///
/// Returns `(data, dims, origin, spacing)` where:
/// - `data` is row-major scalar data with X varying fastest, Y next, Z slowest
///   (VTK's native storage order, matching RITK's `[nz, ny, nx]` tensor layout).
/// - `dims` is `[nx, ny, nz]` — VTK header `DIMENSIONS` **[X, Y, Z]** order, not
///   yet permuted to tensor `[nz, ny, nx]` order.
/// - `origin` / `spacing` are `[ox, oy, oz]` / `[sx, sy, sz]` in VTK **[X, Y, Z]**
///   order, transferring directly to RITK spatial metadata without permutation.
///
/// The stored scalar type (`unsigned_char`, `char`, `unsigned_short`, `short`,
/// `unsigned_int`, `int`, `vtktypeuint64`, `vtktypeint64`, `float`, `double`)
/// converts to `T` under `conversion`: [`Exact`](ritk_codecs::sample::Exact)
/// accepts the stored type or a type it widens to. Binary payloads are
/// big-endian per the VTK legacy specification.
///
/// # Errors
///
/// Returns an error when:
/// - The file cannot be opened or read.
/// - The header does not conform to VTK legacy structured-points format, or its
///   `SCALARS` array has more than one component.
/// - The declared scalar type is `bit`, `long`, `unsigned_long`, or unknown.
/// - The data section is truncated or malformed.
/// - `conversion` refuses the stored type.
// The 4-tuple is a flat decode bundle (scalars, dims, origin, spacing) whose
// element roles are pinned in the "Return convention" doc section above; a
// wrapper struct would add a named type without improving call-site clarity,
// since the consumer destructures all four fields immediately.
#[expect(clippy::type_complexity, reason = "ratchet RITK-LINT-1")]
pub fn read_vtk_flat<T: Sample, C: Conversion, P: AsRef<Path>>(
    path: P,
    conversion: C,
) -> Result<(Vec<T>, [usize; 3], [f64; 3], [f64; 3])> {
    let path = path.as_ref();
    let file = std::fs::File::open(path)
        .with_context(|| format!("failed to open VTK file: {}", path.display()))?;
    let mut reader = BufReader::new(file);

    let header = parse_header(&mut reader).with_context(|| "failed to parse VTK header")?;

    let [nx, ny, nz] = header.dims;
    let expected_voxels = nx
        .checked_mul(ny)
        .and_then(|plane| plane.checked_mul(nz))
        .with_context(|| format!("VTK DIMENSIONS product overflows usize: {nx}×{ny}×{nz}"))?;

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
        stored = %header.sample_type,
        "VTK structured points: reading {} voxels",
        expected_voxels
    );

    let stored = match header.encoding {
        VtkEncoding::Binary => SampleBuffer::read_from(
            &mut reader,
            header.sample_type,
            ByteOrder::BigEndian,
            expected_voxels,
        )
        .map_err(|error| match error.kind() {
            io::ErrorKind::UnexpectedEof => {
                anyhow!("VTK binary scalar data is truncated: {error}")
            }
            _ => anyhow::Error::new(error).context("failed to read VTK binary scalar data"),
        })?,
        VtkEncoding::Ascii => read_ascii_samples(&mut reader, header.sample_type, expected_voxels)
            .with_context(|| "failed to read VTK ASCII scalar data")?,
    };

    let data = conversion.convert::<T>(stored)?;
    Ok((data, header.dims, header.origin, header.spacing))
}

/// Read a VTK legacy structured-points file into a native `Image` of `T`.
///
/// Decodes through [`read_vtk_flat`], then builds the Coeus-native carrier.
/// The stored scalar type converts to `T` under `conversion`.
///
/// # Errors
///
/// Returns an error when:
/// - The file cannot be opened or read.
/// - The header does not conform to VTK legacy structured-points format.
/// - The declared scalar type is unsupported.
/// - The data section is truncated or malformed.
/// - `conversion` refuses the stored type.
pub fn read_vtk<T, C, B, P>(path: P, backend: &B, conversion: C) -> Result<Image<T, B, 3>>
where
    T: Sample,
    C: Conversion,
    B: ComputeBackend,
    P: AsRef<Path>,
{
    let (data, [nx, ny, nz], origin_arr, spacing_arr) = read_vtk_flat(path, conversion)?;

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

    Image::from_flat_on(data, [nz, ny, nx], origin, spacing, direction, backend)
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
        other => bail!("unsupported VTK encoding: {}", other),
    };
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
    let mut sample_type: Option<SampleType> = None;

    loop {
        let line = match next_meaningful_line(reader)? {
            Some(l) => l,
            None => break,
        };
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
            let stype = sample_type_from_name(tokens[2])
                .with_context(|| format!("bad SCALARS type in line: '{}'", line))?;
            let components: usize = match tokens.get(3) {
                Some(token) => token
                    .parse()
                    .with_context(|| format!("bad SCALARS component count in line: '{}'", line))?,
                None => 1,
            };
            if components != 1 {
                bail!(
                    "SCALARS array '{}' has {components} components; structured-points reading supports exactly one",
                    tokens[1]
                );
            }
            sample_type = Some(stype);
            tracing::debug!(%stype, name = tokens[1], "VTK SCALARS parsed");
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
    let sample_type = sample_type.with_context(|| "VTK header missing SCALARS")?;

    Ok(VtkHeader {
        encoding,
        dims,
        origin,
        spacing,
        point_data_n,
        sample_type,
    })
}
