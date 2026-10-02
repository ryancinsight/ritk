use crate::element_type::sample_type_from_element_type;
use crate::spatial::metadata_from_file_transform;
use anyhow::{anyhow, Context, Result};
use coeus_core::ComputeBackend;
use consus_core::ByteOrder;
use flate2::bufread::ZlibDecoder;
use ritk_codecs::parse_header_values;
use ritk_codecs::sample::{
    count_payload_bytes, validate_remaining_payload, Conversion, Sample, SampleBuffer, SampleType,
};
use ritk_image::Image;
use ritk_spatial::Point;
use std::collections::HashMap;
use std::io::{self, BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::Path;

/// Read a MetaImage (.mha or .mhd) file into a 3-D `Image` of `T`.
///
/// # Axis convention
/// MetaImage stores voxels in X-fastest `[X, Y, Z]` order. The same flat byte
/// sequence is RITK-contiguous when shaped as `[Z, Y, X]`, so the returned
/// tensor shape is `[nz, ny, nx]` without a data permutation.
///
/// # Spatial metadata
/// `origin` remains in physical coordinate order. `ElementSpacing` and
/// `TransformMatrix` are converted from MetaImage `[X,Y,Z]` file axes into
/// RITK `[Z,Y,X]` image-axis metadata.
///
/// # Supported element types
/// `MET_CHAR`, `MET_UCHAR`, `MET_SHORT`, `MET_USHORT`, `MET_INT`, `MET_UINT`,
/// `MET_LONG`, `MET_ULONG`, `MET_LONG_LONG`, `MET_ULONG_LONG`, `MET_FLOAT`, and
/// `MET_DOUBLE`, stored in the byte order `BinaryDataByteOrderMSB` names;
/// `MET_LONG` and `MET_ULONG` are MetaIO's four-byte integers. The samples decode in the
/// stored type and then convert to `T` under `conversion`:
/// [`Exact`](ritk_codecs::sample::Exact) accepts the stored type or a type it
/// widens to and refuses a read that could change a value;
/// [`Cast`](ritk_codecs::sample::Cast) converts and warns.
///
/// # File formats
/// * `.mha` — single file; header followed immediately by binary data
///   (`ElementDataFile = LOCAL`).
/// * `.mhd` / `.raw` — ASCII header references a separate raw file.
///
/// Either payload is zlib-deflated when `CompressedData = True`.
///
/// # Errors
///
/// Returns an error when the file cannot be opened, a required header field is
/// missing or malformed, the `ElementType` is not one listed above, the payload
/// does not hold exactly `DimSize` samples, or `conversion` refuses the stored
/// type.
pub fn read_metaimage<T, C, B, P>(path: P, backend: &B, conversion: C) -> Result<Image<T, B, 3>>
where
    T: Sample,
    C: Conversion,
    B: ComputeBackend,
    P: AsRef<Path>,
{
    let DecodedMetaImage {
        data,
        dims,
        origin,
        spacing,
        direction,
    } = decode_metaimage(path, conversion)?;
    Image::from_flat_on(data, dims, origin, spacing, direction, backend)
}

/// Backend-agnostic decoded MetaImage volume: voxels in `[nz, ny, nx]` order plus
/// the derived physical metadata.
struct DecodedMetaImage<T> {
    data: Vec<T>,
    dims: [usize; 3],
    origin: ritk_spatial::Point<3>,
    spacing: ritk_spatial::Spacing<3>,
    direction: ritk_spatial::Direction<3>,
}

fn decode_metaimage<T: Sample, C: Conversion, P: AsRef<Path>>(
    path: P,
    conversion: C,
) -> Result<DecodedMetaImage<T>> {
    let path = path.as_ref();

    let file = std::fs::File::open(path)
        .with_context(|| format!("Cannot open MetaImage file {:?}", path))?;
    let mut reader = BufReader::new(file);

    // ── Header parsing ────────────────────────────────────────────────────
    // Read line-by-line; the `ElementDataFile` line ends the header and leaves
    // the reader at the first payload byte.
    let mut headers: HashMap<String, String> = HashMap::new();
    let mut found_edf = false;

    loop {
        let mut line = String::new();
        let n = reader
            .read_line(&mut line)
            .context("Error reading MetaImage header line")?;
        if n == 0 {
            break; // unexpected EOF before ElementDataFile
        }

        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        // MetaImage uses " = " as the key-value separator.
        if let Some(eq_pos) = trimmed.find(" = ") {
            let key = trimmed[..eq_pos].trim().to_string();
            let value = trimmed[eq_pos + 3..].trim().to_string();
            let is_edf = key == "ElementDataFile";
            headers.insert(key, value);
            if is_edf {
                found_edf = true;
                break; // binary data starts here
            }
        }
    }

    if !found_edf {
        return Err(anyhow!(
            "ElementDataFile key not found in MetaImage header of {:?}",
            path
        ));
    }

    // ── Required fields ───────────────────────────────────────────────────
    let ndims: usize = headers
        .get("NDims")
        .ok_or_else(|| anyhow!("Missing 'NDims' in MetaImage header"))?
        .parse()
        .context("'NDims' is not a valid integer")?;

    // 2-D images are promoted to a degenerate `[1, Y, X]` (z = 1) volume: ritk's
    // Image has three axes, so a 2-D file becomes a single-slice 3-D image.
    if ndims != 2 && ndims != 3 {
        return Err(anyhow!("Expected NDims = 2 or 3, found {}", ndims));
    }

    let dim_sizes = parse_header_values::<usize>(
        headers
            .get("DimSize")
            .ok_or_else(|| anyhow!("Missing 'DimSize' in MetaImage header"))?,
        "DimSize",
        ndims,
    )?;
    let nx = dim_sizes[0];
    let ny = dim_sizes[1];
    let nz = if ndims == 3 { dim_sizes[2] } else { 1 };

    let spacing_raw = parse_header_values::<f64>(
        headers
            .get("ElementSpacing")
            .ok_or_else(|| anyhow!("Missing 'ElementSpacing' in MetaImage header"))?,
        "ElementSpacing",
        ndims,
    )?;
    // Promote spacing with unit z when 2-D.
    let spacing_vals = if ndims == 3 {
        spacing_raw
    } else {
        vec![spacing_raw[0], spacing_raw[1], 1.0]
    };

    let offset_str = headers
        .get("Offset")
        .or_else(|| headers.get("Position"))
        .ok_or_else(|| anyhow!("Missing 'Offset' (or 'Position') in MetaImage header"))?;
    let offset_raw = parse_header_values::<f64>(offset_str, "Offset", ndims)?;
    let offset_vals = if ndims == 3 {
        offset_raw
    } else {
        vec![offset_raw[0], offset_raw[1], 0.0]
    };

    // TransformMatrix is row-major direction cosines (ndims² entries); defaults to
    // identity when absent.  A 2-D `[a b; c d]` matrix promotes to the 3-D
    // `[a b 0; c d 0; 0 0 1]` (identity through-plane z-axis).
    let tm_default = if ndims == 3 {
        "1 0 0 0 1 0 0 0 1"
    } else {
        "1 0 0 1"
    };
    let tm_src = headers
        .get("TransformMatrix")
        .map(|s| s.as_str())
        .unwrap_or(tm_default);
    let tm_raw = parse_header_values::<f64>(tm_src, "TransformMatrix", ndims * ndims)?;
    let tm_vals = if ndims == 3 {
        tm_raw
    } else {
        vec![
            tm_raw[0], tm_raw[1], 0.0, tm_raw[2], tm_raw[3], 0.0, 0.0, 0.0, 1.0,
        ]
    };

    let element_type = headers
        .get("ElementType")
        .ok_or_else(|| anyhow!("Missing 'ElementType' in MetaImage header"))?
        .clone();
    let sample_type = sample_type_from_element_type(&element_type)?;
    let elem_size = sample_type.byte_width();

    // BinaryDataByteOrderMSB = True → big-endian; default is little-endian.
    let byte_order = parse_byte_order_msb(
        headers
            .get("BinaryDataByteOrderMSB")
            .map(|s| s.as_str())
            .unwrap_or("FALSE"),
    );

    // CompressedData = True → the payload is zlib-deflated; default is raw.
    let compressed = headers
        .get("CompressedData")
        .map(|s| s.to_uppercase() == "TRUE")
        .unwrap_or(false);

    let element_data_file = headers
        .get("ElementDataFile")
        .ok_or_else(|| anyhow!("Missing 'ElementDataFile' in MetaImage header"))?
        .clone();

    // ── Binary data ───────────────────────────────────────────────────────
    let total_voxels = checked_voxel_count(nx, ny, nz)?;
    let expected_payload_bytes = total_voxels.checked_mul(elem_size).ok_or_else(|| {
        anyhow!(
            "MetaImage byte count overflow: {} voxels × {} bytes",
            total_voxels,
            elem_size
        )
    })?;

    // The header line that ends the header leaves `reader` at the first payload
    // byte, so an inline payload reads on from it; a detached one opens its file.
    let layout = |origin: String| PayloadLayout {
        sample_type,
        byte_order,
        compressed,
        count: total_voxels,
        expected_bytes: expected_payload_bytes,
        element_type: &element_type,
        origin,
    };
    let samples = if element_data_file.eq_ignore_ascii_case("LOCAL") {
        read_payload(&mut reader, &layout(".mha file".to_string()))?
    } else {
        // External .raw file: resolve relative to the header file's directory.
        let raw_path = path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(&element_data_file);
        let raw = std::fs::File::open(&raw_path)
            .with_context(|| format!("Cannot read raw data file {:?}", raw_path))?;
        read_payload(
            BufReader::new(raw),
            &layout(format!("raw data file {raw_path:?}")),
        )?
    };
    let data = conversion.convert::<T>(samples)?;

    // ── Spatial metadata ──────────────────────────────────────────────────
    // MetaImage X-fastest flat order equals row-major order for RITK [Z,Y,X].
    let origin = Point::new([offset_vals[0], offset_vals[1], offset_vals[2]]);
    let spatial = metadata_from_file_transform(
        [spacing_vals[0], spacing_vals[1], spacing_vals[2]],
        [
            tm_vals[0], tm_vals[1], tm_vals[2], tm_vals[3], tm_vals[4], tm_vals[5], tm_vals[6],
            tm_vals[7], tm_vals[8],
        ],
    );

    Ok(DecodedMetaImage {
        data,
        dims: [nz, ny, nx],
        origin,
        spacing: spatial.spacing,
        direction: spatial.direction,
    })
}

// ── Private helpers ───────────────────────────────────────────────────────────

fn checked_voxel_count(nx: usize, ny: usize, nz: usize) -> Result<usize> {
    nx.checked_mul(ny)
        .and_then(|xy| xy.checked_mul(nz))
        .ok_or_else(|| {
            anyhow!(
                "MetaImage voxel count overflow: DimSize = {} {} {}",
                nx,
                ny,
                nz
            )
        })
}

/// How the payload of one file is laid out: the stored type, its byte order,
/// whether it is zlib-deflated, and the sample count `DimSize` declares.
struct PayloadLayout<'a> {
    sample_type: SampleType,
    byte_order: ByteOrder,
    compressed: bool,
    count: usize,
    expected_bytes: usize,
    element_type: &'a str,
    /// The file the payload comes from, for error context.
    origin: String,
}

impl PayloadLayout<'_> {
    /// The error for a payload that does not hold exactly `count` samples;
    /// `found` says how it differs.
    fn length_mismatch(&self, found: &str) -> anyhow::Error {
        anyhow!(
            "MetaImage payload length mismatch: expected {} bytes from DimSize ({} voxels × {} bytes for {}), but {}",
            self.expected_bytes,
            self.count,
            self.sample_type.byte_width(),
            self.element_type,
            found
        )
    }
}

/// Read exactly `layout.count` samples from `source`, inflating first when the
/// payload is compressed.
fn read_payload<R: BufRead + Seek>(
    mut source: R,
    layout: &PayloadLayout<'_>,
) -> Result<SampleBuffer> {
    if layout.compressed {
        let start = source
            .stream_position()
            .with_context(|| read_failure(layout))?;
        let actual = {
            let mut decoder = ZlibDecoder::new(&mut source);
            count_payload_bytes(&mut decoder, layout.expected_bytes)
                .with_context(|| read_failure(layout))?
        };
        match actual {
            Some(actual) if actual != layout.expected_bytes => {
                return Err(layout.length_mismatch(&format!(
                    "the decompressed payload ended after {actual} bytes"
                )));
            }
            None => {
                return Err(layout.length_mismatch(
                    "the payload continues past that length after decompression",
                ));
            }
            Some(_) => {}
        }
        source
            .seek(SeekFrom::Start(start))
            .with_context(|| read_failure(layout))?;
        read_exact_samples(ZlibDecoder::new(source), layout)
    } else {
        validate_remaining_payload(&mut source, layout.expected_bytes).map_err(|error| {
            if error.kind() == io::ErrorKind::UnexpectedEof {
                layout.length_mismatch(&format!("the payload ended early ({error})"))
            } else if error.kind() == io::ErrorKind::InvalidData {
                layout.length_mismatch("the payload continues past that length")
            } else {
                anyhow::Error::from(error).context(read_failure(layout))
            }
        })?;
        read_exact_samples(source, layout)
    }
}

/// The context of a read failure other than a payload of the wrong length.
fn read_failure(layout: &PayloadLayout<'_>) -> String {
    if layout.compressed {
        format!(
            "Failed to inflate zlib-compressed MetaImage payload from {}",
            layout.origin
        )
    } else {
        format!("Failed to read binary voxel data from {}", layout.origin)
    }
}

/// Decode `layout.count` samples from `stream` and require the stream to end
/// there. The decoder grows its buffer only by samples it has read, so a
/// hostile `DimSize` reserves nothing the stream does not supply.
fn read_exact_samples<R: Read>(mut stream: R, layout: &PayloadLayout<'_>) -> Result<SampleBuffer> {
    let samples = SampleBuffer::read_from(
        &mut stream,
        layout.sample_type,
        layout.byte_order,
        layout.count,
    )
    .map_err(|error| match error.kind() {
        io::ErrorKind::UnexpectedEof => {
            layout.length_mismatch(&format!("the payload ended early ({error})"))
        }
        _ => anyhow::Error::from(error).context(read_failure(layout)),
    })?;
    let mut trailing = Vec::with_capacity(1);
    stream
        .by_ref()
        .take(1)
        .read_to_end(&mut trailing)
        .with_context(|| read_failure(layout))?;
    if !trailing.is_empty() {
        return Err(layout.length_mismatch("the payload continues past that length"));
    }
    Ok(samples)
}

/// The payload byte order a `BinaryDataByteOrderMSB` value names: `True`
/// (case-insensitive) is big-endian, any other value little-endian.
pub(crate) fn parse_byte_order_msb(value: &str) -> ByteOrder {
    if value.eq_ignore_ascii_case("TRUE") {
        ByteOrder::BigEndian
    } else {
        ByteOrder::LittleEndian
    }
}

// ── Public reader struct ──────────────────────────────────────────────────────

/// Thin reader struct for MetaImage files.
///
/// The backend `B` and device are supplied per-call so a single
/// `MetaImageReader` instance can serve multiple backends.
pub struct MetaImageReader;

impl MetaImageReader {
    /// Read a MetaImage file at `path` into an [`Image`] of `T` on `backend`,
    /// converting the stored samples under `conversion`.
    ///
    /// # Errors
    ///
    /// See [`read_metaimage`].
    pub fn read<T, C, B, P>(&self, path: P, backend: &B, conversion: C) -> Result<Image<T, B, 3>>
    where
        T: Sample,
        C: Conversion,
        B: ComputeBackend,
        P: AsRef<Path>,
    {
        read_metaimage(path, backend, conversion)
    }
}
