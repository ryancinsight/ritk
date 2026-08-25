#![expect(clippy::unwrap_used, reason = "ratchet RITK-UNWRAP-1")]
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};

use gaia::Polyline;
use leto::geometry::Point3;

use crate::types::{TckDatatype, TckHeader, TckTractogram};

/// Error returned when reading or writing a `.tck` file.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum TckError {
    /// Input exhausted before the expected number of bytes was read.
    #[error("unexpected end of file at byte offset {offset}")]
    UnexpectedEof {
        /// Byte position where the read stopped.
        offset: usize,
    },

    /// The first header line is not `mrtrix tracks`.
    #[error("invalid .tck magic: first line must be 'mrtrix tracks', got {0:?}")]
    InvalidMagic(String),

    /// A header line is not a valid `key: value` pair.
    #[error("malformed header line at byte offset {offset}: {line:?}")]
    MalformedHeaderLine {
        /// Byte offset within the file.
        offset: usize,
        /// The offending line.
        line: String,
    },

    /// The `datatype` header value is missing or unrecognised.
    #[error("unknown or missing datatype: {0:?}")]
    UnknownDatatype(String),

    /// The `transform` header value does not contain 16 floats.
    #[error("invalid transform: expected 16 float values, got {count}")]
    InvalidTransform {
        /// Number of values parsed.
        count: usize,
    },

    /// A coordinate component is NaN or infinite.
    #[error("non-finite coordinate in streamline {index}, point {point_index}")]
    NonFiniteCoordinate {
        /// Streamline index.
        index: usize,
        /// Point index within the streamline.
        point_index: usize,
    },

    /// Gaia rejected the point sequence.
    #[error("invalid streamline {index}: {source}")]
    InvalidPolyline {
        /// Streamline index.
        index: usize,
        /// Error from Gaia.
        #[source]
        source: gaia::PolylineError,
    },

    /// An I/O error occurred.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// First line of every `.tck` file.
const TCK_MAGIC_LINE: &str = "mrtrix tracks";

/// The `END` sentinel that terminates the header section.
const TCK_END_LINE: &str = "END";

impl TckTractogram {
    /// Read a `.tck` file from any byte source.
    ///
    /// # Errors
    ///
    /// Returns [`TckError`] for invalid magic, missing header fields,
    /// premature EOF, non‑finite coordinates, or invalid Gaia geometry.
    pub fn read(reader: impl Read) -> Result<Self, TckError> {
        let mut buf_reader = BufReader::new(reader);
        let mut offset: usize = 0;

        // ── Header ──────────────────────────────────────────────────────
        let mut first_line = String::new();
        let n = buf_reader.read_line(&mut first_line)?;
        offset += n;
        let first_line = first_line.trim().to_string();
        if first_line != TCK_MAGIC_LINE {
            return Err(TckError::InvalidMagic(first_line));
        }

        let mut fields: Vec<(String, String)> = Vec::new();
        loop {
            let mut line = String::new();
            let n = buf_reader.read_line(&mut line)?;
            offset += n;
            let trimmed = line.trim().to_string();
            if trimmed == TCK_END_LINE {
                break;
            }
            if trimmed.is_empty() {
                continue;
            }
            let (key, value) =
                parse_header_line(&trimmed).ok_or_else(|| TckError::MalformedHeaderLine {
                    offset,
                    line: trimmed.clone(),
                })?;
            fields.push((key.to_string(), value.to_string()));
        }

        let header = build_header(&fields)?;

        // ── Streamlines ─────────────────────────────────────────────────
        let mut streamlines: Vec<Polyline<f64>> = Vec::new();
        let mut current_points: Vec<Point3<f64>> = Vec::new();
        let mut buf = [0u8; 24];
        let point_size = header.datatype.bytes_per_point();

        loop {
            match buf_reader.read_exact(&mut buf[..point_size]) {
                Ok(()) => {}
                Err(ref e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                    break;
                }
                Err(e) => return Err(TckError::Io(e)),
            }

            let (x, y, z) = decode_point(header.datatype, &buf[..point_size]);

            if x.is_infinite() && y.is_infinite() && z.is_infinite() {
                break;
            }

            if x.is_nan() && y.is_nan() && z.is_nan() {
                if !current_points.is_empty() {
                    let index = streamlines.len();
                    let polyline = Polyline::new(std::mem::take(&mut current_points))
                        .map_err(|source| TckError::InvalidPolyline { index, source })?;
                    streamlines.push(polyline);
                }
                continue;
            }

            if !x.is_finite() || !y.is_finite() || !z.is_finite() {
                return Err(TckError::NonFiniteCoordinate {
                    index: streamlines.len(),
                    point_index: current_points.len(),
                });
            }

            current_points.push(Point3::new(x, y, z));
        }

        if !current_points.is_empty() {
            let index = streamlines.len();
            let polyline = Polyline::new(current_points)
                .map_err(|source| TckError::InvalidPolyline { index, source })?;
            streamlines.push(polyline);
        }

        Ok(Self {
            header,
            streamlines,
        })
    }

    /// Write a tractogram to a `.tck` file.
    ///
    /// Streamlines are expected in scanner-space millimetre coordinates.
    ///
    /// # Errors
    ///
    /// Returns [`TckError`] on write failure.
    pub fn write(&self, writer: &mut impl Write) -> Result<(), TckError> {
        writeln!(writer, "{}", TCK_MAGIC_LINE)?;
        writeln!(writer, "datatype: {}", self.header.datatype.as_str())?;

        if let Some(ref version) = self.header.mrtrix_version {
            writeln!(writer, "mrtrix_version: {version}")?;
        }
        if let Some(ref file_path) = self.header.file_path {
            writeln!(writer, "file: {file_path}")?;
        }
        if let Some(ref comments) = self.header.comments {
            for line in comments.lines() {
                writeln!(writer, "comments: {line}")?;
            }
        }
        let count = self.streamlines.len();
        writeln!(writer, "count: {count}")?;
        writeln!(writer, "total_count: {count}")?;

        if let Some(ref t) = self.header.transform {
            write!(
                writer,
                "transform: {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}",
                t[0][0],
                t[0][1],
                t[0][2],
                t[0][3],
                t[1][0],
                t[1][1],
                t[1][2],
                t[1][3],
                t[2][0],
                t[2][1],
                t[2][2],
                t[2][3],
                t[3][0],
                t[3][1],
                t[3][2],
                t[3][3],
            )?;
            writeln!(writer)?;
        }

        for (key, value) in &self.header.fields {
            if matches!(
                key.as_str(),
                "datatype"
                    | "mrtrix_version"
                    | "file"
                    | "comments"
                    | "count"
                    | "total_count"
                    | "transform"
            ) {
                continue;
            }
            writeln!(writer, "{key}: {value}")?;
        }

        writeln!(writer, "END")?;

        let dt = self.header.datatype;
        for polyline in &self.streamlines {
            for point in polyline.points() {
                encode_and_write(point.x, point.y, point.z, dt, writer)?;
            }
            encode_and_write(f64::NAN, f64::NAN, f64::NAN, dt, writer)?;
        }
        encode_and_write(f64::INFINITY, f64::INFINITY, f64::INFINITY, dt, writer)?;

        Ok(())
    }
}

/// Parse a header line into `(key, value)`, splitting on the first colon.
fn parse_header_line(line: &str) -> Option<(&str, &str)> {
    let pos = line.find(':')?;
    let key = line[..pos].trim();
    let value = line[pos + 1..].trim();
    if key.is_empty() {
        return None;
    }
    Some((key, value))
}

/// Build a [`TckHeader`] from parsed key-value pairs.
fn build_header(fields: &[(String, String)]) -> Result<TckHeader, TckError> {
    let mut header = TckHeader::default();
    let mut map = HashMap::with_capacity(fields.len());

    for (key, value) in fields {
        map.insert(key.clone(), value.clone());
        match key.as_str() {
            "count" => {
                header.count =
                    Some(
                        value
                            .parse::<i64>()
                            .map_err(|_| TckError::MalformedHeaderLine {
                                offset: 0,
                                line: format!("{key}: {value}"),
                            })?,
                    );
            }
            "total_count" => {
                header.total_count =
                    Some(
                        value
                            .parse::<i64>()
                            .map_err(|_| TckError::MalformedHeaderLine {
                                offset: 0,
                                line: format!("{key}: {value}"),
                            })?,
                    );
            }
            "datatype" => {
                header.datatype = TckDatatype::parse(value)
                    .ok_or_else(|| TckError::UnknownDatatype(value.clone()))?;
            }
            "transform" => {
                header.transform = Some(parse_transform(value)?);
            }
            "mrtrix_version" => {
                header.mrtrix_version = Some(value.clone());
            }
            "file" => {
                header.file_path = Some(value.clone());
            }
            "comments" => {
                header.comments = Some(value.clone());
            }
            _ => {}
        }
    }

    header.fields = map;
    Ok(header)
}

/// Parse a 16-float `4×4` row-major transform matrix.
fn parse_transform(value: &str) -> Result<[[f64; 4]; 4], TckError> {
    let floats: Vec<f64> = value
        .split_whitespace()
        .filter_map(|s| s.parse::<f64>().ok())
        .collect();
    if floats.len() != 16 {
        return Err(TckError::InvalidTransform {
            count: floats.len(),
        });
    }
    let mut mat = [[0.0; 4]; 4];
    for row in 0..4 {
        for col in 0..4 {
            mat[row][col] = floats[row * 4 + col];
        }
    }
    Ok(mat)
}

/// Decode three scalars from a byte buffer according to `datatype`.
fn decode_point(dt: TckDatatype, buf: &[u8]) -> (f64, f64, f64) {
    match dt {
        TckDatatype::Float32LE => {
            let x = f32::from_le_bytes(buf[0..4].try_into().unwrap()) as f64;
            let y = f32::from_le_bytes(buf[4..8].try_into().unwrap()) as f64;
            let z = f32::from_le_bytes(buf[8..12].try_into().unwrap()) as f64;
            (x, y, z)
        }
        TckDatatype::Float32BE => {
            let x = f32::from_be_bytes(buf[0..4].try_into().unwrap()) as f64;
            let y = f32::from_be_bytes(buf[4..8].try_into().unwrap()) as f64;
            let z = f32::from_be_bytes(buf[8..12].try_into().unwrap()) as f64;
            (x, y, z)
        }
        TckDatatype::Float64LE => {
            let x = f64::from_le_bytes(buf[0..8].try_into().unwrap());
            let y = f64::from_le_bytes(buf[8..16].try_into().unwrap());
            let z = f64::from_le_bytes(buf[16..24].try_into().unwrap());
            (x, y, z)
        }
        TckDatatype::Float64BE => {
            let x = f64::from_be_bytes(buf[0..8].try_into().unwrap());
            let y = f64::from_be_bytes(buf[8..16].try_into().unwrap());
            let z = f64::from_be_bytes(buf[16..24].try_into().unwrap());
            (x, y, z)
        }
    }
}

/// Encode three `f64` scalars and write them using `datatype`.
fn encode_and_write(
    x: f64,
    y: f64,
    z: f64,
    dt: TckDatatype,
    writer: &mut impl Write,
) -> Result<(), TckError> {
    match dt {
        TckDatatype::Float32LE => {
            writer.write_all(&(x as f32).to_le_bytes())?;
            writer.write_all(&(y as f32).to_le_bytes())?;
            writer.write_all(&(z as f32).to_le_bytes())?;
        }
        TckDatatype::Float32BE => {
            writer.write_all(&(x as f32).to_be_bytes())?;
            writer.write_all(&(y as f32).to_be_bytes())?;
            writer.write_all(&(z as f32).to_be_bytes())?;
        }
        TckDatatype::Float64LE => {
            writer.write_all(&x.to_le_bytes())?;
            writer.write_all(&y.to_le_bytes())?;
            writer.write_all(&z.to_le_bytes())?;
        }
        TckDatatype::Float64BE => {
            writer.write_all(&x.to_be_bytes())?;
            writer.write_all(&y.to_be_bytes())?;
            writer.write_all(&z.to_be_bytes())?;
        }
    }
    Ok(())
}

/// Write a MRtrix3‑compatible weights sidecar file.
///
/// The MRtrix3 `.tck` format does not store per‑point scalars inline.
/// Instead, scalars such as FA or MD are written to a separate file using
/// the same binary layout as streamline data: one scalar value per point,
/// streamlines delimited by NaN, and end‑of‑file signalled by Inf.
///
/// `scalars` must have one entry per streamline; each inner `Box<[f32]>`
/// must contain one scalar per streamline point.  The output file can be
/// read by `tckmap` or `mrview` with `-tck_weights_in`.
///
/// # Errors
///
/// Returns [`TckError`] on write failure.
pub fn write_tck_weights(
    scalars: &[Box<[f32]>],
    datatype: TckDatatype,
    writer: &mut impl Write,
) -> Result<(), TckError> {
    writeln!(writer, "{}", TCK_MAGIC_LINE)?;
    writeln!(writer, "datatype: {}", datatype.as_str())?;
    writeln!(writer, "count: {}", scalars.len())?;
    writeln!(writer, "file: .")?;
    writeln!(writer, "END")?;

    for values in scalars {
        for &v in values.iter() {
            let v = v as f64;
            match datatype {
                TckDatatype::Float32LE => {
                    writer.write_all(&(v as f32).to_le_bytes())?;
                }
                TckDatatype::Float32BE => {
                    writer.write_all(&(v as f32).to_be_bytes())?;
                }
                TckDatatype::Float64LE => {
                    writer.write_all(&v.to_le_bytes())?;
                }
                TckDatatype::Float64BE => {
                    writer.write_all(&v.to_be_bytes())?;
                }
            }
        }
        match datatype {
            TckDatatype::Float32LE => {
                writer.write_all(&f32::NAN.to_le_bytes())?;
            }
            TckDatatype::Float32BE => {
                writer.write_all(&f32::NAN.to_be_bytes())?;
            }
            TckDatatype::Float64LE => {
                writer.write_all(&f64::NAN.to_le_bytes())?;
            }
            TckDatatype::Float64BE => {
                writer.write_all(&f64::NAN.to_be_bytes())?;
            }
        }
    }
    match datatype {
        TckDatatype::Float32LE => {
            writer.write_all(&f32::INFINITY.to_le_bytes())?;
        }
        TckDatatype::Float32BE => {
            writer.write_all(&f32::INFINITY.to_be_bytes())?;
        }
        TckDatatype::Float64LE => {
            writer.write_all(&f64::INFINITY.to_le_bytes())?;
        }
        TckDatatype::Float64BE => {
            writer.write_all(&f64::INFINITY.to_be_bytes())?;
        }
    }

    Ok(())
}

/// Read a MRtrix3 weights sidecar file back into per‑streamline scalar
/// values.
///
/// This is the inverse of [`write_tck_weights`].  Unlike the `.tck`
/// streamline reader (which expects 3 values per point), the weights
/// format uses one scalar per point with NaN delimiter and Inf barrier.
///
/// # Errors
///
/// Returns [`TckError`] for invalid magic, missing header fields,
/// premature EOF, or malformed data.
pub fn read_tck_weights(reader: impl Read) -> Result<Vec<Box<[f32]>>, TckError> {
    let mut buf_reader = BufReader::new(reader);

    let mut first_line = String::new();
    buf_reader.read_line(&mut first_line)?;
    if first_line.trim() != TCK_MAGIC_LINE {
        return Err(TckError::InvalidMagic(first_line.trim().to_string()));
    }

    let mut datatype = TckDatatype::Float32LE;
    loop {
        let mut line = String::new();
        buf_reader.read_line(&mut line)?;
        let trimmed = line.trim();
        if trimmed == TCK_END_LINE {
            break;
        }
        if trimmed.is_empty() {
            continue;
        }
        if let Some((key, value)) = parse_header_line(trimmed)
            && key == "datatype"
        {
            datatype = TckDatatype::parse(value)
                .ok_or_else(|| TckError::UnknownDatatype(value.to_string()))?;
        }
    }

    let mut result: Vec<Box<[f32]>> = Vec::new();
    let mut current: Vec<f32> = Vec::new();

    let scalar_size = match datatype {
        TckDatatype::Float32LE | TckDatatype::Float32BE => 4usize,
        TckDatatype::Float64LE | TckDatatype::Float64BE => 8usize,
    };
    let mut buf = vec![0u8; scalar_size];

    loop {
        match buf_reader.read_exact(&mut buf) {
            Ok(()) => {}
            Err(ref e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(TckError::Io(e)),
        }

        let val: f64 = decode_scalar(datatype, &buf);

        if val.is_infinite() {
            break;
        }
        if val.is_nan() {
            if !current.is_empty() {
                result.push(current.into_boxed_slice());
                current = Vec::new();
            }
            continue;
        }

        current.push(val as f32);
    }

    if !current.is_empty() {
        result.push(current.into_boxed_slice());
    }

    Ok(result)
}

/// Decode a single scalar from a byte buffer according to `datatype`.
fn decode_scalar(dt: TckDatatype, buf: &[u8]) -> f64 {
    match dt {
        TckDatatype::Float32LE => f32::from_le_bytes(buf[..4].try_into().unwrap()) as f64,
        TckDatatype::Float32BE => f32::from_be_bytes(buf[..4].try_into().unwrap()) as f64,
        TckDatatype::Float64LE => f64::from_le_bytes(buf[..8].try_into().unwrap()),
        TckDatatype::Float64BE => f64::from_be_bytes(buf[..8].try_into().unwrap()),
    }
}
