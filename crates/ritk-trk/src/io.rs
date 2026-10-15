#![forbid(unsafe_code)]
#![deny(missing_docs)]

use std::io::{Read, Write};

use gaia::Polyline;
use leto::geometry::Point3;

use crate::parse::{
    apply_affine, encode_header, invert_affine, parse_header, read_exact, MAX_TRK_FIELD_COUNT,
    TRK_HEADER_SIZE, TRK_MAGIC,
};
use crate::{TrkError, TrkTractogram};

impl TrkTractogram {
    /// Read a `.trk` file from any byte source.
    ///
    /// # Errors
    ///
    /// Returns [`TrkError`] for invalid magic, wrong header size, premature
    /// EOF, unreasonable point counts, non-finite coordinates, or invalid
    /// Gaia geometry.
    pub fn read(reader: &mut impl Read) -> Result<Self, TrkError> {
        let mut offset: usize = 0;

        let mut header_buf = [0u8; TRK_HEADER_SIZE];
        read_exact(reader, &mut header_buf, &mut offset)?;

        let magic: [u8; 6] = header_buf[..6].try_into().expect("fixed-width byte field");
        if &magic != TRK_MAGIC {
            return Err(TrkError::InvalidMagic {
                got: magic[..5].try_into().expect("fixed-width byte field"),
            });
        }

        let header = parse_header(&header_buf);
        if header.hdr_size != TRK_HEADER_SIZE as i32 {
            return Err(TrkError::InvalidHeaderSize {
                value: header.hdr_size,
            });
        }

        let n_count = header.n_count;
        if n_count < 0 {
            return Err(TrkError::InvalidStreamlineCount { count: n_count });
        }

        // `n_scalars` and `n_properties` index the header's fixed name tables
        // (`scalar_name[10][20]`, `property_name[10][20]`), so a value outside
        // that range is malformed rather than merely unusual. Rejecting it also
        // bounds the per-point stride `3 + n_scalars`: it reaches the decode
        // buffer size as `n_points * stride`, where `n_points` is already capped
        // at 100_000. Without the check a crafted 1 KiB header asked for ~13 GB.
        if !(0..=MAX_TRK_FIELD_COUNT).contains(&header.n_scalars) {
            return Err(TrkError::InvalidScalarCount {
                count: header.n_scalars,
            });
        }
        if !(0..=MAX_TRK_FIELD_COUNT).contains(&header.n_properties) {
            return Err(TrkError::InvalidPropertyCount {
                count: header.n_properties,
            });
        }

        // `n_count` is a claim from a 1000-byte header, not a fact about the
        // file, so nothing is reserved from it: `Vec::with_capacity(n_count)`
        // let a short malformed file demand gigabytes before a single record
        // was read. The vectors grow as records genuinely decode, which bounds
        // them by the real input — each iteration must read at least the
        // 4-byte point count, so the reader reaching EOF ends the loop.
        let n_streamlines = n_count as usize;
        let mut streamlines = Vec::new();
        let mut scalars = Vec::new();
        let mut properties = Vec::new();

        let n_scalars = header.n_scalars as usize;
        let n_properties = header.n_properties as usize;
        let stride = 3 + n_scalars;

        let mut point_buf: Vec<f32> = Vec::new();

        for index in 0..n_streamlines {
            let mut count_buf = [0u8; 4];
            read_exact(reader, &mut count_buf, &mut offset)?;
            let n_points = i32::from_le_bytes(count_buf);
            if !(0..=100_000).contains(&n_points) {
                return Err(TrkError::InvalidPointCount {
                    index,
                    count: n_points,
                });
            }
            let n_points = n_points as usize;

            let total_floats = n_points * stride;
            point_buf.clear();
            point_buf
                .try_reserve_exact(total_floats)
                .map_err(|_| TrkError::InvalidPointCount {
                    index,
                    count: n_points as i32,
                })?;
            read_f32_values(reader, total_floats, &mut point_buf, &mut offset)?;

            if n_scalars > 0 {
                let mut streamline_scalars: Vec<f32> = Vec::with_capacity(n_points * n_scalars);
                for p in 0..n_points {
                    let base = p * stride + 3;
                    streamline_scalars.extend_from_slice(&point_buf[base..base + n_scalars]);
                }
                scalars.push(streamline_scalars.into_boxed_slice());
            }

            let mut points = Vec::with_capacity(n_points);
            for p in 0..n_points {
                let base = p * stride;
                let x = point_buf[base];
                let y = point_buf[base + 1];
                let z = point_buf[base + 2];
                if !x.is_finite() || !y.is_finite() || !z.is_finite() {
                    return Err(TrkError::NonFiniteCoordinate {
                        index,
                        point_index: p,
                    });
                }
                let (px, py, pz) = apply_affine(&header.vox_to_ras, x, y, z);
                points.push(Point3::new(px, py, pz));
            }

            if n_properties > 0 {
                let mut prop = Vec::with_capacity(n_properties);
                read_f32_values(reader, n_properties, &mut prop, &mut offset)?;
                properties.push(prop.into_boxed_slice());
            }

            let polyline = Polyline::new(points)
                .map_err(|source| TrkError::InvalidPolyline { index, source })?;
            streamlines.push(polyline);
        }

        Ok(Self {
            header,
            streamlines,
            scalars,
            properties,
        })
    }

    /// Write a tractogram to a `.trk` file.
    ///
    /// Streamlines are expected in physical RAS+mm coordinates. The inverse
    /// of the header affine is applied so that the on-disk representation
    /// uses voxel-index coordinates.
    ///
    /// # Errors
    ///
    /// Returns [`TrkError`] or an [`std::io::Error`] on write failure.
    pub fn write(&self, writer: &mut impl Write) -> Result<(), TrkError> {
        // The header's name tables cannot describe more fields than this, so a
        // value outside the range has no valid on-disk encoding: reject it here
        // rather than emit a file whose declared stride disagrees with its
        // names, which the reader above would then refuse.
        if !(0..=MAX_TRK_FIELD_COUNT).contains(&self.header.n_scalars) {
            return Err(TrkError::InvalidScalarCount {
                count: self.header.n_scalars,
            });
        }
        if !(0..=MAX_TRK_FIELD_COUNT).contains(&self.header.n_properties) {
            return Err(TrkError::InvalidPropertyCount {
                count: self.header.n_properties,
            });
        }

        let header_buf = encode_header(&self.header);
        writer
            .write_all(&header_buf)
            .map_err(|_| TrkError::UnexpectedEof { offset: 0 })?;

        let inv = invert_affine(&self.header.vox_to_ras);

        let n_scalars = self.header.n_scalars as usize;
        let n_properties = self.header.n_properties as usize;
        let stride = 3 + n_scalars;

        for (index, polyline) in self.streamlines.iter().enumerate() {
            let pts = polyline.points();
            let n_points = pts.len() as i32;
            writer
                .write_all(&n_points.to_le_bytes())
                .map_err(|_| TrkError::UnexpectedEof { offset: 0 })?;

            let mut buf: Vec<f32> = Vec::with_capacity(pts.len() * stride);
            for (point_index, point) in pts.iter().enumerate() {
                let (vx, vy, vz) =
                    apply_affine(&inv, point.x as f32, point.y as f32, point.z as f32);
                buf.push(vx as f32);
                buf.push(vy as f32);
                buf.push(vz as f32);

                if n_scalars > 0 {
                    let base = point_index * n_scalars;
                    buf.extend_from_slice(&self.scalars[index][base..base + n_scalars]);
                }
            }

            write_f32_values(writer, &buf)?;

            if n_properties > 0 {
                write_f32_values(writer, &self.properties[index])?;
            }
        }

        Ok(())
    }
}

/// Read `count` little-endian `f32` values from `reader` into `out`.
///
/// `out` must already hold room for `count` values — the caller reserves it
/// from the confirmed payload size. Values pass through a fixed stack buffer,
/// so the decode holds exactly one payload-sized heap allocation instead of a
/// second byte copy of the same size; the peak for a streamline's point data is
/// therefore its own buffer plus 4 KiB, not twice its own buffer.
fn read_f32_values(
    reader: &mut impl Read,
    count: usize,
    out: &mut Vec<f32>,
    offset: &mut usize,
) -> Result<(), TrkError> {
    /// Stack scratch for one read; a multiple of four so chunks split on values.
    const CHUNK_BYTES: usize = 4096;
    let mut buf = [0u8; CHUNK_BYTES];
    let values_per_chunk = CHUNK_BYTES / 4;

    let mut remaining = count;
    while remaining > 0 {
        let chunk_values = remaining.min(values_per_chunk);
        let chunk_bytes = chunk_values * 4;
        read_exact(reader, &mut buf[..chunk_bytes], offset)?;
        for chunk in buf[..chunk_bytes].chunks_exact(4) {
            out.push(f32::from_le_bytes(
                chunk.try_into().expect("fixed-width byte field"),
            ));
        }
        remaining -= chunk_values;
    }
    Ok(())
}

/// Write `values` as little-endian `f32` bytes to `writer`.
///
/// Serializing through a fixed stack buffer avoids materializing a byte copy of
/// the whole payload, so writing a streamline holds no second allocation the
/// size of its point data — the writer streams the same way the reader does.
fn write_f32_values(writer: &mut impl Write, values: &[f32]) -> Result<(), TrkError> {
    /// Values serialized per write; bounded so the scratch buffer stays on the stack.
    const CHUNK_VALUES: usize = 1024;
    let mut buf = [0u8; CHUNK_VALUES * 4];

    for chunk in values.chunks(CHUNK_VALUES) {
        let mut len = 0;
        for value in chunk {
            buf[len..len + 4].copy_from_slice(&value.to_le_bytes());
            len += 4;
        }
        writer
            .write_all(&buf[..len])
            .map_err(|_| TrkError::UnexpectedEof { offset: 0 })?;
    }
    Ok(())
}
