#![forbid(unsafe_code)]
#![deny(missing_docs)]

use std::io::{Read, Write};

use gaia::Polyline;
use leto::geometry::Point3;

use crate::parse::{
    TRK_HEADER_SIZE, TRK_MAGIC, apply_affine, encode_header, invert_affine, parse_header,
    read_exact,
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

        let magic: [u8; 6] = header_buf[..6].try_into().unwrap();
        if &magic != TRK_MAGIC {
            return Err(TrkError::InvalidMagic {
                got: magic[..5].try_into().unwrap(),
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

        let n_scalars = header.n_scalars.max(0) as usize;
        let n_properties = header.n_properties.max(0) as usize;
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
            {
                let byte_len = total_floats * 4;
                let mut byte_buf = vec![0u8; byte_len];
                read_exact(reader, &mut byte_buf, &mut offset)?;
                for chunk in byte_buf.chunks_exact(4) {
                    point_buf.push(f32::from_le_bytes(chunk.try_into().unwrap()));
                }
            }

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
                let prop_byte_len = n_properties * 4;
                let mut prop_bytes = vec![0u8; prop_byte_len];
                read_exact(reader, &mut prop_bytes, &mut offset)?;
                let mut prop = Vec::with_capacity(n_properties);
                for chunk in prop_bytes.chunks_exact(4) {
                    prop.push(f32::from_le_bytes(chunk.try_into().unwrap()));
                }
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
        let header_buf = encode_header(&self.header);
        writer
            .write_all(&header_buf)
            .map_err(|_| TrkError::UnexpectedEof { offset: 0 })?;

        let inv = invert_affine(&self.header.vox_to_ras);

        let n_scalars = self.header.n_scalars.max(0) as usize;
        let n_properties = self.header.n_properties.max(0) as usize;
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

            let byte_buf: Vec<u8> = buf.iter().flat_map(|v| v.to_le_bytes()).collect();
            writer
                .write_all(&byte_buf)
                .map_err(|_| TrkError::UnexpectedEof { offset: 0 })?;

            if n_properties > 0 {
                let prop_bytes: Vec<u8> = self.properties[index]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect();
                writer
                    .write_all(&prop_bytes)
                    .map_err(|_| TrkError::UnexpectedEof { offset: 0 })?;
            }
        }

        Ok(())
    }
}
