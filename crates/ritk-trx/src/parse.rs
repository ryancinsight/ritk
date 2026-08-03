#![forbid(unsafe_code)]
#![deny(missing_docs)]

use std::collections::HashMap;

use gaia::Polyline;
use leto::geometry::Point3;

use crate::{TrxError, TrxHeader, TrxTractogram};

/// Expected number of spatial dimensions.
pub(crate) const TRX_DIMENSIONS: u32 = 3;

/// Bytes per float32 value.
pub(crate) const F32_BYTES: usize = 4;

/// Bytes per float64 value.
pub(crate) const F64_BYTES: usize = 8;

/// Bytes per uint64 value.
pub(crate) const U64_BYTES: usize = 8;

impl TrxTractogram {
    /// Read a TRX tractogram from in-memory header and raw array buffers.
    ///
    /// DPV data is not read; use [`from_raw_with_dpv`] for round-trip
    /// fidelity.
    ///
    /// # Errors
    ///
    /// Returns [`TrxError`] for array length mismatches, invalid offsets,
    /// unsupported dtypes, or invalid geometry.
    pub fn from_raw(
        header: &TrxHeader,
        positions_raw: &[u8],
        offsets_raw: &[u8],
    ) -> Result<Self, TrxError> {
        Self::from_raw_with_dpv(header, positions_raw, offsets_raw, HashMap::new())
    }

    /// Read a TRX tractogram from in-memory buffers including DPV data.
    ///
    /// Unlike [`from_raw`], this method accepts a `dpv_data` map so that
    /// per-vertex arrays survive a round-trip through encode / decode.
    ///
    /// # Errors
    ///
    /// Returns [`TrxError`] for array length mismatches, invalid offsets,
    /// unsupported dtypes, or invalid geometry.
    pub fn from_raw_with_dpv(
        header: &TrxHeader,
        positions_raw: &[u8],
        offsets_raw: &[u8],
        dpv_data: HashMap<String, Vec<u8>>,
    ) -> Result<Self, TrxError> {
        debug_assert!(
            dpv_data.keys().all(|k| header.dpv.contains_key(k)),
            "dpv_data contains keys not declared in header.dpv"
        );
        debug_assert!(
            header.dpv.keys().all(|k| dpv_data.contains_key(k)),
            "header.dpv declares keys not present in dpv_data"
        );

        let expected_pos_elements = header.nb_points * TRX_DIMENSIONS as u64;
        let pos_element_count = match header.dtype.as_str() {
            "float32" => (positions_raw.len() / F32_BYTES) as u64,
            "float64" => (positions_raw.len() / F64_BYTES) as u64,
            other => return Err(TrxError::UnsupportedDtype(other.into())),
        };
        if pos_element_count != expected_pos_elements {
            return Err(TrxError::PositionsLengthMismatch {
                expected: expected_pos_elements,
                got: pos_element_count,
            });
        }

        let expected_offsets = header.nb_streamlines + 1;
        let offset_count = (offsets_raw.len() / U64_BYTES) as u64;
        if offset_count != expected_offsets {
            return Err(TrxError::OffsetsLengthMismatch {
                expected: expected_offsets,
                got: offset_count,
            });
        }

        let offsets: Vec<u64> = offsets_raw
            .chunks_exact(U64_BYTES)
            .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
            .collect();

        let sentinel = offsets[header.nb_streamlines as usize];
        if sentinel != header.nb_points {
            return Err(TrxError::SentinelMismatch {
                expected: header.nb_points,
                got: sentinel,
            });
        }

        let streamlines = build_streamlines(header, positions_raw, &offsets)?;

        Ok(Self {
            header: header.clone(),
            streamlines,
            dpv_data,
        })
    }
}

/// Build streamlines from raw position/offset data.
pub(crate) fn build_streamlines(
    header: &TrxHeader,
    positions_raw: &[u8],
    offsets: &[u64],
) -> Result<Vec<Polyline<f64>>, TrxError> {
    let n_streamlines = header.nb_streamlines as usize;
    let mut streamlines = Vec::with_capacity(n_streamlines);

    for index in 0..n_streamlines {
        let start = offsets[index] as usize;
        let end = offsets[index + 1] as usize;

        if start > end || end > header.nb_points as usize {
            return Err(TrxError::InvalidOffset {
                index,
                value: end as u64,
                prev: start as u64,
                max: header.nb_points,
            });
        }

        let n_points = end - start;
        if n_points == 0 {
            continue;
        }

        let mut points = Vec::with_capacity(n_points);
        match header.dtype.as_str() {
            "float32" => {
                let base = start * TRX_DIMENSIONS as usize * F32_BYTES;
                for point_index in 0..n_points {
                    let off = base + point_index * TRX_DIMENSIONS as usize * F32_BYTES;
                    let x =
                        f32::from_le_bytes(positions_raw[off..off + 4].try_into().unwrap()) as f64;
                    let y = f32::from_le_bytes(positions_raw[off + 4..off + 8].try_into().unwrap())
                        as f64;
                    let z = f32::from_le_bytes(positions_raw[off + 8..off + 12].try_into().unwrap())
                        as f64;
                    if !x.is_finite() || !y.is_finite() || !z.is_finite() {
                        return Err(TrxError::NonFiniteCoordinate { index, point_index });
                    }
                    points.push(Point3::new(x, y, z));
                }
            }
            "float64" => {
                let base = start * TRX_DIMENSIONS as usize * F64_BYTES;
                for point_index in 0..n_points {
                    let off = base + point_index * TRX_DIMENSIONS as usize * F64_BYTES;
                    let x = f64::from_le_bytes(positions_raw[off..off + 8].try_into().unwrap());
                    let y =
                        f64::from_le_bytes(positions_raw[off + 8..off + 16].try_into().unwrap());
                    let z =
                        f64::from_le_bytes(positions_raw[off + 16..off + 24].try_into().unwrap());
                    if !x.is_finite() || !y.is_finite() || !z.is_finite() {
                        return Err(TrxError::NonFiniteCoordinate { index, point_index });
                    }
                    points.push(Point3::new(x, y, z));
                }
            }
            other => return Err(TrxError::UnsupportedDtype(other.into())),
        }

        let polyline =
            Polyline::new(points).map_err(|source| TrxError::InvalidPolyline { index, source })?;
        streamlines.push(polyline);
    }

    Ok(streamlines)
}
