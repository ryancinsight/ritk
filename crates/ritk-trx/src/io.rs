#![forbid(unsafe_code)]
#![deny(missing_docs)]

use std::collections::HashMap;
use std::path::Path;

use crate::parse::{F32_BYTES, F64_BYTES, U64_BYTES};
use crate::{TrxError, TrxHeader, TrxTractogram};

/// The serialized parts of a TRX file: header, offsets, positions, and the
/// per-vertex data arrays keyed by name.
type RawTrx = (TrxHeader, Vec<u8>, Vec<u8>, HashMap<String, Vec<u8>>);

impl TrxTractogram {
    /// Read a TRX file from a directory path.
    ///
    /// Expects `header.json` and `positions.raw` / `offsets.raw` in the
    /// directory. If the header declares per-vertex data arrays in its
    /// `dpv` map, the corresponding `dpv/<name>.raw` files are also read.
    ///
    /// # Errors
    ///
    /// Returns [`TrxError`] for missing files, JSON parse errors, array
    /// length mismatches, unsupported dtypes, or invalid geometry.
    pub fn read_dir(path: impl AsRef<Path>) -> Result<Self, TrxError> {
        let base = path.as_ref();

        let header_json = std::fs::read_to_string(base.join("header.json"))?;
        let header: TrxHeader = serde_json::from_str(&header_json)?;

        let positions_raw = std::fs::read(base.join("positions.raw"))?;
        let offsets_raw = std::fs::read(base.join("offsets.raw"))?;

        let mut dpv_data: HashMap<String, Vec<u8>> = HashMap::new();
        if !header.dpv.is_empty() {
            let dpv_dir = base.join("dpv");
            for name in header.dpv.keys() {
                let path = dpv_dir.join(format!("{name}.raw"));
                let data = std::fs::read(&path).map_err(|error| {
                    if error.kind() == std::io::ErrorKind::NotFound {
                        std::io::Error::new(
                            error.kind(),
                            format!("DPV file missing: {}", path.display()),
                        )
                    } else {
                        error
                    }
                })?;
                dpv_data.insert(name.clone(), data);
            }
        }

        Self::from_raw_with_dpv(&header, &positions_raw, &offsets_raw, dpv_data)
    }

    /// Write a TRX tractogram to a directory.
    ///
    /// Creates `header.json`, `positions.raw`, and `offsets.raw`.
    /// If DPV data is present, writes `dpv/<name>.raw` for every entry.
    ///
    /// # Errors
    ///
    /// Returns [`TrxError`] on I/O or serialization failure.
    pub fn write_dir(&self, path: impl AsRef<Path>) -> Result<(), TrxError> {
        let base = path.as_ref();
        std::fs::create_dir_all(base)?;

        let (positions_raw, offsets_raw) = self.encode_raw()?;

        let header = self.build_header();
        let header_json = serde_json::to_string_pretty(&header)?;
        std::fs::write(base.join("header.json"), header_json)?;
        std::fs::write(base.join("positions.raw"), &positions_raw)?;
        std::fs::write(base.join("offsets.raw"), &offsets_raw)?;

        if !self.dpv_data.is_empty() {
            let dpv_dir = base.join("dpv");
            std::fs::create_dir_all(&dpv_dir)?;
            for (name, data) in &self.dpv_data {
                std::fs::write(dpv_dir.join(format!("{name}.raw")), data)?;
            }
        }

        Ok(())
    }

    /// Write a TRX tractogram to in-memory buffers.
    ///
    /// Returns `(header, positions_raw, offsets_raw, dpv_data)`.
    /// The `dpv_data` map is the same as [`Self::dpv_data`]; callers
    /// that need per-vertex data should use it directly rather than
    /// relying on the returned map.
    ///
    /// # Errors
    ///
    /// Returns [`TrxError`] on encoding failure.
    pub fn to_raw(&self) -> Result<RawTrx, TrxError> {
        debug_assert!(
            self.dpv_data
                .keys()
                .all(|key| self.header.dpv.contains_key(key)),
            "dpv_data contains keys not declared in header.dpv"
        );
        debug_assert!(
            self.header
                .dpv
                .keys()
                .all(|key| self.dpv_data.contains_key(key)),
            "header.dpv declares keys not present in dpv_data"
        );

        let (positions_raw, offsets_raw) = self.encode_raw()?;
        let header = self.build_header();
        Ok((header, positions_raw, offsets_raw, self.dpv_data.clone()))
    }

    fn build_header(&self) -> TrxHeader {
        let nb_streamlines = self.streamlines.len() as u64;
        let nb_points: u64 = self
            .streamlines
            .iter()
            .map(|streamline| streamline.len() as u64)
            .sum();

        TrxHeader {
            nb_streamlines,
            nb_points,
            dimensions: 3,
            dtype: self.header.dtype.clone(),
            reference: self.header.reference.clone(),
            dpv: self.header.dpv.clone(),
            dps: self.header.dps.clone(),
            dpg: self.header.dpg.clone(),
            groups: self.header.groups.clone(),
        }
    }

    fn encode_raw(&self) -> Result<(Vec<u8>, Vec<u8>), TrxError> {
        let dtype = self.header.dtype.as_str();

        let nb_points: usize = self
            .streamlines
            .iter()
            .map(|streamline| streamline.len())
            .sum();

        let mut positions_raw: Vec<u8> = match dtype {
            "float32" => Vec::with_capacity(nb_points * 3 * F32_BYTES),
            "float64" => Vec::with_capacity(nb_points * 3 * F64_BYTES),
            other => return Err(TrxError::UnsupportedDtype(other.into())),
        };

        let mut offsets_raw: Vec<u8> = Vec::with_capacity((self.streamlines.len() + 1) * U64_BYTES);
        let mut cursor: u64 = 0;

        for polyline in &self.streamlines {
            offsets_raw.extend_from_slice(&cursor.to_le_bytes());

            for point in polyline.points() {
                match dtype {
                    "float32" => {
                        positions_raw.extend_from_slice(&(point.x as f32).to_le_bytes());
                        positions_raw.extend_from_slice(&(point.y as f32).to_le_bytes());
                        positions_raw.extend_from_slice(&(point.z as f32).to_le_bytes());
                    }
                    "float64" => {
                        positions_raw.extend_from_slice(&point.x.to_le_bytes());
                        positions_raw.extend_from_slice(&point.y.to_le_bytes());
                        positions_raw.extend_from_slice(&point.z.to_le_bytes());
                    }
                    _ => unreachable!(),
                }
                cursor += 1;
            }
        }

        offsets_raw.extend_from_slice(&cursor.to_le_bytes());

        Ok((positions_raw, offsets_raw))
    }
}
