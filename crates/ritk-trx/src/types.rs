#![forbid(unsafe_code)]
#![deny(missing_docs)]

use std::collections::HashMap;

use gaia::Polyline;
use serde::{Deserialize, Serialize};

/// Raw TRX output tuple: header, positions, offsets, per-vertex scalar maps.
pub type TrxRawOutput = (TrxHeader, Vec<u8>, Vec<u8>, HashMap<String, Vec<u8>>);

/// The parsed header from a TRX file's `header.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrxHeader {
    /// Total number of streamlines.
    #[serde(rename = "nb_streamlines")]
    pub nb_streamlines: u64,

    /// Total number of points across all streamlines.
    #[serde(rename = "nb_points")]
    pub nb_points: u64,

    /// Number of spatial dimensions (always 3 for tractography).
    #[serde(default = "default_dimensions")]
    pub dimensions: u32,

    /// Data type string for the positions array, e.g. `"float32"`.
    #[serde(default = "default_dtype")]
    pub dtype: String,

    /// Optional NIfTI reference image metadata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference: Option<TrxReference>,

    /// Per-vertex data arrays (e.g. FA, MD).
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub dpv: HashMap<String, TrxArrayDef>,

    /// Per-streamline data arrays (e.g. mean_length).
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub dps: HashMap<String, TrxArrayDef>,

    /// Per-group data.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub dpg: HashMap<String, TrxArrayDef>,

    /// Group definitions.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub groups: Vec<TrxGroup>,
}

/// Definition of a named data array stored alongside positions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrxArrayDef {
    /// Data type string, e.g. `"float32"`, `"int32"`.
    pub dtype: String,

    /// Number of components per element (e.g. 1 for scalar, 3 for vector).
    #[serde(default = "default_n_components")]
    pub n_components: u32,
}

/// NIfTI reference image metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrxReference {
    /// Path to the reference NIfTI file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,

    /// `4×4` row-major affine transform from voxel to physical space.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub affine: Option<[f64; 16]>,

    /// Voxel grid dimensions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<[u32; 3]>,

    /// Voxel size in mm.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voxel_sizes: Option<[f64; 3]>,
}

/// A named group of streamlines.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrxGroup {
    /// Group name.
    pub name: String,

    /// Streamline indices belonging to this group.
    pub indices: Vec<u64>,
}

fn default_dimensions() -> u32 {
    3
}

fn default_dtype() -> String {
    "float32".into()
}

fn default_n_components() -> u32 {
    1
}

impl Default for TrxHeader {
    fn default() -> Self {
        Self {
            nb_streamlines: 0,
            nb_points: 0,
            dimensions: 3,
            dtype: "float32".into(),
            reference: None,
            dpv: HashMap::new(),
            dps: HashMap::new(),
            dpg: HashMap::new(),
            groups: Vec::new(),
        }
    }
}

/// The full contents of a TRX file.
///
/// Streamlines are stored in physical millimetre coordinates.
/// The optional `dpv_data` map carries the raw binary payload for
/// every key declared in [`TrxHeader::dpv`] — one `Vec<u8>` per
/// per-vertex data array.
#[derive(Debug, Clone)]
pub struct TrxTractogram {
    /// Parsed header metadata.
    pub header: TrxHeader,

    /// Streamlines in file order, each a valid Gaia polyline.
    pub streamlines: Vec<Polyline<f64>>,

    /// Raw binary data for each per-vertex data array declared in
    /// `header.dpv`. Keys must match the header's dpv map; values
    /// are the encoded byte buffers (same dtype as the header declaration).
    pub dpv_data: HashMap<String, Vec<u8>>,
}
