//! MINC2 (.mnc / .mnc2) reader and writer for 3-D medical images.
//!
//! # Format
//!
//! MINC2 is the HDF5-based successor to the NetCDF-based MINC1 format,
//! developed at the Montreal Neurological Institute (MNI). It is the
//! standard format for the MNI152 template and ANTs atlas workflows.
//!
//! # HDF5 Layout
//!
//! ```text
//! / (root)
//!   Attributes: ident, minc_version, history
//!   └── minc-2.0/ (group)
//!       ├── dimensions/ (group)
//!       │   ├── xspace (group)
//!       │   │   Attributes: start, step, length, direction_cosines, units
//!       │   ├── yspace (group, same attributes)
//!       │   └── zspace (group, same attributes)
//!       └── image/ (group)
//!           └── 0/ (group)
//!               ├── image (N-D dataset: volume data)
//!               │   Attributes: dimorder, valid_range, signtype, complete
//!               ├── image-max (dataset: per-slice maximum)
//!               └── image-min (dataset: per-slice minimum)
//! ```
//!
//! # Spatial Metadata
//!
//! Each spatial dimension group (`xspace`, `yspace`, `zspace`) carries:
//!
//! | Attribute           | Type      | Semantics                              |
//! |---------------------|-----------|----------------------------------------|
//! | `start`             | `f64`     | Physical origin coordinate (mm)        |
//! | `step`              | `f64`     | Voxel spacing (mm)                     |
//! | `length`            | `i32`     | Number of voxels along this axis       |
//! | `direction_cosines` | `[f64;3]` | Column of the 3×3 direction matrix     |
//!
//! The `dimorder` attribute on `/minc-2.0/image/0/image` (e.g.,
//! `"zspace,yspace,xspace"`) defines how dataset array dimensions map
//! to spatial axes.
//!
//! # Origin / Spacing / Direction Derivation
//!
//! Given dimension metadata for axes ordered by `dimorder`:
//!
//! ```text
//! spacing = [step_dim0, step_dim1, step_dim2]
//! origin  = [start_dim0, start_dim1, start_dim2]
//! direction = [dir_cosines_dim0 | dir_cosines_dim1 | dir_cosines_dim2]
//! ```
//!
//! The RITK tensor shape `[nz, ny, nx]` is derived from the dimorder
//! mapping: the first dimorder entry maps to tensor axis 0, etc.
//!
//! # Data Type Handling
//!
//! The reader converts all MINC2 voxel data types (u8, i8, u16, i16,
//! u32, i32, f32, f64) to `f32` for the RITK tensor. Integer data
//! may be normalized using the `image-min` / `image-max` per-slice
//! datasets and `valid_range` attribute when present.

pub mod attrs;
pub mod convert;
pub(crate) mod hdf5_binary;
pub mod reader;
pub mod spatial;
pub mod writer;

pub use reader::native;
pub use reader::{read_minc, MincReader};
pub use writer::{write_minc, MincWriter};

/// MINC2 HDF5 path to the dimensions group.
pub const DIMENSIONS_PATH: &str = "minc-2.0/dimensions";

/// MINC2 HDF5 path to the image dataset.
pub const IMAGE_PATH: &str = "minc-2.0/image/0/image";

/// Recognized spatial dimension names in canonical order (x, y, z).
pub const SPATIAL_DIM_NAMES: [&str; 3] = ["xspace", "yspace", "zspace"];

/// Parsed metadata for a single MINC2 spatial dimension.
#[derive(Debug, Clone)]
pub struct MincDimension {
    /// Dimension name (e.g. "xspace", "yspace", "zspace").
    pub name: String,
    /// Physical start coordinate in mm.
    pub start: f64,
    /// Voxel spacing in mm.
    pub step: f64,
    /// Number of voxels along this axis.
    pub length: usize,
    /// Direction cosine vector (3 components).
    pub direction_cosines: [f64; 3],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minc_dimension_fields_accessible() {
        let dim = MincDimension {
            name: String::from("xspace"),
            start: -64.0,
            step: 1.0,
            length: 128,
            direction_cosines: [1.0, 0.0, 0.0],
        };
        assert_eq!(dim.name, "xspace");
        assert!((dim.start - (-64.0)).abs() < 1e-10);
        assert!((dim.step - 1.0).abs() < 1e-10);
        assert_eq!(dim.length, 128);
        assert_eq!(dim.direction_cosines, [1.0, 0.0, 0.0]);
    }
}
