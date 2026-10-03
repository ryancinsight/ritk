//! MINC2 (.mnc / .mnc2) reader and writer for 3-D medical images.
//!
//! # Format
//!
//! MINC2 is the HDF5-based successor to the NetCDF-based MINC1 format,
//! developed at the Montreal Neurological Institute (MNI). It is the
//! standard format for the MNI152 template and ANTs atlas workflows.
//!
//! # Typical HDF5 Layout
//!
//! ```text
//! / (root)
//!   └── minc-2.0/ (group)
//!       ├── dimensions/ (group)
//!       │   ├── xspace (dimension object)
//!       │   │   Attributes: start, step, length, direction_cosines, units
//!       │   ├── yspace (same attributes)
//!       │   └── zspace (same attributes)
//!       ├── info/ (group; optional scan and subject metadata)
//!       └── image/ (group)
//!           └── 0/ (group)
//!               ├── image (N-D dataset: volume data)
//!               │   Attributes: dimorder, valid_range, signtype, complete
//!               ├── image-max (optional scaling dataset)
//!               └── image-min (optional scaling dataset)
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
//! The reader keeps the stored voxel type: contiguous MINC2 `u8`, `i8`, `u16`,
//! `i16`, `u32`, `i32`, `u64`, `i64`, `f32`, and `f64` datasets, in either byte
//! order, decode in that type and convert to the caller's `T` under a
//! [`Conversion`](ritk_codecs::sample::Conversion): [`Exact`] refuses any
//! conversion that could change a value, [`Cast`] converts and warns.
//!
//! Integer images carry the MINC pixel conversion from the stored
//! `valid_range` to the scalar or per-slice `image-min` / `image-max` real
//! range. It is one [`RealValueMap`], `(stored - valid_min) * slope +
//! image_min`, per slice along the first spatial axis, applied in `T`;
//! [`read_minc`] returns real intensities and
//! [`read_minc_stored`] returns the stored samples with the unapplied maps, the
//! only way to read an image whose maps are not the identity into an integer
//! `T`. Values outside `valid_range` are rejected because the public image
//! contract has no missing-value mask. Floating-point datasets bypass the map.
//!
//! The writer stores the image's own type, little-endian: `u8`, `i8`, `u16`,
//! `i16`, `u32`, `i32`, `f32`, or `f64`. MINC2 has no 64-bit integer voxel
//! type, so the writer refuses `u64` and `i64` before creating a file. An
//! integer image is written with `image-min` and `image-max` equal to its
//! type's range, the identity map, so it reads back unchanged.
//!
//! This follows the MINC
//! [pixel-conversion contract](https://www.bic.mni.mcgill.ca/software/minc/prog_guide/node19.html)
//! and [standard image variables](https://www.bic.mni.mcgill.ca/software/minc/minc1_format/node5.html).
//!
//! [`Exact`]: ritk_codecs::sample::Exact
//! [`Cast`]: ritk_codecs::sample::Cast

pub mod attrs;
mod datatype;
mod dimension;
pub(crate) mod hdf5_binary;
mod image_ranges;
mod payload;
pub mod reader;
mod real_map;
#[cfg(test)]
mod scaled_fixture;
mod scaling;
pub mod spatial;
pub mod writer;

pub use dimension::{MincDimension, DIMENSIONS_PATH, IMAGE_PATH, SPATIAL_DIM_NAMES};
pub use reader::{read_minc, read_minc_stored, MincReader};
pub use real_map::RealValueMap;
pub use writer::{write_minc, MincWriter};

#[cfg(test)]
mod tests_reader_errors;
#[cfg(test)]
mod tests_samples;
