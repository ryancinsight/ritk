//! MRtrix `.mif` image format I/O for RITK.
//!
//! This crate provides canonical single-source-of-truth implementations for
//! reading and writing the MRtrix3 `.mif` container format — a text header
//! followed by raw binary voxel data.  Both inline (single-file) and
//! detached (`file:` key → `.mif.dat`) layouts are supported.
//!
//! # Key APIs
//!
//! - [`read_mif`]: Read a `.mif` file as a native 3‑D image of the caller's
//!   sample type, with spatial metadata
//! - [`write_mif`]: Write an Image to a `.mif` file in its own sample type, with
//!   full transform encoding
//! - [`read_mif_series`]: Read an acquisition series as one image per volume
//! - [`write_mif_series`]: Write an acquisition series with interleaved frames
//!
//! # Sample types
//!
//! The `datatype` key names the stored type and, for types wider than a byte,
//! the byte order: `Int8`, `UInt8`, `Int16`, `UInt16`, `Int32`, `UInt32`,
//! `Int64`, `UInt64`, `Float32`, and `Float64`, each with an `LE` or `BE`
//! suffix when wider than one byte. The readers decode in that type and byte
//! order and convert to the requested `T` under a
//! [`Conversion`](ritk_codecs::sample::Conversion): `Exact` refuses any
//! conversion that could change a value and `Cast` converts with a warning.
//! The writers store `T` itself, little-endian, and convert nothing. `Bit` and
//! the complex types (`CFloat32`, `CFloat64`) hold something other than one
//! real scalar per voxel and are rejected. The `scaling` header key is not
//! interpreted.
//!
//! # Data offset
//!
//! The header's `file: <name> <offset>` key locates the voxels, and `<offset>`
//! is a byte count from the **start of the file** that holds them. For an
//! inline file (`file: . <offset>`) the offset lies after the `END` line:
//! MRtrix rounds it up to a multiple of 4 and zero-pads between `END` and the
//! data, and it refuses an inline offset of 0. The writers emit exactly that
//! layout; the readers refuse an inline offset inside the header or past the
//! end of the file. A detached file (`file: volume.dat <offset>`) is read from
//! its own byte `<offset>`. Source: MRtrix3 `docs/getting_started/image_data.rst`
//! (key `file`), `core/formats/mrtrix.cpp` (`MRtrix::create`), and
//! `core/formats/mrtrix_utils.cpp` (`get_mrtrix_file_path`).
//!
//! # Acquisition axis
//!
//! Multi‑frame `.mif` files carry a fourth dimension in the `dim` key whose
//! extent is the frame count (e.g., `dim: 128 128 60 33`).  In the
//! contiguous `layout: +0,+1,+2,+3`, axis 3 varies fastest — frames are
//! interleaved voxel‑by‑voxel.  A single‑frame file is an ordinary rank‑3
//! volume.
//!
//! The single‑volume and series entry points are asymmetric on purpose.  The
//! series reader accepts a rank‑3 file as a one‑volume series, because that
//! is what it is.  [`read_mif`] rejects a multi‑frame file rather than
//! returning volume 0, because a series has no correct single‑volume decoding
//! and quietly dropping the remaining frames would report success over lost
//! acquisition data.
//!
//! # Spatial convention
//!
//! - RITK tensors: `[Z, Y, X]` (depth, row, column)
//! - `.mif` storage: `[X, Y, Z]` with X as fastest‑varying raw axis
//! - The raw payload is already in RITK's flat order, so no permutation is needed
//! - The `transform` 4×4 affine maps voxel `[x,y,z]` to scanner coords in mm
//!
//! # Gradient scheme
//!
//! The `.mif` header may contain a `DW_scheme` key with the diffusion
//! gradient table. Extraction of this block is owned by
//! [`ritk_diffusion_scheme::read_mrtrix_scheme`]; callers use that crate
//! directly when reading a DWI series.

pub mod header;
pub mod reader;
pub mod writer;

pub use reader::{read_mif, read_mif_series, MifReader};
pub use writer::{write_mif, write_mif_series, MifWriter};

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_samples;
