//! MRtrix `.mif` image format I/O for RITK.
//!
//! This crate provides canonical single-source-of-truth implementations for
//! reading and writing the MRtrix3 `.mif` container format — a text header
//! followed by raw binary voxel data.  Both inline (single-file) and
//! detached (`file:` key → `.mif.dat`) layouts are supported.
//!
//! # Key APIs
//!
//! - [`read_mif`]: Read a `.mif` file as a native 3‑D image with spatial metadata
//! - [`write_mif`]: Write an Image to a `.mif` file with full transform encoding
//! - [`read_mif_series`]: Read an acquisition series as one image per volume
//! - [`write_mif_series`]: Write an acquisition series with interleaved frames
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
//! gradient table.  Extraction of this block is owned by
//! [`ritk_diffusion_scheme::mrtrix`], which this crate re‑exports for
//! convenience when reading a DWI series.

pub(crate) mod decode;
pub mod header;
pub mod reader;
pub mod writer;

pub use reader::{read_mif, read_mif_series, MifReader};
pub use writer::{write_mif, write_mif_series, MifWriter};

#[cfg(test)]
mod tests;
