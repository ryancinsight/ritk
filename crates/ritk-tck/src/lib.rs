//! MRtrix3 `.tck` tractogram format reader and writer.
//!
//! The MRtrix3 track format is a text-header / binary-data interchange
//! format for streamline tractography results.  A human-readable header
//! section of `key: value` pairs is terminated by the line `END`, after
//! which binary streamline data follows.  This crate reads and writes
//! that format.
//!
//! # Streamline representation
//!
//! Each streamline is returned as a [`gaia::Polyline<f64>`] whose points
//! are expressed in scanner-space millimetre coordinates (the native
//! `.tck` coordinate system).  The optional `transform` header entry is
//! stored in [`TckHeader::transform`] as a `4×4` row-major matrix for
//! consumers that need to map to/from voxel space.
//!
//! # Format notes
//!
//! * Streamlines are separated by a delimiter triplet of three NaN values
//!   of the declared datatype.
//! * The end-of-file is signalled by a barrier triplet of three Inf values
//!   (or physical EOF).
//! * Per-point scalars and per-streamline properties are not natively
//!   stored in the `.tck` format.  Use [`write_tck_weights`] to write a
//!   MRtrix3‑compatible sidecar weights file (same binary layout as
//!   streamline data but with one scalar per point).
//!
//! # References
//!
//! * MRtrix3 track file format (file_tck.h / file_base.h)
//! * <https://mrtrix.readthedocs.io/>

#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod io;
mod types;

pub use io::{read_tck_weights, write_tck_weights, TckError};
pub use types::{TckDatatype, TckHeader, TckTractogram};

#[cfg(test)]
mod tests;
