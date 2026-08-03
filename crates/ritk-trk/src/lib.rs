//! DSI Studio / TrackVis `.trk` tractogram format reader and writer.
//!
//! The TrackVis format is a binary interchange format for streamline
//! tractography results. It stores a fixed 1000-byte header followed by
//! per-streamline data blocks. This crate reads and writes that format,
//! converting between the on-disk voxel coordinate representation and
//! physical (RAS+mm) coordinates via the embedded affine transform.
//!
//! # Streamline representation
//!
//! Each streamline is returned as a [`gaia::Polyline<f64>`] whose points are
//! expressed in physical RAS+mm coordinates. The header affine
//! ([`TrkHeader::vox_to_ras`]) maps voxel indices to that space.
//!
//! # References
//!
//! * TrackVis file format specification (Diffusion Toolkit / DSI Studio)
//! * <http://trackvis.org/docs/?subsect=fileformat>

mod error;
mod io;
mod parse;
mod types;

pub use error::TrkError;
pub use types::{TrkHeader, TrkTractogram};

#[cfg(test)]
mod tests;
