//! TRX (Tractography Reference eXchange) format reader and writer.
//!
//! The TRX format stores tractography data as a directory containing a
//! JSON header (`header.json`) and raw binary arrays (`.raw` files). This
//! crate reads and writes that format, converting between the on-disk
//! flat position/offset representation and Gaia polyline geometry.
//!
//! # Streamline representation
//!
//! Each streamline is returned as a [`gaia::Polyline<f64>`] whose points
//! are expressed in physical millimetre coordinates. The optional NIfTI
//! reference affine is stored in [`TrxHeader::reference`].
//!
//! # Format notes
//!
//! * Positions are stored as a flat `(nb_points, 3)` array with an offset
//!   array (`nb_streamlines + 1`) indexing into it.
//! * Per-vertex data (DPV) and per-streamline data (DPS) are stored as
//!   separate binary files with dtypes declared in the header.
//! * Groups allow hierarchical organisation of streamlines.
//! * The directory layout is typically:
//!   `mytracks.trx/header.json`, `positions.raw`, `offsets.raw`.
//!
//! # References
//!
//! * TRX format specification (tee-ar-ex/trx-cpp)
//! * <https://trx-cpp.readthedocs.io/>

mod error;
mod io;
mod parse;
mod types;

pub use error::TrxError;
pub use types::{TrxArrayDef, TrxGroup, TrxHeader, TrxReference, TrxTractogram};

#[cfg(test)]
mod tests;
