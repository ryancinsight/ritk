//! Sample window and sparse entry types for the direct Parzen path.
//!
//! Decomposed into vertical hierarchy (ARCH-330-02):
//! - `sample_window` — [`SampleWindow`] per-sample context (direct + sparse paths)
//! - `sparse_entry` — [`SparseWFixedEntry`] / [`SparseWFixedT`] sparse cache types

mod sample_window;
mod sparse_entry;

pub(crate) use sample_window::SampleWindow;
pub use sparse_entry::{SparseWFixedEntry, SparseWFixedT};
