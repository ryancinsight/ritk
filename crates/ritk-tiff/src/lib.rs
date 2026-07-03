//! TIFF / BigTIFF reader and writer for RITK.
//!
//! Each IFD page is one Z-slice of a volumetric `Image<B, 3>`.
//! Spatial metadata (origin, spacing, direction) is not encoded in TIFF;
//! readers assign identity defaults.

mod color;
mod reader;
mod writer;

pub use color::{read_tiff_color_to_volume, TiffColorReader};
/// Atlas-native-substrate I/O (plain end-state names, disambiguated from the
/// Burn functions by module path only; folds away when the Burn path is
/// deleted — ADR 0002 A1).
pub mod native {
    pub use crate::reader::native::*;
    pub use crate::writer::native::*;
}
pub use reader::{read_tiff, TiffReader};
pub use writer::{write_tiff, TiffWriter};
