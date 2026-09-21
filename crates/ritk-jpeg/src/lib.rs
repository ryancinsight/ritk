//! JPEG grayscale and RGB image I/O.
//!
//! JPEG carries no physical-space metadata. Readers assign origin `[0,0,0]`,
//! spacing `[1,1,1]`, and identity direction. Writers require `nz == 1` and
//! encode a single 2-D grayscale plane. File decode preserves encoded raster
//! orientation and does not apply EXIF display transforms.

mod color;
mod decode;
mod reader;
mod writer;

pub use color::{read_jpeg_color_to_volume, JpegColorReader};
pub use reader::{read_jpeg, JpegReader};
pub use writer::{write_jpeg, JpegWriter};

#[cfg(test)]
mod tests;
