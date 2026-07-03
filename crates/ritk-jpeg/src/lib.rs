//! JPEG grayscale and RGB image I/O.
//!
//! JPEG carries no physical-space metadata. Readers assign origin `[0,0,0]`,
//! spacing `[1,1,1]`, and identity direction. Writers require `nz == 1` and
//! encode a single 2-D grayscale plane.

mod color;
mod reader;
mod writer;

pub use color::{read_jpeg_color_to_volume, JpegColorReader};
pub use reader::native;
pub use reader::{read_jpeg, JpegReader};
pub use writer::{write_jpeg, JpegWriter};

#[cfg(test)]
mod tests;
