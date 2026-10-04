use super::fixtures::sample_value;
use crate::write_nrrd_with_data;
use anyhow::Result;
use coeus_core::SequentialBackend;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};
use tempfile::tempdir;

type TestBackend = SequentialBackend;

fn make_image(
    data: Vec<f32>,
    dims: [usize; 3],
    origin: Point<3>,
    spacing: Spacing<3>,
    direction: Direction<3>,
) -> Image<f32, TestBackend, 3> {
    Image::from_flat_on(data, dims, origin, spacing, direction, &SequentialBackend)
        .expect("valid image")
}

fn zeros_image(
    dims: [usize; 3],
    origin: Point<3>,
    spacing: Spacing<3>,
    direction: Direction<3>,
) -> Image<f32, TestBackend, 3> {
    let n = dims[0] * dims[1] * dims[2];
    make_image(vec![0.0f32; n], dims, origin, spacing, direction)
}

/// Scan `haystack` for the ASCII byte pattern `needle`.
fn bytes_contain(haystack: &[u8], needle: &str) -> bool {
    let nb = needle.as_bytes();
    haystack.windows(nb.len()).any(|w| w == nb)
}

fn axial_direction() -> Direction<3> {
    Direction::from_row_major([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0])
}

fn nrrd_payload(bytes: &[u8]) -> &[u8] {
    let terminator = b"\n\n";
    let header_end = bytes
        .windows(terminator.len())
        .position(|w| w == terminator)
        .map(|p| p + terminator.len())
        .expect("blank-line terminator not found in NRRD file");
    &bytes[header_end..]
}

fn decode_le_f32_payload(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

mod geometry;
/// A written NRRD file must contain the mandatory header fields.
mod header;
mod round_trip;
