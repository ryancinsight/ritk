mod header;
mod payload;

use coeus_core::SequentialBackend;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};

pub(super) type TestBackend = SequentialBackend;

pub(super) fn make_image(
    data: Vec<f32>,
    dims: [usize; 3],
    origin: Point<3>,
    spacing: Spacing<3>,
    direction: Direction<3>,
) -> Image<f32, TestBackend, 3> {
    Image::from_flat_on(data, dims, origin, spacing, direction, &SequentialBackend)
        .expect("valid image")
}

pub(super) fn zeros_image(
    dims: [usize; 3],
    origin: Point<3>,
    spacing: Spacing<3>,
    direction: Direction<3>,
) -> Image<f32, TestBackend, 3> {
    let n = dims[0] * dims[1] * dims[2];
    make_image(vec![0.0f32; n], dims, origin, spacing, direction)
}

/// Scan `haystack` for the ASCII byte pattern `needle`.
pub(super) fn bytes_contain(haystack: &[u8], needle: &str) -> bool {
    let nb = needle.as_bytes();
    haystack.windows(nb.len()).any(|w| w == nb)
}

pub(super) fn axial_direction() -> Direction<3> {
    Direction::from_row_major([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0])
}

pub(super) fn nrrd_payload(bytes: &[u8]) -> &[u8] {
    let terminator = b"\n\n";
    let header_end = bytes
        .windows(terminator.len())
        .position(|w| w == terminator)
        .map(|p| p + terminator.len())
        .expect("blank-line terminator not found in NRRD file");
    &bytes[header_end..]
}

pub(super) fn payload_samples(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}
