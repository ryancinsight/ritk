//! Coordinate encoding/decoding for N-dimensional image indexing.

/// Decode a flat (row-major) index into D-dimensional coordinates.
pub fn decode_coords(flat: usize, shape: &[usize]) -> Vec<usize> {
    let ndim = shape.len();
    let mut coords = vec![0usize; ndim];
    let mut rem = flat;
    for d in (0..ndim).rev() {
        coords[d] = rem % shape[d];
        rem /= shape[d];
    }
    coords
}

/// Encode D-dimensional coordinates into a flat (row-major) index.
pub fn encode_coords(coords: &[usize], shape: &[usize]) -> usize {
    let ndim = shape.len();
    let mut idx = 0;
    let mut stride = 1;
    for d in (0..ndim).rev() {
        idx += coords[d] * stride;
        stride *= shape[d];
    }
    idx
}
