//! Coordinate encoding/decoding for N-dimensional image indexing.

/// Decode a flat (row-major) index into D-dimensional coordinates
/// (const-generic, zero-allocation stack array).
pub fn decode_coords<const D: usize>(flat: usize, shape: [usize; D]) -> [usize; D] {
    let mut coords = [0usize; D];
    let mut rem = flat;
    for d in (0..D).rev() {
        coords[d] = rem % shape[d];
        rem /= shape[d];
    }
    coords
}

/// Encode D-dimensional coordinates into a flat (row-major) index
/// (const-generic, accepts stack array reference).
pub fn encode_coords<const D: usize>(coords: &[usize; D], shape: [usize; D]) -> usize {
    let mut idx = 0;
    let mut stride = 1;
    for d in (0..D).rev() {
        idx += coords[d] * stride;
        stride *= shape[d];
    }
    idx
}

/// Dynamic (heap-allocating) variant — kept for backward compatibility
/// with callers that use runtime dimensionality `shape.len()`.
pub fn decode_coords_dyn(flat: usize, shape: &[usize]) -> Vec<usize> {
    let ndim = shape.len();
    let mut coords = vec![0usize; ndim];
    let mut rem = flat;
    for d in (0..ndim).rev() {
        coords[d] = rem % shape[d];
        rem /= shape[d];
    }
    coords
}

/// Dynamic (slice-accepting) variant — kept for backward compatibility
/// with callers that use runtime dimensionality `shape.len()`.
pub fn encode_coords_dyn(coords: &[usize], shape: &[usize]) -> usize {
    let ndim = shape.len();
    let mut idx = 0;
    let mut stride = 1;
    for d in (0..ndim).rev() {
        idx += coords[d] * stride;
        stride *= shape[d];
    }
    idx
}
