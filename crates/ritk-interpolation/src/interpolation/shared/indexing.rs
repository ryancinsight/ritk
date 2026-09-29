//! Index math shared by the per-shape interpolation kernels.

/// Compute row-major strides for a shape (last axis contiguous).
pub(crate) fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let rank = shape.len();
    let mut strides = vec![1usize; rank];
    for d in (0..rank.saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * shape[d + 1];
    }
    strides
}

/// Clamp a coordinate to the valid index range for an axis.
pub(crate) fn clamp_index(idx: f32, size: usize) -> usize {
    if size == 0 {
        return 0;
    }
    let max = (size - 1) as f32;
    let clamped = idx.clamp(0.0, max);
    clamped as usize
}
