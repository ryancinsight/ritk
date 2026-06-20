// ── Indexing ────────────────────────────────────────────────────────────────────────

/// Clamped 3-D → linear index.
///
/// Each axis coordinate is clamped to `[0, dim-1]` before computing the
/// row-major linear index `z * ny * nx + y * nx + x`.
#[inline]
pub(crate) fn idx_clamped(z: isize, y: isize, x: isize, nz: usize, ny: usize, nx: usize) -> usize {
    let cz = z.clamp(0, nz as isize - 1) as usize;
    let cy = y.clamp(0, ny as isize - 1) as usize;
    let cx = x.clamp(0, nx as isize - 1) as usize;
    cz * ny * nx + cy * nx + cx
}
