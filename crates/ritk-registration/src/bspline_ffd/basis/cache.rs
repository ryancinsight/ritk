//! Pre-computed B-spline basis cache for all three image axes.
//!
//! Building the cache is O(nz + ny + nx) â€” negligible (~0.01 ms for a 256Â³ volume).
//! Each `evaluate_bspline_displacement_fast` call saves ~50M `cubic_bspline_1d`
//! evaluations.

use super::super::volume_dims::VolumeDims;
use super::scalar::AxisBasis;

/// Pre-computed B-spline basis data for all three axes of a 3-D volume.
#[derive(Clone)]
pub struct BasisCache {
    pub z: AxisBasis,
    pub y: AxisBasis,
    pub x: AxisBasis }

impl BasisCache {
    /// Build the basis cache for the given image dimensions and control spacing.
    pub fn new(dims: VolumeDims, ctrl_spacing: &[f64; 3]) -> Self {
        let [nz, ny, nx] = dims.as_array();
        Self {
            z: AxisBasis::new(nz, ctrl_spacing[0]),
            y: AxisBasis::new(ny, ctrl_spacing[1]),
            x: AxisBasis::new(nx, ctrl_spacing[2]) }
    }

    /// Return interior z-range `[z_lo, z_hi)` where all 4 control points
    /// along z are in-bounds (i.e. `kz âˆˆ [0, cnz-4]`).
    pub fn interior_z_range(&self, cnz: usize) -> (usize, usize) {
        let lo = self.z.k.iter().position(|&k| k >= 0).unwrap_or(0);
        let hi = self
            .z
            .k
            .iter()
            .rposition(|&k| k <= cnz as isize - 4)
            .map(|p| p + 1)
            .unwrap_or(0);
        (lo, hi)
    }

    /// Return interior y-range `[y_lo, y_hi)` where all 4 control points
    /// along y are in-bounds.
    pub fn interior_y_range(&self, cny: usize) -> (usize, usize) {
        let lo = self.y.k.iter().position(|&k| k >= 0).unwrap_or(0);
        let hi = self
            .y
            .k
            .iter()
            .rposition(|&k| k <= cny as isize - 4)
            .map(|p| p + 1)
            .unwrap_or(0);
        (lo, hi)
    }

    /// Return interior x-range `[x_lo, x_hi)` where all 4 control points
    /// along x are in-bounds.
    pub fn interior_x_range(&self, cnx: usize) -> (usize, usize) {
        let lo = self.x.k.iter().position(|&k| k >= 0).unwrap_or(0);
        let hi = self
            .x
            .k
            .iter()
            .rposition(|&k| k <= cnx as isize - 4)
            .map(|p| p + 1)
            .unwrap_or(0);
        (lo, hi)
    }
}
