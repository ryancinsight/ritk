//! Control-grid construction and dense-path selection.

use super::super::super::volume_dims::VolumeDims;

/// Upper bound on the control-lattice product for the dense support path.
pub const DENSE_LATTICE_CUTOFF: usize = 1_000_000;

/// Compute control-grid dimensions from image dimensions and control spacing.
///
/// The control lattice extends one extra control point beyond each boundary.
/// Along axis `d`, the dimension is `ceil(dims[d] / spacing[d]) + 3`.
pub fn init_control_grid(dims: VolumeDims, ctrl_spacing: &[f64; 3]) -> [usize; 3] {
    let d = dims.as_array();
    let mut ctrl_dims = [0usize; 3];
    for axis in 0..3 {
        ctrl_dims[axis] = (d[axis] as f64 / ctrl_spacing[axis]).ceil() as usize + 3;
    }
    ctrl_dims
}

/// Return whether the bounded dense support path should handle a lattice.
#[inline]
pub fn should_use_dense_path(ctrl_dims: &[usize; 3]) -> bool {
    let ctrl_n = ctrl_dims[0] * ctrl_dims[1] * ctrl_dims[2];
    ctrl_n <= DENSE_LATTICE_CUTOFF
}
