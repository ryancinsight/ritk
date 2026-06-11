//! Convenience warp via B-spline displacement field.

use super::basis::evaluate_bspline_displacement;
use super::volume_dims::VolumeDims;
use crate::deformable_field_ops::warp_image;

/// Warp an image using the B-spline displacement field.
///
/// Convenience wrapper that evaluates the dense displacement from control
/// points and then applies trilinear-interpolated warping.
///
/// # Returns
/// Warped image as a flat `Vec<f32>` of length `dims[0] * dims[1] * dims[2]`.
pub fn warp_image_bspline(
    moving: &[f32],
    dims: VolumeDims,
    cp_z: &[f32],
    cp_y: &[f32],
    cp_x: &[f32],
    ctrl_dims: &[usize; 3],
    ctrl_spacing: &[f64; 3],
) -> Vec<f32> {
    let disp = evaluate_bspline_displacement(cp_z, cp_y, cp_x, ctrl_dims, ctrl_spacing, dims);
    warp_image(moving, dims.as_array(), &disp.z, &disp.y, &disp.x)
}
