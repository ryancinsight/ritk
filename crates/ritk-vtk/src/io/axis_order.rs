//! Axis mapping between legacy VTK image files and RITK tensor images.
//!
//! Legacy VTK structured-points headers express dimensions, origin, and spacing
//! in XYZ order, and list scalar values with X varying fastest ([VTK dataset
//! format](https://docs.vtk.org/en/v9.6.1/vtk_file_formats/vtk_legacy_file_format.html#dataset-format)).
//! RITK tensors use ZYX storage axes. The reader therefore reverses the header
//! tuples while retaining the x-fastest value stream, and records the matching
//! direction columns. Legacy structured-points has no direction field, so the
//! writer accepts only that representable axis mapping.

use ritk_spatial::Direction;

/// Reorders a three-axis value between VTK XYZ and RITK ZYX order.
pub(crate) fn xyz_to_zyx<T>([x, y, z]: [T; 3]) -> [T; 3] {
    [z, y, x]
}

/// Direction columns for VTK XYZ axes represented by RITK's ZYX tensor axes.
pub(crate) fn vtk_image_direction() -> Direction<3> {
    Direction::from_rows([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
}

#[cfg(test)]
mod tests {
    use super::{vtk_image_direction, xyz_to_zyx};
    #[test]
    fn axis_order_reversal_is_an_involution() {
        let xyz = [0.5, 1.5, 2.0];
        assert_eq!(xyz_to_zyx(xyz), [2.0, 1.5, 0.5]);
        assert_eq!(xyz_to_zyx(xyz_to_zyx(xyz)), xyz);
    }
    #[test]
    fn direction_columns_match_tensor_axis_order() {
        assert_eq!(
            vtk_image_direction().to_row_major(),
            [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]
        );
    }
}
