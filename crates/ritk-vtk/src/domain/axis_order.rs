//! Axis mapping between RITK tensor images and VTK image coordinates.
//!
//! RITK stores three-dimensional tensor axes as ZYX, while VTK structured
//! points use XYZ and list scalar samples with X varying fastest. Reordering
//! the metadata axes preserves the contiguous scalar sequence.

use ritk_spatial::Direction;

/// Reorders a three-axis value between XYZ and ZYX order.
pub(crate) fn reverse_axes<T>([first, middle, last]: [T; 3]) -> [T; 3] {
    [last, middle, first]
}

/// Maps direction columns from RITK tensor axes into VTK XYZ axes.
pub(crate) fn tensor_to_vtk_direction(direction: Direction<3>) -> Direction<3> {
    let [depth, row, column] = direction.axis_directions_array();
    Direction::from_columns([column, row, depth])
}

/// Direction whose columns map RITK ZYX tensor axes to VTK-aligned XYZ space.
pub(crate) fn vtk_image_direction() -> Direction<3> {
    tensor_to_vtk_direction(Direction::identity())
}

#[cfg(test)]
mod tests {
    use super::{reverse_axes, tensor_to_vtk_direction, vtk_image_direction};
    use ritk_spatial::{Direction, Vector};

    #[test]
    fn axis_order_reversal_is_an_involution() {
        let xyz = [0.5, 1.5, 2.0];
        assert_eq!(reverse_axes(xyz), [2.0, 1.5, 0.5]);
        assert_eq!(reverse_axes(reverse_axes(xyz)), xyz);
    }

    #[test]
    fn tensor_direction_reorders_axis_columns() {
        let direction = Direction::from_rows([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);

        assert_eq!(
            tensor_to_vtk_direction(direction),
            Direction::from_columns([
                Vector::new([3.0, 6.0, 9.0]),
                Vector::new([2.0, 5.0, 8.0]),
                Vector::new([1.0, 4.0, 7.0]),
            ])
        );
    }

    #[test]
    fn default_tensor_direction_maps_to_vtk_axes() {
        assert_eq!(
            vtk_image_direction().to_row_major(),
            [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]
        );
    }
}
