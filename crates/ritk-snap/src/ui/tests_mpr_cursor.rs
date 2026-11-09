use super::*;

#[test]
fn centered_cursor_uses_shape_midpoint() {
    let cursor = LinkedCursor::centered([9, 11, 21]);
    assert_eq!(cursor.voxel(), [4, 5, 10]);
}

#[test]
fn set_axis_slice_clamps_to_shape() {
    let mut cursor = LinkedCursor::centered([8, 10, 20]);
    cursor.set_axis_slice([8, 10, 20], 1, 999);
    assert_eq!(cursor.voxel(), [4, 9, 10]);
}

#[test]
fn row_col_voxel_mapping_is_invertible_per_axis_plane() {
    let sample = [
        (0usize, 4usize, 7usize, 3usize),
        (1usize, 5usize, 2usize, 9usize),
        (2usize, 6usize, 1usize, 8usize),
    ];
    for (axis, slice, row, col) in sample {
        let voxel = map_view_row_col_to_voxel(axis, slice, row, col);
        let (inv_row, inv_col) = map_voxel_to_view_row_col(axis, voxel)
            .expect("valid axis must produce inverse coordinates");
        assert_eq!(inv_row, row, "row inverse mismatch for axis {axis}");
        assert_eq!(inv_col, col, "col inverse mismatch for axis {axis}");
    }
}
