use super::*;

#[test]
fn centered_cursor_uses_shape_midpoint() {
    let cursor = LinkedCursor::centered([9, 11, 21]);
    assert_eq!(cursor.voxel(), [4, 5, 10]);
}

#[test]
fn viewport_mapping_axial_center_hits_expected_voxel() {
    let rect = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(200.0, 100.0));
    let voxel = viewport_point_to_voxel([8, 10, 20], 0, 3, egui::pos2(100.0, 50.0), rect)
        .expect("center point must map to a voxel");
    assert_eq!(voxel, [3, 5, 10]);
}

#[test]
fn viewport_mapping_coronal_maps_row_to_depth() {
    let rect = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(100.0, 160.0));
    let voxel = viewport_point_to_voxel([8, 10, 20], 1, 4, egui::pos2(50.0, 80.0), rect)
        .expect("point must map to voxel");
    assert_eq!(voxel, [4, 4, 10]);
}

#[test]
fn viewport_mapping_rejects_outside_points() {
    let rect = egui::Rect::from_min_size(egui::pos2(10.0, 10.0), egui::vec2(50.0, 50.0));
    assert!(viewport_point_to_voxel([8, 10, 20], 2, 6, egui::pos2(5.0, 5.0), rect).is_none());
}

#[test]
fn cursor_click_updates_hidden_axes() {
    let rect = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(200.0, 100.0));
    let mut cursor = LinkedCursor::from_slices([8, 10, 20], 3, 5, 9);
    let voxel = cursor
        .update_from_viewport_point([8, 10, 20], 0, 3, egui::pos2(150.0, 20.0), rect)
        .expect("click must update cursor");
    assert_eq!(voxel, [3, 2, 15]);
    assert_eq!(cursor.voxel(), [3, 2, 15]);
}

#[test]
fn projected_crosshair_matches_voxel_center() {
    let rect = egui::Rect::from_min_size(egui::pos2(10.0, 20.0), egui::vec2(200.0, 100.0));
    let cursor = LinkedCursor::from_slices([8, 10, 20], 3, 5, 9);
    let pos = cursor
        .viewport_crosshair([8, 10, 20], 0, rect)
        .expect("cursor projection must exist");
    assert_eq!(pos, egui::pos2(105.0, 75.0));
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

#[test]
fn viewport_projection_then_inverse_returns_same_voxel_on_fixed_slice() {
    let shape = [8, 10, 20];
    let rect = egui::Rect::from_min_size(egui::pos2(25.0, 40.0), egui::vec2(300.0, 180.0));
    let samples = [
        (0usize, [3usize, 4usize, 15usize]),
        (1usize, [6usize, 2usize, 11usize]),
        (2usize, [5usize, 7usize, 1usize]),
    ];

    for (axis, voxel) in samples {
        let point = voxel_to_viewport_point(shape, axis, voxel, rect)
            .expect("in-range voxel must project to viewport point");
        let slice = voxel[axis];
        let round_trip = viewport_point_to_voxel(shape, axis, slice, point, rect)
            .expect("projected point must map back to a voxel");
        assert_eq!(
            round_trip, voxel,
            "voxel projection/inverse mismatch on axis {axis}"
        );
    }
}
