//! Linked MPR cursor state and viewport/voxel coordinate transforms.
//!
//! This module is the SSOT for the study-coordinate cursor shared by the axial,
//! coronal, and sagittal viewports.

/// Linked crosshair cursor stored in voxel coordinates `[z, y, x]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinkedCursor {
    voxel: [usize; 3],
}

impl LinkedCursor {
    /// Create a cursor at the geometric midpoint of `shape`.
    pub fn centered(shape: [usize; 3]) -> Self {
        Self {
            voxel: [shape[0] / 2, shape[1] / 2, shape[2] / 2],
        }
    }

    /// Create a cursor from the active per-axis slice indices.
    pub fn from_slices(shape: [usize; 3], axial: usize, coronal: usize, sagittal: usize) -> Self {
        let mut cursor = Self::centered(shape);
        cursor.voxel = [
            clamp_index(axial, shape[0]),
            clamp_index(coronal, shape[1]),
            clamp_index(sagittal, shape[2]),
        ];
        cursor
    }

    /// Return the current voxel coordinate.
    pub fn voxel(&self) -> [usize; 3] {
        self.voxel
    }

    /// Set the cursor to `voxel`, clamped to `shape`.
    pub fn set_voxel(&mut self, shape: [usize; 3], voxel: [usize; 3]) {
        self.voxel = [
            clamp_index(voxel[0], shape[0]),
            clamp_index(voxel[1], shape[1]),
            clamp_index(voxel[2], shape[2]),
        ];
    }

    /// Update the slice coordinate corresponding to `axis`.
    pub fn set_axis_slice(&mut self, shape: [usize; 3], axis: usize, slice_index: usize) {
        match axis {
            0 => self.voxel[0] = clamp_index(slice_index, shape[0]),
            1 => self.voxel[1] = clamp_index(slice_index, shape[1]),
            2 => self.voxel[2] = clamp_index(slice_index, shape[2]),
            _ => {}
        }
    }

    /// Map a viewport point into a study voxel using the current slice on `axis`.
    pub fn update_from_viewport_point(
        &mut self,
        shape: [usize; 3],
        axis: usize,
        slice_index: usize,
        point: egui::Pos2,
        rect: egui::Rect,
    ) -> Option<[usize; 3]> {
        let voxel = viewport_point_to_voxel(shape, axis, slice_index, point, rect)?;
        self.set_voxel(shape, voxel);
        Some(self.voxel)
    }

    /// Project the linked voxel into viewport coordinates for `axis`.
    pub fn viewport_crosshair(
        &self,
        shape: [usize; 3],
        axis: usize,
        rect: egui::Rect,
    ) -> Option<egui::Pos2> {
        voxel_to_viewport_point(shape, axis, self.voxel, rect)
    }
}

/// Return `(width, height)` for the 2-D slice shown by `axis`.
pub fn axis_slice_dimensions(shape: [usize; 3], axis: usize) -> Option<(usize, usize)> {
    match axis {
        0 => Some((shape[2], shape[1])),
        1 => Some((shape[2], shape[0])),
        2 => Some((shape[1], shape[0])),
        _ => None,
    }
}

/// Map a viewport `(row, col)` back into the study voxel `[z, y, x]`.
pub fn map_view_row_col_to_voxel(
    axis: usize,
    slice_index: usize,
    row: usize,
    col: usize,
) -> [usize; 3] {
    match axis {
        0 => [slice_index, row, col],
        1 => [row, slice_index, col],
        2 => [row, col, slice_index],
        _ => [slice_index, row, col],
    }
}

/// Map a viewport point into a voxel on the currently displayed slice.
pub fn viewport_point_to_voxel(
    shape: [usize; 3],
    axis: usize,
    slice_index: usize,
    point: egui::Pos2,
    rect: egui::Rect,
) -> Option<[usize; 3]> {
    let (width, height) = axis_slice_dimensions(shape, axis)?;
    if width == 0 || height == 0 || rect.width() <= 0.0 || rect.height() <= 0.0 {
        return None;
    }
    if !rect.contains(point) {
        return None;
    }

    let x = ((point.x - rect.min.x) / rect.width()).clamp(0.0, 0.999_999);
    let y = ((point.y - rect.min.y) / rect.height()).clamp(0.0, 0.999_999);
    let col = (x * width as f32).floor() as usize;
    let row = (y * height as f32).floor() as usize;
    Some(map_view_row_col_to_voxel(axis, slice_index, row, col))
}

/// Project a study voxel into viewport coordinates for `axis`.
pub fn voxel_to_viewport_point(
    shape: [usize; 3],
    axis: usize,
    voxel: [usize; 3],
    rect: egui::Rect,
) -> Option<egui::Pos2> {
    let (width, height) = axis_slice_dimensions(shape, axis)?;
    if width == 0 || height == 0 || rect.width() <= 0.0 || rect.height() <= 0.0 {
        return None;
    }

    let (row, col) = match axis {
        0 => (voxel[1], voxel[2]),
        1 => (voxel[0], voxel[2]),
        2 => (voxel[0], voxel[1]),
        _ => return None,
    };
    if row >= height || col >= width {
        return None;
    }

    let x = rect.min.x + ((col as f32 + 0.5) / width as f32) * rect.width();
    let y = rect.min.y + ((row as f32 + 0.5) / height as f32) * rect.height();
    Some(egui::pos2(x, y))
}

fn clamp_index(index: usize, len: usize) -> usize {
    if len == 0 {
        0
    } else {
        index.min(len - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centered_cursor_uses_shape_midpoint() {
        let cursor = LinkedCursor::centered([9, 11, 21]);
        assert_eq!(cursor.voxel(), [4, 5, 10]);
    }

    #[test]
    fn viewport_mapping_axial_center_hits_expected_voxel() {
        let rect = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(200.0, 100.0));
        let voxel = viewport_point_to_voxel(
            [8, 10, 20],
            0,
            3,
            egui::pos2(100.0, 50.0),
            rect,
        )
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
}
