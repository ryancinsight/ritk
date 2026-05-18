use super::state::SnapApp;
use crate::render::colormap::Colormap;
use crate::tools::interaction::{Annotation, RoiKind, ToolState};
use crate::tools::kind::ToolKind;
use crate::ui::{
    anatomical_label_for_axis, axis_for_plane_in_volume, intensity_at_voxel, pan_from_drag_delta,
    viewport_point_to_voxel, window_level_from_drag_delta, zoom_from_drag_delta, AnatomicalPlane,
    WINDOW_LEVEL_SENSITIVITY,
};

// ── Pointer / interaction event handlers ───────────────────────────────────

impl SnapApp {
    pub(crate) fn on_drag_start(&mut self, pos: Option<egui::Pos2>) {
        let Some(pos) = pos else { return };
        match self.active_tool {
            ToolKind::Pan => {
                self.tool_state = ToolState::Panning {
                    start: pos,
                    viewport_origin: egui::Pos2::new(self.pan_offset.x, self.pan_offset.y),
                };
            }
            ToolKind::Zoom => {
                self.tool_state = ToolState::Zooming {
                    start: pos,
                    original_zoom: self.zoom,
                };
            }
            ToolKind::WindowLevel => {
                let wc = self.viewer_state.window_center.unwrap_or(128.0) as f64;
                let ww = self.viewer_state.window_width.unwrap_or(256.0) as f64;
                self.tool_state = ToolState::WindowLevelDrag {
                    start: pos,
                    original_center: wc,
                    original_width: ww,
                };
            }
            ToolKind::RoiRect => {
                self.tool_state = ToolState::RoiDrag {
                    start: pos,
                    current: pos,
                    kind: RoiKind::Rect,
                };
            }
            ToolKind::RoiEllipse => {
                self.tool_state = ToolState::RoiDrag {
                    start: pos,
                    current: pos,
                    kind: RoiKind::Ellipse,
                };
            }
            _ => {}
        }
    }

    pub(crate) fn on_drag(&mut self, pos: Option<egui::Pos2>) {
        let Some(pos) = pos else { return };
        match self.tool_state.clone() {
            ToolState::Panning {
                start,
                viewport_origin,
            } => {
                self.pan_offset = pan_from_drag_delta(viewport_origin, start, pos);
            }
            ToolState::Zooming {
                start,
                original_zoom,
            } => {
                let drag_delta_y = pos.y - start.y;
                self.zoom = zoom_from_drag_delta(original_zoom, drag_delta_y);
                self.status_message = format!("Zoom: {:.0}%", self.zoom * 100.0);
            }
            ToolState::WindowLevelDrag {
                start,
                original_center,
                original_width,
            } => {
                let (new_center, new_width) = window_level_from_drag_delta(
                    original_center,
                    original_width,
                    pos.x - start.x,
                    pos.y - start.y,
                    WINDOW_LEVEL_SENSITIVITY,
                );
                self.viewer_state.window_center = Some(new_center as f32);
                self.viewer_state.window_width = Some(new_width as f32);
                self.texture_dirty = true;
                self.coronal_dirty = true;
                self.sagittal_dirty = true;
                self.mip_dirty = true;
            }
            ToolState::RoiDrag { start, kind, .. } => {
                self.tool_state = ToolState::RoiDrag {
                    start,
                    current: pos,
                    kind,
                };
            }
            _ => {}
        }
    }

    pub(crate) fn on_drag_end(&mut self, pos: Option<egui::Pos2>) {
        if pos.is_some() {
            match self.tool_state.clone() {
                ToolState::RoiDrag {
                    start,
                    current,
                    kind: RoiKind::Rect,
                } => {
                    self.finalise_roi_rect(start, current);
                }
                ToolState::RoiDrag {
                    start,
                    current,
                    kind: RoiKind::Ellipse,
                } => {
                    self.finalise_roi_ellipse(start, current);
                }
                _ => {}
            }
        }
        self.tool_state = ToolState::Idle;
    }

    pub(crate) fn on_click(&mut self, pos: Option<egui::Pos2>) {
        let Some(pos) = pos else { return };
        match self.active_tool {
            ToolKind::MeasureLength => match self.tool_state.clone() {
                ToolState::MeasureLength1 { p1 } => {
                    let p1_arr = [p1.y, p1.x];
                    let p2_arr = [pos.y, pos.x];
                    let spacing = self.slice_spacing_2d();
                    let length_mm = Annotation::compute_length(p1_arr, p2_arr, spacing);
                    self.annotations.push(Annotation::Length {
                        p1: p1_arr,
                        p2: p2_arr,
                        length_mm,
                    });
                    self.tool_state = ToolState::Idle;
                }
                _ => {
                    self.tool_state = ToolState::MeasureLength1 { p1: pos };
                }
            },
            ToolKind::MeasureAngle => match self.tool_state.clone() {
                ToolState::MeasureAngle2 { p1, p2 } => {
                    let a = [p1.y, p1.x];
                    let b = [p2.y, p2.x];
                    let c = [pos.y, pos.x];
                    let angle_deg = Annotation::compute_angle(a, b, c);
                    self.annotations.push(Annotation::Angle {
                        p1: a,
                        p2: b,
                        p3: c,
                        angle_deg,
                    });
                    self.tool_state = ToolState::Idle;
                }
                ToolState::MeasureLength1 { p1 } => {
                    self.tool_state = ToolState::MeasureAngle2 { p1, p2: pos };
                }
                _ => {
                    self.tool_state = ToolState::MeasureLength1 { p1: pos };
                }
            },
            ToolKind::PointHu => {
                if let Some(vol) = &self.loaded {
                    let row = pos.y as usize;
                    let col = pos.x as usize;
                    let (pixels, width, _height) =
                        vol.extract_slice(self.axis, self.viewer_state.slice_index);
                    let idx = row * width + col;
                    let value = if idx < pixels.len() { pixels[idx] } else { 0.0 };
                    self.annotations.push(Annotation::HuPoint {
                        pos: [pos.y, pos.x],
                        value,
                    });
                    self.status_message = format!("HU at col={col} row={row}: {value:.0}");
                }
            }
            ToolKind::LabelPaint | ToolKind::LabelErase => {}
            _ => {}
        }
    }

    // ── Annotation helpers ────────────────────────────────────────────────

    /// Compute ROI rect statistics for the pixel region between `start` and
    /// `end` (screen-space corners) on the current primary-axis slice.
    fn finalise_roi_rect(&mut self, start: egui::Pos2, end: egui::Pos2) {
        let Some(vol) = &self.loaded else { return };
        let p1 = [start.y, start.x];
        let p2 = [end.y, end.x];
        let spacing = self.slice_spacing_2d();
        let (pixels, width, height) = vol.extract_slice(self.axis, self.viewer_state.slice_index);
        let (mean, std_dev, min, max, area_mm2) =
            Annotation::compute_roi_rect_stats(p1, p2, &pixels, width, height, spacing);
        self.annotations.push(Annotation::RoiRect {
            top_left: [p1[0].min(p2[0]), p1[1].min(p2[1])],
            bottom_right: [p1[0].max(p2[0]), p1[1].max(p2[1])],
            mean,
            std_dev,
            min,
            max,
            area_mm2,
        });
        self.status_message =
            format!("ROI: \u{03bc}={mean:.1} \u{03c3}={std_dev:.1} [{min:.0}, {max:.0}] {area_mm2:.1} mm\u{b2}");
    }

    fn finalise_roi_ellipse(&mut self, start: egui::Pos2, end: egui::Pos2) {
        let Some(vol) = &self.loaded else { return };
        let p1 = [start.y, start.x];
        let p2 = [end.y, end.x];
        let spacing = self.slice_spacing_2d();
        let (pixels, width, height) = vol.extract_slice(self.axis, self.viewer_state.slice_index);
        let (center, radii, mean, std_dev, min, max, area_mm2) =
            Annotation::compute_roi_ellipse_stats(p1, p2, &pixels, width, height, spacing);
        self.annotations.push(Annotation::RoiEllipse {
            center,
            radii,
            mean,
            std_dev,
            min,
            max,
            area_mm2,
        });
        self.status_message = format!(
            "Ellipse ROI: \u{03bc}={mean:.1} \u{03c3}={std_dev:.1} [{min:.0}, {max:.0}] {area_mm2:.1} mm\u{b2}"
        );
    }

    /// Per-axis 2-D pixel spacing `[row_spacing, col_spacing]` in mm/pixel.
    ///
    /// | axis | row spacing | col spacing |
    /// |------|-------------|-------------|
    /// | 0 axial   | dy | dx |
    /// | 1 coronal | dz | dx |
    /// | 2 sagittal| dz | dy |
    pub(crate) fn slice_spacing_2d(&self) -> [f32; 2] {
        let Some(vol) = &self.loaded else {
            return [1.0, 1.0];
        };
        let [dz, dy, dx] = vol.spacing.map(|s| s as f32);
        match self.axis {
            0 => [dy, dx],
            1 => [dz, dx],
            _ => [dz, dy],
        }
    }

    pub(crate) fn apply_label_at_pointer(
        &mut self,
        axis: usize,
        pos: Option<egui::Pos2>,
        rect: egui::Rect,
    ) {
        let Some(point) = pos else { return };
        let Some(volume) = &self.loaded else { return };
        let Some(voxel) = viewport_point_to_voxel(
            volume.shape,
            axis,
            self.axis_slice_info(axis).0,
            point,
            rect,
        ) else {
            return;
        };
        let Some(editor) = self.label_editor.as_mut() else {
            return;
        };
        let result = match self.active_tool {
            ToolKind::LabelPaint => editor.paint_sphere(voxel, self.label_brush_radius),
            ToolKind::LabelErase => editor.erase_sphere(voxel, self.label_brush_radius),
            _ => return,
        };
        match result {
            Ok(changed) if changed > 0 => {
                self.status_message = format!(
                    "Label edit axis={} voxel=[{},{},{}] changed {} voxels",
                    axis, voxel[0], voxel[1], voxel[2], changed
                );
            }
            Ok(_) => {}
            Err(e) => {
                self.status_message = format!("Label edit failed: {e}");
            }
        }
    }

    pub(crate) fn update_linked_cursor_from_pointer(
        &mut self,
        axis: usize,
        pos: Option<egui::Pos2>,
        rect: egui::Rect,
    ) {
        let Some(point) = pos else { return };
        let slice_index = self.axis_slice_info(axis).0;
        let Some(volume) = &self.loaded else { return };
        let Some(cursor) = self.linked_cursor.as_mut() else {
            return;
        };
        let Some(voxel) =
            cursor.update_from_viewport_point(volume.shape, axis, slice_index, point, rect)
        else {
            return;
        };
        self.viewer_state.slice_index = voxel[0];
        self.coronal_slice = voxel[1];
        self.sagittal_slice = voxel[2];
        self.axis = axis.min(2);
        self.texture_dirty = true;
        self.coronal_dirty = true;
        self.sagittal_dirty = true;
        self.mip_dirty = true;
        self.status_message = format!(
            "Linked cursor axis={} voxel=[{},{},{}]",
            axis, voxel[0], voxel[1], voxel[2]
        );
    }

    pub(crate) fn update_pointer_intensity(
        &mut self,
        axis: usize,
        pos: Option<egui::Pos2>,
        rect: egui::Rect,
    ) {
        let Some(point) = pos else {
            self.pointer_intensity = 0.0;
            self.pointer_suv = None;
            return;
        };
        let Some(volume) = &self.loaded else {
            self.pointer_intensity = 0.0;
            self.pointer_suv = None;
            return;
        };
        let slice_index = self.axis_slice_info(axis).0;
        let Some(voxel) = viewport_point_to_voxel(volume.shape, axis, slice_index, point, rect)
        else {
            self.pointer_intensity = 0.0;
            self.pointer_suv = None;
            return;
        };
        self.pointer_intensity = intensity_at_voxel(volume, voxel);
        self.pointer_suv = Self::compute_suv_from_volume(volume, self.pointer_intensity as f64);
    }

    /// Select the default colormap for a modality string.
    ///
    /// PT → `Colormap::Hot` (standard PET display); all others → `Colormap::Grayscale`.
    pub(crate) fn colormap_for_modality(modality: Option<&str>) -> Colormap {
        if modality == Some("PT") {
            Colormap::Hot
        } else {
            Colormap::Grayscale
        }
    }

    /// Compute SUVbw for a PET voxel value [Bq/mL].
    ///
    /// Returns `None` when `PetAcquisitionParams::from_loaded_volume` fails,
    /// `pixel_bqml` is non-finite, or the result is non-finite.
    pub(crate) fn compute_suv_from_volume(
        vol: &crate::LoadedVolume,
        pixel_bqml: f64,
    ) -> Option<f32> {
        use crate::dicom::pet::PetAcquisitionParams;
        if !pixel_bqml.is_finite() {
            return None;
        }
        let pet = PetAcquisitionParams::from_loaded_volume(vol)?;
        let delta_t = PetAcquisitionParams::delta_t_s_from_vol(vol);
        let suv = pet.pixel_to_suvbw(pixel_bqml, delta_t);
        if suv.is_finite() {
            Some(suv as f32)
        } else {
            None
        }
    }

    /// Compute SUVbw at the linked-cursor voxel position, if available.
    ///
    /// Consumed by the overlay renderer for PET/CT SUV workflow (GAP-176-RAD-02).
    pub(crate) fn current_cursor_suv(&self) -> Option<f32> {
        let volume = self.loaded.as_ref()?;
        let cursor = self.linked_cursor?;
        let [z, y, x] = cursor.voxel();
        let pixel = volume.pixel_at(z, y, x) as f64;
        Self::compute_suv_from_volume(volume, pixel)
    }

    pub(crate) fn current_cursor_value(&self) -> Option<f32> {
        let volume = self.loaded.as_ref()?;
        let cursor = self.linked_cursor?;
        let [z, y, x] = cursor.voxel();
        Some(volume.pixel_at(z, y, x))
    }

    pub(crate) fn axis_for_plane(&self, plane: AnatomicalPlane) -> usize {
        axis_for_plane_in_volume(self.loaded.as_ref(), plane)
    }

    pub(crate) fn axis_label(&self, axis: usize) -> &'static str {
        anatomical_label_for_axis(self.loaded.as_ref(), axis)
    }
}
