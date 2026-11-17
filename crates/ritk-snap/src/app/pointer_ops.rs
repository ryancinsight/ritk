mod measurement;

use super::state::SnapApp;
use crate::render::NamedColorMap;
use crate::tools::interaction::{Annotation, ImagePoint, PolygonKind, RoiKind, ToolState};
use crate::tools::kind::ToolKind;
#[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
use crate::ui::{anatomical_label_for_axis, axis_for_plane_in_volume, AnatomicalPlane};
use crate::ui::{
    image_point_to_voxel, intensity_at_voxel, pan_from_drag_delta, window_level_from_drag_delta,
    zoom_from_drag_delta, WINDOW_LEVEL_SENSITIVITY,
};
use crate::viewer::{DEFAULT_WINDOW_CENTER, DEFAULT_WINDOW_WIDTH};
use metis_ui_lang::{PathCapture, PathPush};

// ── Pointer / interaction event handlers ───────────────────────────────────

impl SnapApp {
    /// Begin the gesture `tool` performs from `pos`.
    ///
    /// A gesture started while a multi-click measurement is pending suspends
    /// that measurement, so a reader can adjust the display between anchors.
    pub(crate) fn on_drag_start(&mut self, tool: ToolKind, pos: Option<ImagePoint>) {
        let Some(pos) = pos else { return };
        let gesture = match tool {
            ToolKind::Pan => ToolState::Panning {
                start: pos,
                viewport_origin: self.pan_offset,
            },
            ToolKind::Zoom => ToolState::Zooming {
                start: pos,
                original_zoom: self.zoom,
            },
            ToolKind::WindowLevel => ToolState::WindowLevelDrag {
                start: pos,
                original_center: f64::from(
                    self.viewer_state
                        .window_center
                        .unwrap_or(DEFAULT_WINDOW_CENTER),
                ),
                original_width: f64::from(
                    self.viewer_state
                        .window_width
                        .unwrap_or(DEFAULT_WINDOW_WIDTH),
                ),
            },
            ToolKind::RoiRect => ToolState::RoiDrag {
                start: pos,
                current: pos,
                kind: RoiKind::Rect,
            },
            ToolKind::RoiEllipse => ToolState::RoiDrag {
                start: pos,
                current: pos,
                kind: RoiKind::Ellipse,
            },
            ToolKind::MeasureLength => ToolState::LengthDrag {
                start: pos,
                current: pos,
            },
            ToolKind::MeasureDeviation => ToolState::DeviationDrag {
                start: pos,
                current: pos,
            },
            _ => return,
        };
        if self.tool_state.is_pending_measurement() {
            self.pointer.suspended_measurement = Some(self.tool_state.clone());
        }
        self.pointer.drag_measurement_location = matches!(
            gesture,
            ToolState::LengthDrag { .. }
                | ToolState::DeviationDrag { .. }
                | ToolState::RoiDrag { .. }
        )
        .then(|| self.annotation_location());
        self.tool_state = gesture;
    }

    pub(crate) fn on_drag(&mut self, pos: Option<ImagePoint>) {
        let Some(pos) = pos else { return };
        match self.tool_state.clone() {
            ToolState::Panning {
                start,
                viewport_origin,
            } => {
                let offset = pan_from_drag_delta(viewport_origin, start, pos);
                self.pan_offset = offset;
            }
            ToolState::Zooming {
                start,
                original_zoom,
            } => {
                let drag_delta_y = pos.y() - start.y();
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
                    pos.x() - start.x(),
                    pos.y() - start.y(),
                    WINDOW_LEVEL_SENSITIVITY,
                );
                self.viewer_state.window_center = Some(new_center as f32);
                self.viewer_state.window_width = Some(new_width as f32);
                self.bump_visual_revision();
            }
            ToolState::RoiDrag { start, kind, .. } => {
                self.tool_state = ToolState::RoiDrag {
                    start,
                    current: pos,
                    kind,
                };
            }
            ToolState::LengthDrag { start, .. } => {
                self.tool_state = ToolState::LengthDrag {
                    start,
                    current: pos,
                };
            }
            ToolState::MeasureLength1 { p1 }
                if self.active_tool == ToolKind::MeasureLength
                    && self.pointer.pending_measurement_location
                        == Some(self.annotation_location()) =>
            {
                self.pointer.drag_measurement_location =
                    self.pointer.pending_measurement_location.take();
                self.tool_state = ToolState::LengthDrag {
                    start: p1,
                    current: pos,
                };
            }
            ToolState::DeviationDrag { start, .. } => {
                self.tool_state = ToolState::DeviationDrag {
                    start,
                    current: pos,
                };
            }
            _ => {}
        }
    }

    pub(crate) fn on_drag_end(&mut self, pos: Option<ImagePoint>) {
        if pos.is_none() {
            self.pointer.pending_polygon_move = None;
        }
        let recorded = self.pointer.drag_measurement_location.take();
        if pos.is_some() && recorded.is_none_or(|location| location == self.annotation_location()) {
            // Only a measurement takes its release as a final point. The
            // viewport mapping a display gesture's release already includes
            // that gesture's own zoom and pan, so reapplying it would move it.
            if matches!(
                self.tool_state,
                ToolState::RoiDrag { .. }
                    | ToolState::LengthDrag { .. }
                    | ToolState::DeviationDrag { .. }
            ) {
                self.on_drag(pos);
            }
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
                ToolState::LengthDrag { start, current } if start != current => {
                    self.record_length(start, current);
                }
                ToolState::DeviationDrag { start, current } if start != current => {
                    self.finalise_deviation(start, current);
                }
                _ => {}
            }
        }
        self.finish_gesture();
    }

    /// Release the gesture owner and return to any suspended measurement.
    ///
    /// State that no drag owns, such as a waiting measurement anchor, stays.
    fn finish_gesture(&mut self) {
        self.pointer.owner = None;
        if self.tool_state.is_drag_gesture() {
            self.tool_state = self
                .pointer
                .suspended_measurement
                .take()
                .unwrap_or(ToolState::Idle);
        }
    }

    /// End a click without discarding a multi-click measurement anchor.
    ///
    /// A click is also the release of a press that may have initialized a
    /// transient drag state. Measurement tools use the release itself to
    /// advance their anchors, so they must be allowed to observe their prior
    /// state before the transient gesture is cleared.
    pub(crate) fn on_click_end(&mut self) {
        if self.tool_state.is_drag_gesture() {
            self.finish_gesture();
        } else {
            self.pointer.owner = None;
            self.pointer.suspended_measurement = None;
        }
    }

    pub(crate) fn on_click(&mut self, pos: Option<ImagePoint>) {
        let Some(pos) = pos else { return };
        let location = self.annotation_location();
        if self.tool_state.is_pending_measurement()
            && self
                .pointer
                .pending_measurement_location
                .is_some_and(|anchor| anchor != location)
        {
            self.tool_state = ToolState::Idle;
        }
        match self.active_tool {
            ToolKind::MeasureLength => match self.tool_state.clone() {
                ToolState::MeasureLength1 { p1 } => {
                    self.record_length(p1, pos);
                    self.tool_state = ToolState::Idle;
                }
                _ => {
                    self.tool_state = ToolState::MeasureLength1 { p1: pos };
                }
            },
            ToolKind::MeasureAngle => match self.tool_state.clone() {
                ToolState::MeasureAngle2 { p1, p2 } => {
                    let a = [p1.y(), p1.x()];
                    let b = [p2.y(), p2.x()];
                    let c = [pos.y(), pos.x()];
                    let angle_deg = match self
                        .slice_plane_spacing()
                        .and_then(Annotation::validate_spacing)
                    {
                        Ok(spacing) => Annotation::compute_angle_with_spacing(a, b, c, spacing),
                        Err(error) => {
                            self.status_message = format!("Measurement rejected: {error}");
                            self.tool_state = ToolState::Idle;
                            return;
                        }
                    };
                    self.push_annotation(Annotation::Angle {
                        location: None,
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
            ToolKind::RoiClosedPolygon | ToolKind::RoiOpenPolygon => {
                if let ToolState::Polygon { points, kind } = &mut self.tool_state {
                    let samples = points.as_slice();
                    let closes = *kind == PolygonKind::Closed
                        && samples.len() >= 3
                        && samples.first().is_some_and(|point| {
                            let dx = f64::from(point.x()) - f64::from(pos.x());
                            let dy = f64::from(point.y()) - f64::from(pos.y());
                            dx.hypot(dy) <= 6.0
                        });
                    if closes {
                        self.finish_pending_polygon();
                    } else if matches!(points.push(pos), PathPush::Full) {
                        self.status_message = "Polygon sample limit reached".to_owned();
                    }
                } else {
                    self.tool_state = ToolState::Polygon {
                        points: PathCapture::new(pos),
                        kind: if self.active_tool == ToolKind::RoiOpenPolygon {
                            PolygonKind::Open
                        } else {
                            PolygonKind::Closed
                        },
                    };
                }
            }
            ToolKind::PointHu => {
                if let Some(vol) = &self.loaded {
                    let Some(voxel) =
                        image_point_to_voxel(vol.shape, self.axis, location.slice, pos)
                    else {
                        return;
                    };
                    let value = intensity_at_voxel(vol, voxel);
                    self.push_annotation(Annotation::HuPoint {
                        location: None,
                        pos: [pos.y(), pos.x()],
                        value,
                    });
                    let [depth, row, col] = voxel;
                    self.status_message = format!("Value at [{depth},{row},{col}]: {value:.0}");
                }
            }
            ToolKind::LabelPaint | ToolKind::LabelErase => {}
            _ => {}
        }
        self.pointer.pending_measurement_location =
            self.tool_state.is_pending_measurement().then_some(location);
    }

    pub(crate) fn apply_label_at_pointer(&mut self, axis: usize, pos: Option<ImagePoint>) {
        let Some(point) = pos else { return };
        let Some(volume) = &self.loaded else { return };
        let Some(voxel) =
            image_point_to_voxel(volume.shape, axis, self.axis_slice_info(axis).0, point)
        else {
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
        pos: Option<ImagePoint>,
    ) {
        let Some(point) = pos else { return };
        let slice_index = self.axis_slice_info(axis).0;
        let Some(volume) = &self.loaded else { return };
        let Some(cursor) = self.linked_cursor.as_mut() else {
            return;
        };
        let Some(voxel) = cursor.update_from_image_point(volume.shape, axis, slice_index, point)
        else {
            return;
        };
        self.viewer_state.slice_index = voxel[0];
        self.coronal_slice = voxel[1];
        self.sagittal_slice = voxel[2];
        self.axis = axis.min(2);
        self.bump_visual_revision();
        self.status_message = format!(
            "Linked cursor axis={} voxel=[{},{},{}]",
            axis, voxel[0], voxel[1], voxel[2]
        );
    }

    pub(crate) fn update_pointer_intensity(&mut self, axis: usize, pos: Option<ImagePoint>) {
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
        let Some(voxel) = image_point_to_voxel(volume.shape, axis, slice_index, point) else {
            self.pointer_intensity = 0.0;
            self.pointer_suv = None;
            return;
        };
        self.pointer_intensity = intensity_at_voxel(volume, voxel);
        self.pointer_suv = Self::compute_suv_from_volume(volume, self.pointer_intensity as f64);
    }

    /// Select the default colormap for a modality string.
    ///
    /// PT → `NamedColorMap::Hot` (standard PET display); all others → `NamedColorMap::Grayscale`.
    pub(crate) fn colormap_for_modality(modality: Option<&str>) -> NamedColorMap {
        if modality == Some("PT") {
            NamedColorMap::Hot
        } else {
            NamedColorMap::Grayscale
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
    #[cfg(not(target_arch = "wasm32"))]
    #[cfg(feature = "eframe-shell")]
    pub(crate) fn current_cursor_suv(&self) -> Option<f32> {
        let volume = self.loaded.as_ref()?;
        let cursor = self.linked_cursor?;
        let [z, y, x] = cursor.voxel();
        let pixel = volume.pixel_at(z, y, x) as f64;
        Self::compute_suv_from_volume(volume, pixel)
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[cfg(feature = "eframe-shell")]
    pub(crate) fn current_cursor_value(&self) -> Option<f32> {
        let volume = self.loaded.as_ref()?;
        let cursor = self.linked_cursor?;
        let [z, y, x] = cursor.voxel();
        Some(volume.pixel_at(z, y, x))
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[cfg(feature = "eframe-shell")]
    pub(crate) fn axis_for_plane(&self, plane: AnatomicalPlane) -> usize {
        axis_for_plane_in_volume(self.loaded.as_ref(), plane)
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[cfg(feature = "eframe-shell")]
    pub(crate) fn axis_label(&self, axis: usize) -> &'static str {
        anatomical_label_for_axis(self.loaded.as_ref(), axis)
    }
}
