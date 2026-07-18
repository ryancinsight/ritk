use super::state::{RtDoseOverlayCacheEntry, SnapApp};
use crate::ui::rtdose_overlay::extract_dose_slice_for_volume;
use crate::ui::rtdose_texture::{build_overlay_image, overlay_alpha, positive_finite_dose_range};
use crate::ui::{
    axis_slice_dimensions, compute_roi_dose_analytics, map_view_row_col_to_voxel,
    project_rt_struct_contours_for_slice, rt_dose_analytics::VolumeGeometry };
use ritk_annotation::Visibility;
use tracing::{error, info};

impl SnapApp {
    #[cfg(test)]
    pub(crate) fn rt_dose_plan_link_status(&self) -> Option<String> {
        let dose = self.rt_dose.as_ref()?;
        let Some(ref_uid) = dose
            .referenced_rt_plan_sop_instance_uid
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
        else {
            return Some("Plan linkage: no ReferencedRTPlanSequence UID".to_owned());
        };
        match self.rt_plan.as_ref() {
            None => Some(format!(
                "Plan linkage: references UID {ref_uid} (no RT-PLAN loaded)"
            )),
            Some(plan) => {
                let loaded_uid = plan.sop_instance_uid.trim();
                if loaded_uid.is_empty() {
                    Some(format!(
                        "Plan linkage: references UID {ref_uid} (loaded RT-PLAN has empty SOP UID)"
                    ))
                } else if loaded_uid == ref_uid {
                    Some(format!(
                        "Plan linkage: linked to loaded RT-PLAN UID {ref_uid}"
                    ))
                } else {
                    Some(format!(
                        "Plan linkage: mismatch (dose references {ref_uid}, loaded plan is {loaded_uid})"
                    ))
                }
            }
        }
    }

    pub(crate) fn refresh_rt_dvh_cache(&mut self) {
        let (Some(vol), Some(rt_struct), Some(rt_dose), Some(roi_number)) = (
            self.loaded.as_ref(),
            self.rt_struct.as_ref(),
            self.rt_dose.as_ref(),
            self.rt_dvh_selected_roi,
        ) else {
            self.rt_dvh_cache = None;
            return;
        };
        self.rt_dvh_cache = compute_roi_dose_analytics(
            rt_struct,
            rt_dose,
            roi_number,
            VolumeGeometry {
                shape: vol.shape,
                origin: vol.origin,
                direction: vol.direction,
                spacing: vol.spacing },
            128,
        );
    }

    pub(crate) fn load_rt_struct_file(&mut self, path: std::path::PathBuf) {
        match ritk_io::read_rt_struct(&path) {
            Ok(rt) => {
                let roi_count = rt.rois.len();
                let label = rt.structure_set_label.clone();
                self.rt_dvh_selected_roi = rt.rois.first().map(|roi| roi.roi_number);
                self.rt_struct = Some(rt);
                self.show_rt_struct_overlay = true;
                self.refresh_rt_dvh_cache();
                self.status_message = format!(
                    "Loaded RT-STRUCT {} ({} ROIs) from {}",
                    label,
                    roi_count,
                    path.display()
                );
                info!("{}", self.status_message);
            }
            Err(e) => {
                self.status_message =
                    format!("RT-STRUCT load failed for {}: {e:#}", path.display());
                error!("{}", self.status_message);
            }
        }
    }

    pub(crate) fn load_rt_dose_file(&mut self, path: std::path::PathBuf) {
        match ritk_io::read_rt_dose(&path) {
            Ok(grid) => {
                let max_dose_gy = grid.dose_gy.iter().copied().fold(0.0_f64, f64::max);
                self.status_message = format!(
                    "Loaded RT-DOSE ({} type, {}Ã—{}Ã—{} grid) from {}",
                    grid.dose_type.as_dicom_str(),
                    grid.rows,
                    grid.cols,
                    grid.n_frames,
                    path.display()
                );
                info!("{}", self.status_message);
                self.rt_dose = Some(grid);
                self.rt_dose_max_gy = Some(max_dose_gy);
                self.clear_rt_dose_overlay_cache();
                self.show_rt_dose_overlay = true;
                self.refresh_rt_dvh_cache();
            }
            Err(e) => {
                self.status_message = format!("RT-DOSE load failed for {}: {e:#}", path.display());
                error!("{}", self.status_message);
            }
        }
    }

    pub(crate) fn load_rt_plan_file(&mut self, path: std::path::PathBuf) {
        match ritk_io::read_rt_plan(&path) {
            Ok(plan) => {
                let beam_count = plan.beams.len();
                let fg_count = plan.fraction_groups.len();
                let label = plan.rt_plan_label.clone();
                self.rt_plan = Some(plan);
                self.refresh_rt_dvh_cache();
                self.status_message = format!(
                    "Loaded RT-PLAN {} ({} beams, {} fraction groups) from {}",
                    label,
                    beam_count,
                    fg_count,
                    path.display()
                );
                info!("{}", self.status_message);
            }
            Err(e) => {
                self.status_message = format!("RT-PLAN load failed for {}: {e:#}", path.display());
                error!("{}", self.status_message);
            }
        }
    }

    pub(crate) fn clear_rt_dose_overlay_cache(&mut self) {
        self.rt_dose_overlay_cache = std::array::from_fn(|_| None);
    }

    /// Draw the RT-DOSE heat-map overlay on the given viewport.
    pub(crate) fn draw_rt_dose_overlay(
        &mut self,
        painter: &egui::Painter,
        rect: egui::Rect,
        axis: usize,
        slice_idx: usize,
    ) {
        let (Some(rt_dose), Some(vol)) = (&self.rt_dose, &self.loaded) else {
            return;
        };

        let axis_slot = axis.min(2);
        let vol_shape = vol.shape;
        let dose_dims = [rt_dose.n_frames, rt_dose.rows, rt_dose.cols];
        let opacity_alpha = overlay_alpha(self.rt_dose_opacity);

        if let Some(entry) = self.rt_dose_overlay_cache[axis_slot].as_ref() {
            if entry.slice_idx == slice_idx
                && entry.vol_shape == vol_shape
                && entry.dose_dims == dose_dims
                && entry.opacity_alpha == opacity_alpha
            {
                painter.image(
                    entry.texture.id(),
                    rect,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    egui::Color32::WHITE,
                );
                return;
            }
        }

        let [depth, rows, cols] = vol_shape;
        let vol_origin = vol.origin;
        let vol_dir: [f64; 9] = vol.direction;
        let vol_spacing = vol.spacing;

        let Some(dose_map) = extract_dose_slice_for_volume(
            rt_dose,
            axis,
            slice_idx,
            [depth, rows, cols],
            vol_origin,
            vol_dir,
            vol_spacing,
        ) else {
            return;
        };

        let Some((min_dose, max_dose)) = positive_finite_dose_range(&dose_map) else {
            return;
        };

        let (slice_rows, slice_cols) = match axis {
            0 => (rows, cols),
            1 => (depth, cols),
            _ => (depth, rows) };

        if slice_rows == 0 || slice_cols == 0 {
            return;
        }

        let Some(color_image) = build_overlay_image(
            &dose_map,
            slice_rows,
            slice_cols,
            min_dose,
            max_dose,
            self.rt_dose_opacity,
        ) else {
            return;
        };

        let tex_name = format!("rtdose_overlay_axis{}_slice{}", axis_slot, slice_idx);
        let texture =
            painter
                .ctx()
                .load_texture(tex_name, color_image, egui::TextureOptions::LINEAR);
        let texture_id = texture.id();

        self.rt_dose_overlay_cache[axis_slot] = Some(RtDoseOverlayCacheEntry {
            slice_idx,
            vol_shape,
            dose_dims,
            opacity_alpha,
            texture });

        painter.image(
            texture_id,
            rect,
            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
            egui::Color32::WHITE,
        );
    }

    pub(crate) fn draw_label_overlay(
        &self,
        painter: &egui::Painter,
        rect: egui::Rect,
        axis: usize,
    ) {
        let Some(editor) = &self.label_editor else {
            return;
        };
        let Some(volume) = &self.loaded else {
            return;
        };
        let Some((width, height)) = axis_slice_dimensions(volume.shape, axis) else {
            return;
        };
        if width == 0 || height == 0 {
            return;
        }

        let slice_index = self.axis_slice_info(axis).0;
        let cell_w = rect.width() / width as f32;
        let cell_h = rect.height() / height as f32;

        for row in 0..height {
            for col in 0..width {
                let voxel = map_view_row_col_to_voxel(axis, slice_index, row, col);
                let label_id = editor.current_map().label_at(voxel);
                if label_id == 0 {
                    continue;
                }
                let Some(entry) = editor.current_map().table.get_label(label_id) else {
                    continue;
                };
                if entry.visible == Visibility::Hidden {
                    continue;
                }
                let x0 = rect.min.x + col as f32 * cell_w;
                let y0 = rect.min.y + row as f32 * cell_h;
                painter.rect_filled(
                    egui::Rect::from_min_size(egui::pos2(x0, y0), egui::vec2(cell_w, cell_h)),
                    0.0,
                    egui::Color32::from_rgba_unmultiplied(
                        entry.color.r(),
                        entry.color.g(),
                        entry.color.b(),
                        entry.color.a(),
                    ),
                );
            }
        }
    }

    pub(crate) fn draw_rt_struct_overlay(
        &self,
        painter: &egui::Painter,
        rect: egui::Rect,
        axis: usize,
        image_h: usize,
        image_w: usize,
    ) {
        let Some(volume) = &self.loaded else {
            return;
        };
        let Some(rt) = &self.rt_struct else {
            return;
        };
        if image_h == 0 || image_w == 0 {
            return;
        }

        let (slice_index, _) = self.axis_slice_info(axis);

        let projected = project_rt_struct_contours_for_slice(
            rt,
            axis,
            slice_index,
            volume.shape,
            volume.origin,
            volume.direction,
            volume.spacing,
        );

        let to_screen = |row: f32, col: f32| -> egui::Pos2 {
            egui::pos2(
                rect.min.x + ((col + 0.5) / image_w as f32) * rect.width(),
                rect.min.y + ((row + 0.5) / image_h as f32) * rect.height(),
            )
        };

        for contour in projected {
            let color =
                egui::Color32::from_rgb(contour.color[0], contour.color[1], contour.color[2]);
            if contour.points_row_col.len() == 1 {
                let [row, col] = contour.points_row_col[0];
                painter.circle_filled(to_screen(row, col), 2.0, color);
                continue;
            }
            for pair in contour.points_row_col.windows(2) {
                let a = pair[0];
                let b = pair[1];
                painter.line_segment(
                    [to_screen(a[0], a[1]), to_screen(b[0], b[1])],
                    egui::Stroke::new(1.5_f32, color),
                );
            }
            if contour.closed {
                if let (Some(first), Some(last)) = (
                    contour.points_row_col.first().copied(),
                    contour.points_row_col.last().copied(),
                ) {
                    painter.line_segment(
                        [to_screen(last[0], last[1]), to_screen(first[0], first[1])],
                        egui::Stroke::new(1.5_f32, color),
                    );
                }
            }
        }
    }
}
