use super::state::{SeriesLoadTarget, SnapApp};
use crate::session::ViewerSessionSnapshot;
use crate::tools::interaction::ToolState;
use crate::ui::window_presets::WindowPreset;
use crate::ui::{
    decide_dropped_input_action, format_lps, show_colorbar, voxel_to_lps, DroppedInputAction,
    LinkedCursor, MAX_ZOOM, MIN_ZOOM,
};

impl SnapApp {
    pub(crate) fn mark_all_textures_dirty(&mut self) {
        self.texture_dirty = true;
        self.coronal_dirty = true;
        self.sagittal_dirty = true;
        self.mip_dirty = true;
        self.secondary_texture_dirty = true;
    }

    pub(crate) fn undo_label_edit_shortcut(&mut self) {
        let Some(editor) = self.label_editor.as_mut() else {
            return;
        };
        if editor.undo() {
            self.status_message = "Segmentation undo.".to_owned();
        }
    }

    pub(crate) fn redo_label_edit_shortcut(&mut self) {
        let Some(editor) = self.label_editor.as_mut() else {
            return;
        };
        if editor.redo() {
            self.status_message = "Segmentation redo.".to_owned();
        }
    }

    /// Apply a [`WindowPreset`] and mark all textures dirty.
    pub(crate) fn apply_preset(&mut self, preset: WindowPreset) {
        self.viewer_state.window_center = Some(preset.center as f32);
        self.viewer_state.window_width = Some(preset.width as f32);
        self.texture_dirty = true;
        self.coronal_dirty = true;
        self.sagittal_dirty = true;
        self.mip_dirty = true;
    }

    pub(crate) fn session_snapshot(&self) -> ViewerSessionSnapshot {
        ViewerSessionSnapshot {
            source: self.loaded.as_ref().and_then(|vol| vol.source.clone()),
            viewer_state: self.viewer_state,
            colormap: self.colormap,
            axis: self.axis,
            active_tool: self.active_tool,
            multi_planar: self.multi_planar,
            show_overlay: self.show_overlay,
            show_crosshair: self.show_crosshair,
            show_rt_struct_overlay: self.show_rt_struct_overlay,
            show_rt_dose_overlay: self.show_rt_dose_overlay,
            rt_dose_opacity: self.rt_dose_opacity,
            show_series_browser: self.show_series_browser,
            sidebar_tab: self.sidebar_tab,
            coronal_slice: self.coronal_slice,
            sagittal_slice: self.sagittal_slice,
            pan_offset: [self.pan_offset.x, self.pan_offset.y],
            zoom: self.zoom,
            cine_enabled: self.cine.enabled,
            cine_fps: self.cine.fps,
            annotations: self.annotations.clone(),
        }
    }

    pub(crate) fn apply_session_snapshot(&mut self, snapshot: ViewerSessionSnapshot) {
        self.viewer_state = snapshot.viewer_state;
        self.colormap = snapshot.colormap;
        self.axis = snapshot.axis.min(2);
        self.active_tool = snapshot.active_tool;
        self.multi_planar = snapshot.multi_planar;
        self.show_overlay = snapshot.show_overlay;
        self.show_crosshair = snapshot.show_crosshair;
        self.show_rt_struct_overlay = snapshot.show_rt_struct_overlay;
        self.show_rt_dose_overlay = snapshot.show_rt_dose_overlay;
        self.rt_dose_opacity = snapshot.rt_dose_opacity;
        self.show_series_browser = snapshot.show_series_browser;
        self.sidebar_tab = snapshot.sidebar_tab;
        self.coronal_slice = snapshot.coronal_slice;
        self.sagittal_slice = snapshot.sagittal_slice;
        self.pan_offset = egui::Vec2::new(snapshot.pan_offset[0], snapshot.pan_offset[1]);
        self.zoom = snapshot.zoom.clamp(MIN_ZOOM, MAX_ZOOM);
        self.cine.restore(snapshot.cine_enabled, snapshot.cine_fps);
        self.annotations = snapshot.annotations;
        self.tool_state = ToolState::Idle;
        self.texture_dirty = true;
        self.coronal_dirty = true;
        self.sagittal_dirty = true;
        self.mip_dirty = true;
        self.linked_cursor = self.loaded.as_ref().map(|vol| {
            LinkedCursor::from_slices(
                vol.shape,
                self.viewer_state.slice_index,
                self.coronal_slice,
                self.sagittal_slice,
            )
        });
        if let Some(source) = snapshot.source {
            self.pending_load = Some(source);
        }
    }

    /// Ingest shell-dropped inputs (desktop) or browser-dropped file handles.
    ///
    /// Files with filesystem paths are routed to the same code paths as File-menu
    /// actions. Browser-provided handles without paths are acknowledged with a
    /// deterministic status message.
    pub(crate) fn handle_dropped_inputs(&mut self, ctx: &egui::Context) {
        let dropped = ctx.input_mut(|i| std::mem::take(&mut i.raw.dropped_files));
        match decide_dropped_input_action(&dropped) {
            DroppedInputAction::QueueDicom(path) => {
                self.scan_for_series(path.clone());
                self.pending_load = Some(path.clone());
                self.status_message = format!("Queued dropped DICOM input: {}", path.display());
            }
            DroppedInputAction::LoadVolume(path) => {
                self.load_volume_file(path);
            }
            DroppedInputAction::LoadVolumeBytes { name, bytes } => {
                self.load_volume_bytes(name, bytes.as_ref());
            }
            DroppedInputAction::LoadDicomSeriesBytes { files } => {
                self.load_dicom_series_bytes(files);
            }
            DroppedInputAction::Message(msg) => {
                self.status_message = msg;
            }
            DroppedInputAction::None => {}
        }
    }

    // ── Left panel ────────────────────────────────────────────────────────────

    pub(crate) fn show_left_panel(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("info_panel")
            .min_width(220.0)
            .max_width(360.0)
            .show(ctx, |ui| {
                if !self.show_series_browser {
                    ui.heading("Series Browser Hidden");
                    ui.label("Enable 'Show Series Browser' from the View menu or ribbon.");
                    return;
                }

                ui.heading("Series Browser");

                ui.horizontal(|ui| {
                    ui.label("Target:");
                    ui.selectable_value(
                        &mut self.series_load_target,
                        SeriesLoadTarget::Primary,
                        "Primary",
                    );
                    ui.selectable_value(
                        &mut self.series_load_target,
                        SeriesLoadTarget::Secondary,
                        "Secondary",
                    );
                });

                ui.separator();

                let sidebar_result = {
                    let pointer_suv = self.pointer_suv;
                    let cursor_suv = self.current_cursor_suv();
                    let tree_ref = &self.series_tree;
                    let sel_ref = &mut self.selected_series;
                    let tab_ref = &mut self.sidebar_tab;
                    let vol_ref = self.loaded.as_ref();
                    let mut panel = crate::ui::sidebar::SidebarPanel::new(
                        tree_ref,
                        sel_ref,
                        tab_ref,
                        vol_ref,
                        pointer_suv,
                        cursor_suv,
                    );
                    panel.show(ui)
                };

                if let Some(folder) = sidebar_result {
                    match self.series_load_target {
                        SeriesLoadTarget::Primary => self.pending_load = Some(folder),
                        SeriesLoadTarget::Secondary => self.pending_secondary_load = Some(folder),
                    }
                }
            });
    }

    // ── Bottom status bar ─────────────────────────────────────────────────────

    pub(crate) fn show_bottom_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(&self.status_message);
                if let Some(vol) = &self.loaded {
                    let (slice_idx, total) = self.axis_slice_info(self.status_axis);
                    let _ = vol; // vol borrow used implicitly via axis_slice_info
                    ui.separator();
                    ui.label(format!("Slice {}/{}", slice_idx + 1, total));
                    ui.separator();
                    let axis_name = self.axis_label(self.status_axis);
                    ui.label(axis_name);

                    // Voxel I/J/K index and physical LPS position from linked cursor.
                    if let (Some(cursor), Some(vol)) = (self.linked_cursor, self.loaded.as_ref()) {
                        let [d, r, c] = cursor.voxel();
                        ui.separator();
                        ui.label(format!("I={d} J={r} K={c}"));
                        let lps = voxel_to_lps([d, r, c], vol.origin, vol.direction, vol.spacing);
                        ui.separator();
                        ui.label(format_lps(lps));
                    }
                }
            });
        });
    }

    pub(crate) fn show_aux_windows(&mut self, ctx: &egui::Context) {
        let mut show_filter = self.show_filter_panel;
        if show_filter {
            egui::Window::new("Processing")
                .open(&mut show_filter)
                .resizable(true)
                .show(ctx, |ui| {
                    let applied =
                        crate::ui::filter_panel::show_filter_panel(ui, &mut self.active_filter);
                    if applied {
                        self.apply_filter_to_loaded_volume();
                    }
                });
        }
        self.show_filter_panel = show_filter;

        let mut show_cb = self.show_colorbar;
        if show_cb {
            egui::Window::new("Colorbar")
                .open(&mut show_cb)
                .resizable(false)
                .show(ctx, |ui| {
                    let wc = self.viewer_state.window_center.unwrap_or(128.0);
                    let ww = self.viewer_state.window_width.unwrap_or(256.0).max(1.0);
                    show_colorbar(ui, self.colormap, wc, ww);
                });
        }
        self.show_colorbar = show_cb;

        // ── PACS Network Panel ────────────────────────────────────────────────
        let mut show_pacs = self.show_pacs_panel;
        if show_pacs {
            egui::Window::new("PACS Network")
                .open(&mut show_pacs)
                .default_width(560.0)
                .resizable(true)
                .show(ctx, |ui| {
                    #[cfg(not(target_arch = "wasm32"))]
                    let (scp_listening, scp_actual_port) = self
                        .pacs_scp_handle
                        .as_ref()
                        .map(|h| (true, h.port()))
                        .unwrap_or((false, 0));
                    #[cfg(target_arch = "wasm32")]
                    let (scp_listening, scp_actual_port) = (false, 0u16);
                    #[cfg(not(target_arch = "wasm32"))]
                    let pacs_pending_count = self.pacs_pending_instances.len();
                    #[cfg(target_arch = "wasm32")]
                    let pacs_pending_count = 0usize;

                    let action = crate::ui::pacs_panel::show_pacs_panel(
                        ui,
                        &mut self.pacs_config,
                        &mut self.pacs_query_state,
                        &mut self.pacs_echo_display,
                        &mut self.pacs_patient_filter,
                        &mut self.pacs_modality_filter,
                        &mut self.pacs_study_date_filter,
                        &mut self.pacs_accession_filter,
                        scp_listening,
                        scp_actual_port,
                        self.pacs_received_count,
                        pacs_pending_count,
                        self.pacs_auto_loaded_this_frame,
                        &mut self.pacs_selected_row,
                    );
                    self.handle_pacs_action(action);
                });
        }
        self.show_pacs_panel = show_pacs;
    }

    pub(crate) fn show_right_info_panel(&self, ui: &mut egui::Ui) {
        ui.vertical(|ui| {
            ui.heading("MPR Info");
            ui.separator();

            if let Some(vol) = &self.loaded {
                let [depth, rows, cols] = vol.shape;
                let [dz, dy, dx] = vol.spacing;
                let wc = self.viewer_state.window_center.unwrap_or(128.0);
                let ww = self.viewer_state.window_width.unwrap_or(256.0);

                egui::Grid::new("mpr_info_grid")
                    .num_columns(2)
                    .spacing([8.0, 4.0])
                    .striped(true)
                    .show(ui, |ui| {
                        let row = |ui: &mut egui::Ui, k: &str, v: &str| {
                            ui.label(k);
                            ui.label(v);
                            ui.end_row();
                        };

                        row(ui, "W / L:", &format!("{ww:.0} / {wc:.0}"));

                        let (slice_idx, total) = match self.status_axis {
                            0 => (self.viewer_state.slice_index, depth),
                            1 => (self.coronal_slice, rows),
                            2 => (self.sagittal_slice, cols),
                            _ => (self.viewer_state.slice_index, depth),
                        };
                        let axis_name = self.axis_label(self.status_axis);
                        row(ui, axis_name, &format!("{}/{}", slice_idx + 1, total));

                        row(ui, "Dims:", &format!("{depth}×{rows}×{cols}"));
                        row(ui, "Spacing:", &format!("{dz:.2}×{dy:.2}×{dx:.2} mm"));
                        row(ui, "Modality:", vol.modality.as_deref().unwrap_or("—"));
                        row(ui, "Patient:", vol.patient_name.as_deref().unwrap_or("—"));
                    });

                ui.separator();
                ui.label("Scroll wheel: navigate slices.");
                ui.label("Ctrl/Cmd + scroll: zoom in/out.");
                ui.label("Ctrl/Cmd + 0: zoom to fit.");
                ui.label("Ctrl/Cmd + Z / Shift+Z / Y: segmentation undo/redo.");
                ui.label("Arrow/Page: previous/next slice. Home/End: first/last slice.");
                ui.label("Click: move the linked MPR cursor.");
                ui.label("Drag: active tool interaction.");
            } else {
                ui.label("No volume loaded.");
                ui.label("Open a DICOM folder or NIfTI file via File menu.");
            }
        });
    }
}
