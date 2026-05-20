use super::state::SnapApp;
use crate::render::colormap::Colormap;
use crate::ui::window_presets::WindowPreset;
use crate::ui::AnatomicalPlane;
use crate::ModalityDisplay;

#[cfg(not(target_arch = "wasm32"))]
use rfd::FileDialog;

#[cfg(target_arch = "wasm32")]
struct FileDialog;

#[cfg(target_arch = "wasm32")]
impl FileDialog {
    fn new() -> Self {
        Self
    }
    fn set_file_name(self, _name: &str) -> Self {
        self
    }
    fn add_filter(self, _name: &str, _extensions: &[&str]) -> Self {
        self
    }
    fn pick_file(self) -> Option<std::path::PathBuf> {
        None
    }
    fn pick_folder(self) -> Option<std::path::PathBuf> {
        None
    }
    fn save_file(self) -> Option<std::path::PathBuf> {
        None
    }
}

impl SnapApp {
    pub(crate) fn show_menu_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                // ── File ─────────────────────────────────────────────────────
                ui.menu_button("File", |ui| {
                    #[cfg(target_arch = "wasm32")]
                    {
                        ui.colored_label(
                            egui::Color32::YELLOW,
                            "Browser build: local file/folder dialogs are not available yet.",
                        );
                        ui.separator();
                    }

                    if ui.button("Open DICOM folder…").clicked() {
                        ui.close_menu();
                        if let Some(folder) = FileDialog::new().pick_folder() {
                            self.scan_for_series(folder.clone());
                            self.pending_load = Some(folder);
                        }
                    }

                    if ui.button("Open DICOMDIR…").clicked() {
                        ui.close_menu();
                        if let Some(path) = FileDialog::new().set_file_name("DICOMDIR").pick_file()
                        {
                            self.scan_for_series(path.clone());
                            self.pending_load = Some(path);
                        }
                    }

                    if ui.button("Open DICOM file…").clicked() {
                        ui.close_menu();
                        if let Some(path) = FileDialog::new()
                            .add_filter("DICOM", &["dcm", "dicom"])
                            .pick_file()
                        {
                            self.scan_for_series(path.clone());
                            self.pending_load = Some(path);
                        }
                    }

                    if ui.button("Open NIfTI / MHA / NRRD file\u{2026}").clicked() {
                        ui.close_menu();
                        if let Some(path) = FileDialog::new()
                            .add_filter(
                                "Medical images",
                                &["nii", "gz", "mha", "mhd", "nrrd", "nhdr", "mgh", "mgz"],
                            )
                            .pick_file()
                        {
                            self.load_volume_file(path);
                        }
                    }

                    if ui.button("Open Mesh\u{2026}").clicked() {
                        ui.close_menu();
                        if let Some(path) = FileDialog::new()
                            .add_filter("Surface meshes", &["stl", "obj", "ply"])
                            .pick_file()
                        {
                            self.load_mesh_file(path);
                        }
                    }

                    if ui.button("Open RT-STRUCT file…").clicked() {
                        ui.close_menu();
                        if let Some(path) =
                            FileDialog::new().add_filter("DICOM", &["dcm"]).pick_file()
                        {
                            self.load_rt_struct_file(path);
                        }
                    }

                    if ui.button("Open RT Dose file…").clicked() {
                        ui.close_menu();
                        if let Some(path) =
                            FileDialog::new().add_filter("DICOM", &["dcm"]).pick_file()
                        {
                            self.load_rt_dose_file(path);
                        }
                    }

                    if ui.button("Open RT Plan file…").clicked() {
                        ui.close_menu();
                        if let Some(path) =
                            FileDialog::new().add_filter("DICOM", &["dcm"]).pick_file()
                        {
                            self.load_rt_plan_file(path);
                        }
                    }

                    if ui.button("Export current slice as PNG…").clicked() {
                        ui.close_menu();
                        self.export_current_slice();
                    }

                    if ui.button("Export all MPR slices as PNG…").clicked() {
                        ui.close_menu();
                        self.export_all_mpr_slices();
                    }

                    if ui.button("Export clinical distribution package…").clicked() {
                        ui.close_menu();
                        self.export_clinical_distribution_dialog();
                    }

                    if ui.button("Save segmentation as NIfTI…").clicked() {
                        ui.close_menu();
                        self.save_segmentation_dialog();
                    }

                    if ui.button("Save segmentation as DICOM-SEG…").clicked() {
                        ui.close_menu();
                        self.save_segmentation_dicom_seg_dialog();
                    }

                    if ui.button("Export label surface as VTK…").clicked() {
                        ui.close_menu();
                        self.export_surface_dialog();
                    }

                    if ui.button("Load segmentation from NIfTI…").clicked() {
                        ui.close_menu();
                        self.load_segmentation_dialog();
                    }

                    if ui.button("Load segmentation from DICOM-SEG…").clicked() {
                        ui.close_menu();
                        self.load_segmentation_dicom_seg_dialog();
                    }

                    if ui.button("Save session…").clicked() {
                        ui.close_menu();
                        self.save_session_dialog();
                    }

                    if ui.button("Load session…").clicked() {
                        ui.close_menu();
                        self.load_session_dialog();
                    }

                    if ui.button("Close study").clicked() {
                        ui.close_menu();
                        self.close_study();
                    }

                    ui.separator();

                    if ui.button("Quit").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });

                // ── View ─────────────────────────────────────────────────────
                ui.menu_button("View", |ui| {
                    let layout_label = if self.multi_planar {
                        "⬛ Multi-planar 2×2 (active)"
                    } else {
                        "⬜ Single view (active)"
                    };
                    if ui.button(layout_label).clicked() {
                        ui.close_menu();
                        self.multi_planar = !self.multi_planar;
                        // Mark all textures dirty so the new layout renders
                        // immediately.
                        self.texture_dirty = true;
                        self.coronal_dirty = true;
                        self.sagittal_dirty = true;
                        self.mip_dirty = true;
                    }

                    ui.separator();

                    let overlay_label = if self.show_overlay {
                        "✔ Show Overlay"
                    } else {
                        " Show Overlay"
                    };
                    if ui.button(overlay_label).clicked() {
                        ui.close_menu();
                        self.show_overlay = !self.show_overlay;
                    }

                    let label_overlay_label = if self.show_label_overlay {
                        "✔ Show Label Overlay"
                    } else {
                        " Show Label Overlay"
                    };
                    if ui.button(label_overlay_label).clicked() {
                        ui.close_menu();
                        self.show_label_overlay = !self.show_label_overlay;
                    }

                    let rt_overlay_label = if self.show_rt_struct_overlay {
                        "✔ Show RT-STRUCT Overlay"
                    } else {
                        " Show RT-STRUCT Overlay"
                    };
                    if ui.button(rt_overlay_label).clicked() {
                        ui.close_menu();
                        self.show_rt_struct_overlay = !self.show_rt_struct_overlay;
                    }

                    let rt_dose_label = if self.show_rt_dose_overlay {
                        "\u{2714} Show RT-DOSE Overlay"
                    } else {
                        " Show RT-DOSE Overlay"
                    };
                    if ui.button(rt_dose_label).clicked() {
                        ui.close_menu();
                        self.show_rt_dose_overlay = !self.show_rt_dose_overlay;
                    }

                    ui.checkbox(&mut self.show_mesh_overlay, "Show Mesh Overlay");

                    let filter_label = if self.show_filter_panel {
                        "✔ Show Filter Panel"
                    } else {
                        " Show Filter Panel"
                    };
                    if ui.button(filter_label).clicked() {
                        ui.close_menu();
                        self.show_filter_panel = !self.show_filter_panel;
                    }

                    let xhair_label = if self.show_crosshair {
                        "✔ Show Crosshair"
                    } else {
                        " Show Crosshair"
                    };
                    if ui.button(xhair_label).clicked() {
                        ui.close_menu();
                        self.show_crosshair = !self.show_crosshair;
                    }

                    let browser_label = if self.show_series_browser {
                        "✔ Show Series Browser"
                    } else {
                        " Show Series Browser"
                    };
                    if ui.button(browser_label).clicked() {
                        ui.close_menu();
                        self.show_series_browser = !self.show_series_browser;
                    }

                    ui.separator();

                    // Colormap sub-menu kept in View.
                    ui.menu_button("Colormap", |ui| {
                        for &cm in Colormap::all() {
                            if ui
                                .selectable_label(self.colormap == cm, cm.label())
                                .clicked()
                                && self.colormap != cm
                            {
                                self.colormap = cm;
                                self.texture_dirty = true;
                                self.coronal_dirty = true;
                                self.sagittal_dirty = true;
                                self.mip_dirty = true;
                            }
                        }
                    });

                    ui.separator();

                    ui.label("Orientation");
                    if ui.button("Flip Horizontal [H]").clicked() {
                        ui.close_menu();
                        self.view_transform = self.view_transform.toggle_flip_h();
                        self.mark_all_textures_dirty();
                    }
                    if ui.button("Flip Vertical [V]").clicked() {
                        ui.close_menu();
                        self.view_transform = self.view_transform.toggle_flip_v();
                        self.mark_all_textures_dirty();
                    }
                    if ui.button("Rotate CW 90° [R]").clicked() {
                        ui.close_menu();
                        self.view_transform = self.view_transform.rotate_cw();
                        self.mark_all_textures_dirty();
                    }
                    if ui.button("Rotate CCW 90° [Shift+R]").clicked() {
                        ui.close_menu();
                        self.view_transform = self.view_transform.rotate_ccw();
                        self.mark_all_textures_dirty();
                    }
                    if ui.button("Reset Orientation [O]").clicked() {
                        ui.close_menu();
                        self.view_transform = self.view_transform.reset();
                        self.mark_all_textures_dirty();
                    }

                    ui.separator();

                    let colorbar_label = if self.show_colorbar {
                        "✔ Show Colorbar"
                    } else {
                        " Show Colorbar"
                    };
                    if ui.button(colorbar_label).clicked() {
                        ui.close_menu();
                        self.show_colorbar = !self.show_colorbar;
                    }
                });

                // ── Image ────────────────────────────────────────────────────
                ui.menu_button("Image", |ui| {
                    ui.menu_button("Axis", |ui| {
                        for plane in [
                            AnatomicalPlane::Axial,
                            AnatomicalPlane::Coronal,
                            AnatomicalPlane::Sagittal,
                        ] {
                            let idx = self.axis_for_plane(plane);
                            if ui
                                .selectable_label(self.axis == idx, plane.label())
                                .clicked()
                                && self.axis != idx
                            {
                                ui.close_menu();
                                self.axis = idx;
                                self.texture_dirty = true;
                            }
                        }
                    });

                    ui.separator();

                    if ui.button("Zoom to fit (Ctrl/Cmd+0)").clicked() {
                        ui.close_menu();
                        self.reset_view_to_fit();
                    }
                    if ui.button("Next slice").clicked() {
                        ui.close_menu();
                        self.step_slice(1);
                    }
                    if ui.button("Previous slice").clicked() {
                        ui.close_menu();
                        self.step_slice(-1);
                    }
                });

                // ── Window (W/L presets) ──────────────────────────────────────
                ui.menu_button("Window", |ui| {
                    ui.label("CT Presets");
                    ui.separator();
                    for preset in WindowPreset::ct_presets() {
                        if ui.button(preset.name).clicked() {
                            ui.close_menu();
                            self.apply_preset(*preset);
                        }
                    }

                    ui.separator();
                    ui.label("MR Presets");
                    ui.separator();
                    for preset in WindowPreset::mr_presets() {
                        if ui.button(preset.name).clicked() {
                            ui.close_menu();
                            self.apply_preset(*preset);
                        }
                    }

                    ui.separator();

                    if ui.button("Reset W/L").clicked() {
                        ui.close_menu();
                        let (wc, ww) = if let Some(vol) = &self.loaded {
                            let d = ModalityDisplay::for_modality(vol.modality.as_deref());
                            (d.window_center as f32, d.window_width as f32)
                        } else {
                            (128.0, 256.0)
                        };
                        self.viewer_state.window_center = Some(wc);
                        self.viewer_state.window_width = Some(ww);
                        self.texture_dirty = true;
                        self.coronal_dirty = true;
                        self.sagittal_dirty = true;
                        self.mip_dirty = true;
                    }
                });

                // ── PACS ───────────────────────────────────────────────────
                ui.menu_button("PACS", |ui| {
                    let pacs_label = if self.show_pacs_panel {
                        "✔ PACS Network Panel"
                    } else {
                        " PACS Network Panel"
                    };
                    if ui.button(pacs_label).clicked() {
                        ui.close_menu();
                        self.show_pacs_panel = !self.show_pacs_panel;
                    }
                });
            });
        });
    }
}
