//! Central-panel layout methods for [`SnapApp`].

use super::state::SnapApp;
use crate::ui::AnatomicalPlane;
// ── Layout constants ─────────────────────────────────────────────────────────

/// Fraction of the available height reserved for the MPR info panel.
const MPR_INFO_HEIGHT_FRAC: f32 = 0.24;

/// Minimum pixel height of the MPR info panel.
const MPR_INFO_MIN_H: f32 = 110.0;

/// Maximum pixel height of the MPR info panel.
const MPR_INFO_MAX_H: f32 = 210.0;

impl SnapApp {
    // ── Single-viewport central panel ────────────────────────────────────
    pub(crate) fn show_central_panel_single(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            if self.loaded.is_none() {
                ui.centered_and_justified(|ui| {
                    ui.label("Open a DICOM folder or NIfTI file via File menu to begin.");
                });
                return;
            }
            self.render_axis_viewport(ui, ctx, self.axis);
        });
    }

    // ── Multi-planar side-by-side viewport ───────────────────────────────────
    /// Render 2×2 MPR viewports (Coronal / Axial / Sagittal / 3D-MIP) with
    /// a shared Info panel row below.
    pub(crate) fn show_central_panel_multi(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let avail = ui.available_size();
            let info_h = (avail.y * MPR_INFO_HEIGHT_FRAC).clamp(MPR_INFO_MIN_H, MPR_INFO_MAX_H);
            let grid_h = (avail.y - info_h - 6.0).max(160.0);
            let row_h = grid_h / 2.0;
            let col_w = avail.x / 2.0;

            let axial_axis = self.axis_for_plane(AnatomicalPlane::Axial);
            let coronal_axis = self.axis_for_plane(AnatomicalPlane::Coronal);
            let sagittal_axis = self.axis_for_plane(AnatomicalPlane::Sagittal);

            ui.allocate_ui(egui::vec2(avail.x, grid_h), |ui| {
                ui.horizontal(|ui| {
                    ui.allocate_ui(egui::vec2(col_w, row_h), |ui| {
                        self.render_axis_viewport(ui, ctx, coronal_axis);
                    });
                    ui.allocate_ui(egui::vec2(col_w, row_h), |ui| {
                        self.render_axis_viewport(ui, ctx, axial_axis);
                    });
                });
                ui.horizontal(|ui| {
                    ui.allocate_ui(egui::vec2(col_w, row_h), |ui| {
                        self.render_axis_viewport(ui, ctx, sagittal_axis);
                    });
                    ui.allocate_ui(egui::vec2(col_w, row_h), |ui| {
                        self.render_mip_viewport(ui, ctx); // 3D-MIP
                    });
                });
            });

            ui.separator();
            ui.allocate_ui(egui::vec2(avail.x, info_h), |ui| {
                self.show_right_info_panel(ui);
            });
        });
    }

    pub(crate) fn show_central_panel_dual(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            if self.loaded.is_none() {
                ui.centered_and_justified(|ui| {
                    ui.label("Open a volume to use 2-plane layout.");
                });
                return;
            }

            let avail = ui.available_size();
            let info_h = (avail.y * MPR_INFO_HEIGHT_FRAC).clamp(MPR_INFO_MIN_H, MPR_INFO_MAX_H);
            let view_h = (avail.y - info_h - 6.0).max(120.0);
            let view_w = avail.x / 2.0;

            ui.allocate_ui(egui::vec2(avail.x, view_h), |ui| {
                ui.horizontal(|ui| {
                    ui.allocate_ui(egui::vec2(view_w, view_h), |ui| {
                        self.render_axis_viewport(ui, ctx, self.dual_axes[0]);
                    });
                    ui.allocate_ui(egui::vec2(view_w, view_h), |ui| {
                        self.render_axis_viewport(ui, ctx, self.dual_axes[1]);
                    });
                });
            });

            ui.separator();
            ui.allocate_ui(egui::vec2(avail.x, info_h), |ui| {
                self.show_right_info_panel(ui);
            });
        });
    }

    pub(crate) fn show_central_panel_compare(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            if self.loaded.is_none() {
                ui.centered_and_justified(|ui| {
                    ui.label("Open a primary volume to use compare layout.");
                });
                return;
            }

            let avail = ui.available_size();
            let info_h = (avail.y * MPR_INFO_HEIGHT_FRAC).clamp(MPR_INFO_MIN_H, MPR_INFO_MAX_H);
            let view_h = (avail.y - info_h - 6.0).max(120.0);
            let view_w = avail.x / 2.0;

            ui.allocate_ui(egui::vec2(avail.x, view_h), |ui| {
                ui.horizontal(|ui| {
                    ui.allocate_ui(egui::vec2(view_w, view_h), |ui| {
                        self.render_axis_viewport(ui, ctx, self.compare_axes[0]);
                    });
                    ui.allocate_ui(egui::vec2(view_w, view_h), |ui| {
                        self.render_secondary_compare_viewport(
                            ui,
                            ctx,
                            self.compare_axes[0],
                            self.compare_axes[1],
                        );
                    });
                });
            });

            ui.separator();
            ui.allocate_ui(egui::vec2(avail.x, info_h), |ui| {
                ui.columns(2, |cols| {
                    cols[0].heading("Primary");
                    self.show_right_info_panel(&mut cols[0]);

                    cols[1].heading(if self.compare_fused_overlay {
                        "Fused"
                    } else {
                        "Secondary"
                    });
                    if let Some(vol) = &self.loaded_secondary {
                        let [d, r, c] = vol.shape;
                        let [dz, dy, dx] = vol.spacing;
                        cols[1].label(format!("Dims: {d}x{r}x{c}"));
                        cols[1].label(format!("Spacing: {dz:.2}x{dy:.2}x{dx:.2} mm"));
                        cols[1].label(format!(
                            "Modality: {}",
                            vol.modality.as_deref().unwrap_or("—")
                        ));
                        cols[1].label(format!(
                            "Series: {}",
                            vol.series_description.as_deref().unwrap_or("—")
                        ));
                        if self.compare_fused_overlay {
                            cols[1].label(format!("Alpha: {:.2}", self.compare_fusion_alpha));
                        }
                    } else {
                        cols[1].label("No secondary volume loaded.");
                    }
                });
            });
        });
    }
}
