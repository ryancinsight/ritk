use super::state::SeriesLoadTarget;
use super::state::SnapApp;
use crate::tools::interaction::ToolState;
use crate::tools::kind::ToolKind;
use crate::viewer::{DEFAULT_WINDOW_CENTER, DEFAULT_WINDOW_WIDTH};

#[cfg(not(target_arch = "wasm32"))]
use rfd::FileDialog;

#[cfg(target_arch = "wasm32")]
struct FileDialog;

#[cfg(target_arch = "wasm32")]
impl FileDialog {
    fn new() -> Self {
        Self
    }
    fn pick_folder(self) -> Option<std::path::PathBuf> {
        None
    }
}

impl SnapApp {
    pub(crate) fn show_ribbon_toolbar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("ribbon_toolbar")
            .frame(
                egui::Frame::none()
                    .fill(egui::Color32::from_rgb(45, 45, 45))
                    .inner_margin(egui::Margin::symmetric(4.0, 2.0)),
            )
            .show(ctx, |ui| {
                ui.set_min_height(36.0);
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(4.0, 2.0);
                    ui.spacing_mut().button_padding = egui::vec2(2.0, 2.0);

                    ui.menu_button("File", |ui| {
                        if ui.button("Open Primary Series").clicked() {
                            ui.close_menu();
                            if let Some(folder) = FileDialog::new().pick_folder() {
                                self.scan_for_series(folder.clone());
                                self.pending_load = Some(folder);
                            }
                        }
                        if ui.button("Open Secondary Series").clicked() {
                            ui.close_menu();
                            if let Some(folder) = FileDialog::new().pick_folder() {
                                self.scan_for_series(folder.clone());
                                self.pending_secondary_load = Some(folder);
                            }
                        }
                        ui.separator();
                        if ui.button("Swap Primary/Secondary").clicked() {
                            ui.close_menu();
                            std::mem::swap(&mut self.loaded, &mut self.loaded_secondary);
                            std::mem::swap(&mut self.colormap, &mut self.secondary_colormap);
                            self.texture = None;
                            self.secondary_texture = None;
                            self.mark_all_textures_dirty();
                            self.refresh_cached_histogram();
                        }
                    });

                    ui.menu_button("Layout", |ui| {
                        let single_active =
                            !self.multi_planar && !self.dual_plane && !self.compare_side_by_side;
                        if ui
                            .selectable_label(single_active, "Single (1-Up)")
                            .clicked()
                        {
                            self.multi_planar = false;
                            self.dual_plane = false;
                            self.compare_side_by_side = false;
                            self.compare_fused_overlay = false;
                            self.mark_all_textures_dirty();
                            ui.close_menu();
                        }
                        if ui
                            .selectable_label(self.dual_plane, "Dual Plane (2-Up)")
                            .clicked()
                        {
                            self.dual_plane = true;
                            self.multi_planar = false;
                            self.compare_side_by_side = false;
                            self.compare_fused_overlay = false;
                            self.mark_all_textures_dirty();
                            ui.close_menu();
                        }
                        if ui
                            .selectable_label(self.multi_planar, "MPR Grid (3-Up)")
                            .clicked()
                        {
                            self.multi_planar = true;
                            self.dual_plane = false;
                            self.compare_side_by_side = false;
                            self.compare_fused_overlay = false;
                            self.mark_all_textures_dirty();
                            ui.close_menu();
                        }
                        if ui
                            .selectable_label(
                                self.compare_side_by_side,
                                "Compare (Primary/Secondary)",
                            )
                            .clicked()
                        {
                            self.compare_side_by_side = true;
                            self.multi_planar = false;
                            self.dual_plane = false;
                            self.mark_all_textures_dirty();
                            ui.close_menu();
                        }
                    });

                    ui.menu_button("Target", |ui| {
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

                    ui.menu_button("Axes", |ui| {
                        if self.dual_plane {
                            ui.label("Dual-Plane (Left/Right)");
                            for (axis_side, side_label) in [(0, "Left"), (1, "Right")] {
                                ui.horizontal(|ui| {
                                    ui.label(side_label);
                                    for (name, idx) in [("Ax", 0), ("Co", 1), ("Sa", 2)] {
                                        let is_active = if axis_side == 0 {
                                            self.dual_axes[0] == idx
                                        } else {
                                            self.dual_axes[1] == idx
                                        };
                                        if ui.selectable_label(is_active, name).clicked() {
                                            if axis_side == 0 {
                                                self.dual_axes[0] = idx;
                                            } else {
                                                self.dual_axes[1] = idx;
                                            }
                                        }
                                    }
                                });
                            }
                            ui.separator();
                        }

                        if self.compare_side_by_side {
                            ui.label("Compare (Primary/Secondary)");
                            for (side_num, side_label) in [(0, "Primary"), (1, "Secondary")] {
                                ui.horizontal(|ui| {
                                    ui.label(side_label);
                                    for (name, idx) in [("Ax", 0), ("Co", 1), ("Sa", 2)] {
                                        let is_active = if side_num == 0 {
                                            self.compare_axes[0] == idx
                                        } else {
                                            self.compare_axes[1] == idx
                                        };
                                        if ui.selectable_label(is_active, name).clicked() {
                                            if side_num == 0 {
                                                self.compare_axes[0] = idx;
                                                self.texture_dirty = true;
                                            } else {
                                                self.compare_axes[1] = idx;
                                                self.secondary_texture_dirty = true;
                                            }
                                        }
                                    }
                                });
                            }
                        }

                        if !self.dual_plane && !self.compare_side_by_side {
                            ui.label("Axis controls appear in Dual Plane and Compare layouts.");
                        }
                    });

                    ui.menu_button("Compare", |ui| {
                        if self.compare_side_by_side {
                            if ui.button("Preset Ax | Ax").clicked() {
                                self.compare_axes = [0, 0];
                                self.texture_dirty = true;
                                self.secondary_texture_dirty = true;
                            }
                            if ui.button("Preset Co | Co").clicked() {
                                self.compare_axes = [1, 1];
                                self.texture_dirty = true;
                                self.secondary_texture_dirty = true;
                            }
                            if ui.button("Preset Sa | Sa").clicked() {
                                self.compare_axes = [2, 2];
                                self.texture_dirty = true;
                                self.secondary_texture_dirty = true;
                            }
                            ui.separator();
                            ui.label("Secondary W/L");
                            let mut c = self
                                .secondary_window_center
                                .unwrap_or(DEFAULT_WINDOW_CENTER);
                            let mut w = self
                                .secondary_window_width
                                .unwrap_or(DEFAULT_WINDOW_WIDTH)
                                .max(1.0);
                            let c_changed = ui
                                .add(
                                    egui::DragValue::new(&mut c)
                                        .speed(1.0)
                                        .fixed_decimals(0)
                                        .prefix("C:"),
                                )
                                .changed();
                            let w_changed = ui
                                .add(
                                    egui::DragValue::new(&mut w)
                                        .speed(1.0)
                                        .fixed_decimals(0)
                                        .prefix("W:"),
                                )
                                .changed();
                            if c_changed || w_changed {
                                self.secondary_window_center = Some(c);
                                self.secondary_window_width = Some(w.max(1.0));
                                self.secondary_texture_dirty = true;
                            }
                            ui.separator();
                            if ui
                                .checkbox(&mut self.compare_fused_overlay, "Fused Overlay")
                                .changed()
                            {
                                self.secondary_texture_dirty = true;
                            }
                            if self.compare_fused_overlay {
                                let alpha_changed = ui
                                    .add(
                                        egui::Slider::new(
                                            &mut self.compare_fusion_alpha,
                                            0.0..=1.0,
                                        )
                                        .text("Secondary Alpha"),
                                    )
                                    .changed();
                                if alpha_changed {
                                    self.secondary_texture_dirty = true;
                                }
                            }
                        } else {
                            ui.label("Enable Compare layout to configure compare options.");
                        }
                    });

                    ui.menu_button("Tools", |ui| {
                        for (label, tool) in [
                            ("Pan", ToolKind::Pan),
                            ("Zoom", ToolKind::Zoom),
                            ("W/L", ToolKind::WindowLevel),
                            ("Length", ToolKind::MeasureLength),
                            ("Angle", ToolKind::MeasureAngle),
                            ("Paint", ToolKind::LabelPaint),
                        ] {
                            if ui
                                .selectable_label(self.active_tool == tool, label)
                                .on_hover_text(label)
                                .clicked()
                            {
                                self.active_tool = tool;
                                self.tool_state = ToolState::Idle;
                                ui.close_menu();
                            }
                        }
                    });

                    // Right-align status
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.small(format!("v{}", env!("CARGO_PKG_VERSION")));
                    });
                });
            });
    }
}
