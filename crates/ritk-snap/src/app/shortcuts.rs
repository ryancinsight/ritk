use super::state::SnapApp;
use crate::ui::{fit_view_transform, tool_kind_for_key};

impl SnapApp {
    pub(crate) fn consume_global_shortcuts(&mut self, ctx: &egui::Context) {
        let zoom_to_fit = egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::Num0);
        let redo_shift_z = egui::KeyboardShortcut::new(
            egui::Modifiers {
                command: true,
                shift: true,
                ..Default::default()
            },
            egui::Key::Z,
        );
        let redo_y = egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::Y);
        let undo_z = egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::Z);

        if ctx.input_mut(|input| input.consume_shortcut(&zoom_to_fit)) {
            self.reset_view_to_fit();
        }

        if ctx.input_mut(|input| {
            input.consume_shortcut(&redo_shift_z) || input.consume_shortcut(&redo_y)
        }) {
            self.redo_label_edit_shortcut();
        }

        if ctx.input_mut(|input| input.consume_shortcut(&undo_z)) {
            self.undo_label_edit_shortcut();
        }

        let nav = ctx.input(|input| {
            (
                input.key_pressed(egui::Key::ArrowUp),
                input.key_pressed(egui::Key::ArrowDown),
                input.key_pressed(egui::Key::PageUp),
                input.key_pressed(egui::Key::PageDown),
                input.key_pressed(egui::Key::Home),
                input.key_pressed(egui::Key::End),
            )
        });
        self.apply_slice_navigation_shortcuts(nav.0, nav.1, nav.2, nav.3, nav.4, nav.5);

        // ── Tool selection shortcuts ──────────────────────────────────────────
        ctx.input(|input| {
            for key in &input.keys_down {
                if let Some(tool) = tool_kind_for_key(*key) {
                    self.active_tool = tool;
                    break;
                }
            }
        });

        // ── Viewport orientation shortcuts ────────────────────────────────────
        let (flip_h, flip_v, rotate_cw, rotate_ccw, reset_orient) = ctx.input(|input| {
            let shift = input.modifiers.shift;
            (
                input.key_pressed(egui::Key::H),
                input.key_pressed(egui::Key::V),
                !shift && input.key_pressed(egui::Key::R),
                shift && input.key_pressed(egui::Key::R),
                input.key_pressed(egui::Key::O),
            )
        });

        if flip_h {
            self.view_transform = self.view_transform.toggle_flip_h();
            self.mark_all_textures_dirty();
        }
        if flip_v {
            self.view_transform = self.view_transform.toggle_flip_v();
            self.mark_all_textures_dirty();
        }
        if rotate_cw {
            self.view_transform = self.view_transform.rotate_cw();
            self.mark_all_textures_dirty();
        }
        if rotate_ccw {
            self.view_transform = self.view_transform.rotate_ccw();
            self.mark_all_textures_dirty();
        }
        if reset_orient {
            self.view_transform = self.view_transform.reset();
            self.mark_all_textures_dirty();
        }
    }

    pub(crate) fn apply_slice_navigation_shortcuts(
        &mut self,
        arrow_up: bool,
        arrow_down: bool,
        page_up: bool,
        page_down: bool,
        home: bool,
        end: bool,
    ) {
        if arrow_up || page_up {
            self.step_slice(-1);
        } else if arrow_down || page_down {
            self.step_slice(1);
        } else if home {
            self.jump_active_axis_slice_boundary(false);
        } else if end {
            self.jump_active_axis_slice_boundary(true);
        }
    }

    fn jump_active_axis_slice_boundary(&mut self, end: bool) {
        let (_, total) = self.axis_slice_info(self.axis);
        let target = if end { total.saturating_sub(1) } else { 0 };
        self.set_slice_for_axis(self.axis, target);
    }

    pub(crate) fn reset_view_to_fit(&mut self) {
        let (pan_offset, zoom) = fit_view_transform();
        self.pan_offset = egui::Vec2::new(pan_offset[0], pan_offset[1]);
        self.zoom = zoom;
        self.texture_dirty = true;
        self.coronal_dirty = true;
        self.sagittal_dirty = true;
        self.mip_dirty = true;
        self.status_message = "Zoom reset to fit.".to_owned();
    }
}
