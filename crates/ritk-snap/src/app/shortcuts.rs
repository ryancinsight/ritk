use super::state::SnapApp;
use crate::tools::interaction::ViewportOffset;
use crate::ui::{fit_view_transform, tool_kind_for_key};

impl SnapApp {
    pub(crate) fn consume_global_shortcuts(&mut self, ctx: &egui::Context) {
        // Popups own keyboard navigation before the viewport processes a frame.
        if ctx.memory(egui::Memory::any_popup_open) {
            return;
        }
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

        if ctx.input(|input| input.key_pressed(egui::Key::Space)) {
            self.toggle_cine();
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
            self.bump_visual_revision();
        }
        if flip_v {
            self.view_transform = self.view_transform.toggle_flip_v();
            self.bump_visual_revision();
        }
        if rotate_cw {
            self.view_transform = self.view_transform.rotate_cw();
            self.bump_visual_revision();
        }
        if rotate_ccw {
            self.view_transform = self.view_transform.rotate_ccw();
            self.bump_visual_revision();
        }
        if reset_orient {
            self.view_transform = self.view_transform.reset();
            self.bump_visual_revision();
        }
    }

    pub(crate) fn reset_view_to_fit(&mut self) {
        let (pan_offset, zoom) = fit_view_transform();
        self.pan_offset = ViewportOffset::new(pan_offset[0], pan_offset[1]);
        self.zoom = zoom;
        self.bump_visual_revision();
        self.status_message = "Zoom reset to fit.".to_owned();
    }
}
