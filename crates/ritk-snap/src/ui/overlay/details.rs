//! Metadata disclosure interaction and popup ownership.

use super::{Color32, FontId, OverlayRenderer, Rect, MARGIN, OVERLAY_FONT_SIZE};

impl OverlayRenderer {
    /// Expose overflow annotations through a visible Details control and popup.
    ///
    /// A control that cannot fit within the image is allocated in the surrounding
    /// UI flow. The scrollable popup retains the complete metadata strings.
    /// Arrow keys scroll by a text line, Page keys by the viewport height, and
    /// Home/End reach the content boundaries. Escape dismisses the popup.
    pub fn show_details(ui: &mut egui::Ui, rect: Rect, text: &str) {
        let rect = rect.intersect(ui.clip_rect());
        let label = egui::RichText::new("Details").color(Color32::WHITE);
        let size = ui
            .painter()
            .layout_no_wrap(
                "Details".to_owned(),
                FontId::proportional(OVERLAY_FONT_SIZE),
                Color32::WHITE,
            )
            .size()
            + 2.0 * ui.spacing().button_padding;
        let button = egui::Button::new(label.size(OVERLAY_FONT_SIZE)).fill(Color32::BLACK);
        let bounds = Rect::from_min_size(rect.min + egui::vec2(MARGIN, MARGIN), size);
        let response = if rect.contains_rect(bounds) {
            ui.put(bounds, button)
        } else {
            ui.advance_cursor_after_rect(rect);
            ui.add(button)
        };
        let popup_id = response.id.with("metadata-popup");
        if response.clicked() {
            ui.memory_mut(|memory| memory.toggle_popup(popup_id));
        }
        retain_popup_owner(ui.ctx(), popup_id);
        // Bound the popup content by the screen interior minus its frame. The
        // popup area can reposition at screen edges; long metadata scrolls.
        let available = ui.ctx().screen_rect().shrink(MARGIN).size()
            - egui::Frame::popup(ui.style()).total_margin().sum();
        let width = rect
            .width()
            .max(response.rect.width())
            .min(available.x)
            .max(1.0);
        egui::popup::popup_below_widget(
            ui,
            popup_id,
            &response,
            egui::popup::PopupCloseBehavior::CloseOnClickOutside,
            |ui| {
                ui.set_width(width);
                egui::ScrollArea::vertical()
                    .animated(false)
                    .max_height(available.y.max(1.0))
                    .show(ui, |ui| {
                        let label = ui.label(text);
                        let line = ui.text_style_height(&egui::TextStyle::Body);
                        let page = available.y.max(line);
                        let delta = ui.input_mut(|input| {
                            [
                                (egui::Key::ArrowUp, line),
                                (egui::Key::ArrowDown, -line),
                                (egui::Key::PageUp, page),
                                (egui::Key::PageDown, -page),
                                (egui::Key::Home, label.rect.height()),
                                (egui::Key::End, -label.rect.height()),
                            ]
                            .into_iter()
                            .filter_map(|(key, delta)| {
                                input
                                    .consume_key(egui::Modifiers::NONE, key)
                                    .then_some(delta)
                            })
                            .sum::<f32>()
                        });
                        ui.scroll_with_delta(egui::vec2(0.0, delta));
                    });
            },
        );
    }
}

#[derive(Clone, Copy)]
struct PopupOwner {
    id: egui::Id,
    frame: u64,
}

fn owner_key(ctx: &egui::Context) -> egui::Id {
    egui::Id::new(("metadata-popup-owner", ctx.viewport_id()))
}

fn retain_popup_owner(ctx: &egui::Context, popup_id: egui::Id) {
    let registration = egui::Id::new("metadata-popup-lifecycle");
    let installed = ctx.data_mut(|state| {
        let installed = state.get_temp::<bool>(registration).unwrap_or(false);
        state.insert_temp(registration, true);
        installed
    });
    if !installed {
        // The egui plugin boundary requires an erased callback; registration is
        // once per context, outside the per-text rendering path.
        ctx.on_end_frame(
            "metadata-popup-lifecycle",
            std::sync::Arc::new(dismiss_absent_owner),
        );
    }
    if ctx.memory(|memory| memory.is_popup_open(popup_id)) {
        let owner = PopupOwner {
            id: popup_id,
            frame: ctx.frame_nr(),
        };
        let key = owner_key(ctx);
        ctx.data_mut(|state| state.insert_temp(key, owner));
    }
}

fn dismiss_absent_owner(ctx: &egui::Context) {
    let key = owner_key(ctx);
    let Some(owner) = ctx.data(|state| state.get_temp::<PopupOwner>(key)) else {
        return;
    };
    let owned_popup_open = ctx.memory(|memory| memory.is_popup_open(owner.id));
    if owner.frame != ctx.frame_nr() || !owned_popup_open {
        if owned_popup_open {
            ctx.memory_mut(egui::Memory::close_popup);
            ctx.request_repaint();
        }
        ctx.data_mut(|state| state.remove::<PopupOwner>(key));
    }
}
