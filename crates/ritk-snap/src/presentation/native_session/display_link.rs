//! Transfer only the display components changed by a reader gesture.
use super::{synchronization::PositionSync, NativeViewerSession};
use crate::{app::SnapApp, tools::interaction::ViewportOffset, ui::pan_from_drag_delta};

#[derive(Clone, Copy, PartialEq)]
pub(super) struct DisplayState {
    zoom: f32,
    pan: ViewportOffset,
    window: (f32, f32),
}

impl DisplayState {
    pub(super) fn read(app: &SnapApp) -> Self {
        Self {
            zoom: app.zoom,
            pan: app.pan_offset,
            window: app.browser_window_level_values(),
        }
    }
}

impl NativeViewerSession {
    pub(super) fn link_display(
        &mut self,
        source: usize,
        before: DisplayState,
        after: DisplayState,
    ) {
        let mode = self.position_sync;
        let navigation = self.window_chrome.link_navigation;
        let window = self.window_chrome.link_window;
        if mode == PositionSync::Off || !(navigation || window) || before == after {
            return;
        }
        self.for_each_comparison(source, |source, target| {
            let Some((s, t)) = source
                .patient_stack
                .as_ref()
                .zip(target.patient_stack.as_ref())
            else {
                return;
            };
            let eligible = match mode {
                PositionSync::Automatic => s.can_link(t),
                PositionSync::Manual => s.anchor.is_some() && t.anchor.is_some() && s.can_anchor(t),
                PositionSync::Off => false,
            };
            if !eligible {
                return;
            }
            // A common screen translation and scale ratio preserve the reader's
            // existing offsets and relative magnifications across unlike fits.
            if navigation && before.pan != after.pan {
                let delta = ViewportOffset::new(
                    after.pan.x() - before.pan.x(),
                    after.pan.y() - before.pan.y(),
                );
                if let Ok(pan) = pan_from_drag_delta(target.pan_offset, delta) {
                    target.pan_offset = pan;
                }
            }
            if navigation && before.zoom != after.zoom {
                let zoom = if target.zoom == before.zoom {
                    after.zoom
                } else {
                    target.zoom * (after.zoom / before.zoom)
                };
                if zoom.is_finite() && zoom > 0.0 {
                    // Explicit 1:1 may lie beyond the normal interactive range.
                    // Retain such values when synchronizing equal starting fits.
                    target.zoom = zoom.clamp(
                        crate::ui::zoom::MIN_ZOOM.min(after.zoom),
                        crate::ui::zoom::MAX_ZOOM.max(after.zoom),
                    );
                }
            }
            if window
                && before.window != after.window
                && target.browser_window_level_values() != after.window
            {
                target.viewer_state.window_center = Some(after.window.0);
                target.viewer_state.window_width = Some(after.window.1);
                target.bump_visual_revision();
            }
        });
    }
}
