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
        let Some(grid) = self.workspace_layout.grid() else {
            return;
        };
        let link_zoom = navigation && before.zoom != after.zoom;
        // Read each pane's own rotated, spacing-aware fit before borrowing its
        // app mutably. The fixed-capacity list follows visible grid order.
        let pixel_zooms: arrayvec::ArrayVec<_, { super::layout::MAX_GRID_PANELS }> = if link_zoom {
            (0..grid.panel_count())
                .map(|panel| self.one_to_one_zoom(panel))
                .collect()
        } else {
            Default::default()
        };
        self.for_each_comparison(source, |index, source, target| {
            if source.axis != 0 || target.axis != 0 {
                return;
            }
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
            if link_zoom {
                target.zoom = crate::ui::zoom::linked_zoom(
                    target.zoom,
                    before.zoom,
                    after.zoom,
                    pixel_zooms[index],
                );
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
