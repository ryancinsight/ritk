//! Anatomical plane assignment and menu focus for the shared workstation.
use super::{window_controls::Menu, NativeViewerSession, WindowAction};
use crate::ui::AnatomicalPlane;

impl NativeViewerSession {
    pub(super) fn focus_menu_pane(&mut self, action: WindowAction) {
        if let Some(pane) = self.menu_pane.take()
            && matches!(action, WindowAction::OpenMenu(Menu::View | Menu::Anonymize))
            && let Some(view) = self.viewports.get(pane).filter(|v| v.panel_width > 0)
        {
            if self.workspace_layout.is_grid() {
                self.active_panel = pane;
            } else {
                self.app.axis = view.axis();
            }
            self.menu_pane = Some(pane);
        }
    }

    pub(super) fn set_anatomical_plane(&mut self, plane: AnatomicalPlane) -> bool {
        if self.active_app().loaded.is_none() {
            return false;
        }
        let axis = self.active_app().axis_for_plane(plane);
        if !self.workspace_layout.is_grid() {
            // Swapping preserves a bijection between the three pane slots and source axes.
            let from = self
                .mpr_axes
                .iter()
                .position(|&a| a == self.app.axis)
                .expect("displayed axis");
            let to = self
                .mpr_axes
                .iter()
                .position(|&a| a == axis)
                .expect("anatomical axis");
            self.mpr_axes.swap(from, to);
        }
        self.end_held_gestures();
        self.active_app_mut().axis = axis;
        true
    }
}
