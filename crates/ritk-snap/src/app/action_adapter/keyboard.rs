//! Viewer keyboard transitions for the host action adapter.

use super::{
    SnapApp, ViewerActionDisposition, VIRTUAL_KEY_ARROW_DOWN, VIRTUAL_KEY_ARROW_UP,
    VIRTUAL_KEY_CINE_FPS_DOWN, VIRTUAL_KEY_CINE_FPS_UP, VIRTUAL_KEY_CINE_TOGGLE,
    VIRTUAL_KEY_CROSSHAIR_TOGGLE, VIRTUAL_KEY_END, VIRTUAL_KEY_HOME, VIRTUAL_KEY_PAGE_DOWN,
    VIRTUAL_KEY_PAGE_UP,
};
use crate::presentation::PointerButton;
use crate::tools::interaction::ToolState;
use crate::ui::tool_kind_for_virtual_key;

impl SnapApp {
    pub(super) fn apply_virtual_key(
        &mut self,
        virtual_key: u32,
        repeated: bool,
    ) -> ViewerActionDisposition {
        if virtual_key == 0x1b && !repeated && self.pointer.pending_polygon_move.is_some() {
            self.on_drag_end(None);
            return ViewerActionDisposition::Continue { repaint: true };
        }
        if virtual_key == 0x1b
            && !repeated
            && (self.tool_state.is_pending_measurement()
                || matches!(
                    self.tool_state,
                    ToolState::DeviationDrag { .. }
                        | ToolState::ArrowDrag { .. }
                        | ToolState::PencilDrag { .. }
                        | ToolState::CobbDrag { .. }
                ))
        {
            self.tool_state = ToolState::Idle;
            self.pointer.abandon();
            return ViewerActionDisposition::Continue { repaint: true };
        }
        if virtual_key == 0x0d && !repeated && self.finish_pending_polygon() {
            if self.tool_state.is_idle() {
                self.pointer.owner = self
                    .pointer
                    .owner
                    .filter(|owner| owner.button != PointerButton::Left || owner.modified);
            }
            return ViewerActionDisposition::Continue { repaint: true };
        }
        if virtual_key == 0x2e && !repeated {
            return ViewerActionDisposition::Continue {
                repaint: self.delete_selected_polygon(),
            };
        }
        if virtual_key == VIRTUAL_KEY_CROSSHAIR_TOGGLE {
            if repeated || self.loaded.is_none() {
                return ViewerActionDisposition::Continue { repaint: false };
            }
            self.show_crosshair = !self.show_crosshair;
            self.status_message = if self.show_crosshair {
                "Linked crosshair shown (X)".to_owned()
            } else {
                "Linked crosshair hidden (X)".to_owned()
            };
            return ViewerActionDisposition::Continue { repaint: true };
        }
        if virtual_key == VIRTUAL_KEY_CINE_TOGGLE {
            if repeated {
                return ViewerActionDisposition::Continue { repaint: false };
            }
            return ViewerActionDisposition::Continue {
                repaint: self.toggle_cine(),
            };
        }
        if virtual_key == VIRTUAL_KEY_CINE_FPS_UP || virtual_key == VIRTUAL_KEY_CINE_FPS_DOWN {
            if repeated {
                return ViewerActionDisposition::Continue { repaint: false };
            }
            let delta = if virtual_key == VIRTUAL_KEY_CINE_FPS_UP {
                1.0
            } else {
                -1.0
            };
            return ViewerActionDisposition::Continue {
                repaint: self.adjust_cine_fps(delta),
            };
        }
        if let Some(tool) = tool_kind_for_virtual_key(virtual_key) {
            if repeated {
                return ViewerActionDisposition::Continue { repaint: false };
            }
            self.select_tool(tool);
            return ViewerActionDisposition::Continue { repaint: true };
        }
        let (arrow_up, arrow_down, page_up, page_down, home, end) = match virtual_key {
            VIRTUAL_KEY_PAGE_UP => (false, false, true, false, false, false),
            VIRTUAL_KEY_PAGE_DOWN => (false, false, false, true, false, false),
            VIRTUAL_KEY_END => (false, false, false, false, false, true),
            VIRTUAL_KEY_HOME => (false, false, false, false, true, false),
            VIRTUAL_KEY_ARROW_UP => (true, false, false, false, false, false),
            VIRTUAL_KEY_ARROW_DOWN => (false, true, false, false, false, false),
            _ => return ViewerActionDisposition::Continue { repaint: false },
        };
        self.apply_slice_navigation_shortcuts(arrow_up, arrow_down, page_up, page_down, home, end);
        ViewerActionDisposition::Continue { repaint: true }
    }
}
