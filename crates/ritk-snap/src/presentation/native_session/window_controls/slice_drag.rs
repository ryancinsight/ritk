//! Pointer ownership for a native image pane's slice scrollbar.

use anyhow::Result;
use metis_ui_lang::ScrollbarDrag;

use super::super::layout::{screen_coordinate, NativeViewport, WorkspaceLayout};
use super::{PointerOwner, WindowAction, WindowChrome, WindowChromeEvent};
use crate::presentation::{PointerButton, PresentationEvent};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct SliceDrag {
    drag: ScrollbarDrag,
    panel_index: usize,
    axis: usize,
}

impl SliceDrag {
    pub(super) fn begin(
        viewports: &[NativeViewport],
        layout: WorkspaceLayout,
        x: f64,
        y: f64,
    ) -> Result<Option<(Self, WindowAction)>> {
        let x = screen_coordinate(x, "slice scrollbar x")?;
        let y = screen_coordinate(y, "slice scrollbar y")?;
        for (index, viewport) in viewports.iter().enumerate() {
            let Some(scrollbar) = viewport
                .scrollbar
                .filter(|bar| bar.geometry.track_contains(x, y))
            else {
                continue;
            };
            let geometry = scrollbar.geometry;
            let thumb_grab = geometry.grab_offset(x, y);
            let grab = thumb_grab.unwrap_or(geometry.thumb().height / 2);
            let clicked_offset = thumb_grab
                .is_none()
                .then(|| geometry.offset_for_pointer(y, grab));
            let offset = clicked_offset.unwrap_or(scrollbar.offset);
            let drag = Self {
                drag: geometry.begin_drag(y, grab, offset),
                panel_index: if layout.is_grid() { index } else { 0 },
                axis: viewport.axis(),
            };
            let slice = clicked_offset.map(usize::try_from).transpose()?;
            return Ok(Some((drag, drag.action(slice))));
        }
        Ok(None)
    }

    fn action(self, index: Option<usize>) -> WindowAction {
        WindowAction::SelectSlice {
            panel_index: self.panel_index,
            axis: self.axis,
            index,
        }
    }

    fn drag_to(self, y: f64) -> Result<WindowAction> {
        let y = screen_coordinate(y, "slice scrollbar drag y")?;
        let slice = usize::try_from(self.drag.offset_for_pointer(y))?;
        Ok(self.action(Some(slice)))
    }
}

impl WindowChrome {
    pub(super) fn handle_slice_drag(
        &mut self,
        event: &PresentationEvent,
    ) -> Result<Option<WindowChromeEvent>> {
        let PointerOwner::Slice(drag) = self.pointer_owner else {
            return Ok(None);
        };
        let action = match *event {
            PresentationEvent::PointerMove { y, .. } => Some(drag.drag_to(y)?),
            PresentationEvent::PointerUp {
                y,
                button: PointerButton::Left,
                ..
            } => {
                let action = Some(drag.drag_to(y)?);
                self.cancel_slice_drag();
                self.release(PointerButton::Left);
                action
            }
            PresentationEvent::PointerCancel {
                button: PointerButton::Left,
                ..
            } => {
                self.cancel_slice_drag();
                self.release(PointerButton::Left);
                None
            }
            PresentationEvent::PointerDown {
                button: PointerButton::Left,
                ..
            } => {
                self.cancel_slice_drag();
                return Ok(None);
            }
            PresentationEvent::PointerDown { button, .. } => {
                self.press(button);
                None
            }
            PresentationEvent::PointerUp { button, .. }
            | PresentationEvent::PointerCancel { button, .. } => {
                self.release(button);
                None
            }
            PresentationEvent::PointerWheel { .. } => None,
            _ => return Ok(None),
        };
        Ok(Some(WindowChromeEvent {
            consumed: true,
            repaint: false,
            action,
        }))
    }
}
