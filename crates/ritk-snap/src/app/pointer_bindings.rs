//! Assignment of pointer buttons to viewer tools.
//!
//! The defaults reproduce the 2D viewer table of the
//! [RadiAnt mouse functions](https://www.radiantviewer.com/dicom-viewer-manual/mouse_functions.html):
//! the left button runs the selected tool, while the other buttons keep fixed
//! tools so a reader can adjust the display without leaving a measurement.

use crate::presentation::PointerButton;
use crate::tools::interaction::ToolState;
use crate::tools::kind::ToolKind;

/// Tools assigned to the buttons other than the left button.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PointerBindings {
    middle: ToolKind,
    right: ToolKind,
    back: ToolKind,
    forward: ToolKind,
}

impl Default for PointerBindings {
    fn default() -> Self {
        Self {
            middle: ToolKind::WindowLevel,
            right: ToolKind::Zoom,
            back: ToolKind::Pan,
            forward: ToolKind::MeasureLength,
        }
    }
}

impl PointerBindings {
    /// Resolve the tool a press of `button` starts.
    pub(crate) const fn tool_for(self, button: PointerButton, selected: ToolKind) -> ToolKind {
        match button {
            PointerButton::Left => selected,
            PointerButton::Middle => self.middle,
            PointerButton::Right => self.right,
            PointerButton::X1 => self.back,
            PointerButton::X2 => self.forward,
        }
    }
}

/// The button and tool that own a gesture from press to release.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PointerGestureOwner {
    pub(crate) button: PointerButton,
    pub(crate) tool: ToolKind,
}

/// Pointer state of one viewer panel.
#[derive(Debug, Clone, Default)]
pub(crate) struct PointerGestures {
    /// Tools assigned to the non-primary buttons.
    pub(crate) bindings: PointerBindings,
    /// Button and tool that own the gesture in progress.
    pub(crate) owner: Option<PointerGestureOwner>,
    /// Multi-click measurement held while another gesture runs.
    pub(crate) suspended_measurement: Option<ToolState>,
}
