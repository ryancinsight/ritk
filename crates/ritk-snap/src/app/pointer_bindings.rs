//! Assignment of pointer buttons to viewer tools.
//!
//! The defaults reproduce the 2D viewer table of the
//! [RadiAnt mouse functions](https://www.radiantviewer.com/dicom-viewer-manual/mouse_functions.html):
//! the left button runs the selected tool, while the other buttons keep fixed
//! tools so a reader can adjust the display without leaving a measurement.

use crate::presentation::PointerButton;
use crate::tools::interaction::ToolState;
use crate::tools::kind::ToolKind;
use thiserror::Error;

/// Failure reassigning a pointer button to a viewer tool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub(crate) enum PointerBindingError {
    /// The left button always runs the selected tool.
    #[error("the left button runs the selected tool and cannot be reassigned")]
    LeftButton,
}

/// Tools assigned to the buttons other than the left button.
///
/// The assignments serialize to the host settings store so a reader's layout
/// survives a restart. [`ToolKind`] already carries the serde representation;
/// the struct order is the stored order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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

    /// Assign `tool` to a non-primary button.
    ///
    /// # Errors
    /// Returns [`PointerBindingError::LeftButton`] when `button` is
    /// [`PointerButton::Left`], which always runs the selected tool.
    pub(crate) fn assign(
        &mut self,
        button: PointerButton,
        tool: ToolKind,
    ) -> Result<(), PointerBindingError> {
        match button {
            PointerButton::Left => Err(PointerBindingError::LeftButton),
            PointerButton::Middle => {
                self.middle = tool;
                Ok(())
            }
            PointerButton::Right => {
                self.right = tool;
                Ok(())
            }
            PointerButton::X1 => {
                self.back = tool;
                Ok(())
            }
            PointerButton::X2 => {
                self.forward = tool;
                Ok(())
            }
        }
    }
}

/// The button and tool that own a gesture from press to release.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PointerGestureOwner {
    pub(crate) button: PointerButton,
    pub(crate) tool: ToolKind,
    /// The gesture runs a modifier override rather than the selected tool.
    ///
    /// A modified left press behaves like the equivalent non-primary button
    /// gesture: it never places measurement anchors and its release never runs
    /// the click path of the still-selected tool.
    pub(crate) modified: bool,
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

impl PointerGestures {
    /// Forget the running gesture and any measurement it suspended.
    ///
    /// Used when the state a gesture would return to no longer exists, such as
    /// after the volume is replaced or another tool is selected.
    pub(crate) fn abandon(&mut self) {
        self.owner = None;
        self.suspended_measurement = None;
    }
}
