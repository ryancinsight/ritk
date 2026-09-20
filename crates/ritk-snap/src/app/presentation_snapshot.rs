//! Projection of mutable viewer state into the host-neutral presentation contract.

use super::SnapApp;
use crate::presentation::PresentationSnapshot;
use crate::render::WindowLevel;
use crate::tools::kind::ToolKind;
use crate::viewer::{DEFAULT_WINDOW_CENTER, DEFAULT_WINDOW_WIDTH};

impl SnapApp {
    /// Project viewer state into the host-neutral presentation contract.
    pub(crate) fn presentation_snapshot(&self) -> PresentationSnapshot {
        let slice_indices = std::array::from_fn(|axis| self.axis_slice_info(axis).0);
        let slice_counts = std::array::from_fn(|axis| self.axis_slice_info(axis).1);
        let window_level = WindowLevel::new(
            f64::from(
                self.viewer_state
                    .window_center
                    .unwrap_or(DEFAULT_WINDOW_CENTER),
            ),
            f64::from(
                self.viewer_state
                    .window_width
                    .unwrap_or(DEFAULT_WINDOW_WIDTH)
                    .max(1.0),
            ),
        );
        let active_tool_index = ToolKind::all()
            .iter()
            .position(|tool| *tool == self.active_tool)
            .expect("invariant: active tool belongs to ToolKind::all");
        PresentationSnapshot::from_parts(
            self.visual_revision,
            self.loaded.is_some(),
            self.axis,
            slice_indices,
            slice_counts,
            window_level,
            self.cine.enabled,
            self.cine.fps,
            self.zoom,
            self.pan_offset,
            None,
            active_tool_index,
            self.active_tool.label(),
        )
    }
}
