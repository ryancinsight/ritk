//! Unmodified native shortcuts for measurements and image readouts.

use super::{WindowAction, WorkspaceLayout};
use crate::tools::kind::ToolKind;

pub(super) fn action(key: u32, layout: WorkspaceLayout) -> Option<WindowAction> {
    let tool = match char::from_u32(key)? {
        'L' => ToolKind::MeasureLength,
        'E' => ToolKind::RoiEllipse,
        'G' => ToolKind::RoiClosedPolygon,
        'O' => ToolKind::RoiOpenPolygon,
        'A' => ToolKind::MeasureAngle,
        'C' => ToolKind::MeasureCobb,
        'D' => ToolKind::MeasureDeviation,
        'Y' => ToolKind::Arrow,
        'P' => ToolKind::Pencil,
        'Q' if !layout.is_grid() => ToolKind::Crosshair,
        _ if key == 0x7b && !layout.is_grid() => return Some(WindowAction::ToggleOverlay),
        _ => return None,
    };
    Some(WindowAction::SelectTool(tool))
}
