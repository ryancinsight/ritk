use super::super::state::SnapApp;
use crate::tools::interaction::{Annotation, ImagePoint, ToolState};
use metis_ui_lang::PathCapture;

impl SnapApp {
    pub(super) fn pencil_start(&self, pos: ImagePoint) -> ToolState {
        ToolState::PencilDrag {
            capture: PathCapture::new(pos),
        }
    }

    pub(super) fn finish_pencil(
        &mut self,
        location: Option<crate::tools::interaction::AnnotationLocation>,
    ) {
        let ToolState::PencilDrag { capture } = &self.tool_state else {
            return;
        };
        if capture.as_slice().len() < 2 {
            return;
        }
        let points = capture
            .as_slice()
            .iter()
            .map(|point| [point.y(), point.x()])
            .collect();
        self.push_annotation(Annotation::Pencil { location, points });
    }
}
