use super::super::state::SnapApp;
use crate::tools::interaction::{Annotation, ImagePoint, ToolState};
use metis_ui_lang::{PathCapture, PathPush};

impl SnapApp {
    pub(super) fn pencil_start(&self, pos: ImagePoint) -> ToolState {
        ToolState::PencilDrag {
            capture: PathCapture::new(pos),
        }
    }

    pub(super) fn pencil_drag(&mut self, pos: ImagePoint) {
        let ToolState::PencilDrag { mut capture } = self.tool_state.clone() else {
            return;
        };
        if matches!(capture.push(pos), PathPush::Full) {
            self.status_message = "Pencil sample limit reached".to_owned();
        }
        self.tool_state = ToolState::PencilDrag { capture };
    }

    pub(super) fn finish_pencil(
        &mut self,
        location: Option<crate::tools::interaction::AnnotationLocation>,
    ) {
        let ToolState::PencilDrag { capture } = self.tool_state.clone() else {
            return;
        };
        if capture.as_slice().len() >= 2 {
            self.push_annotation(Annotation::Pencil {
                location,
                points: capture
                    .as_slice()
                    .iter()
                    .map(|point| [point.y(), point.x()])
                    .collect(),
            });
        }
    }
}
