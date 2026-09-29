use super::*;

impl SnapApp {
    pub(crate) fn update_linked_cursor_from_voxel(&mut self, voxel: [usize; 3]) {
        let Some(volume) = &self.loaded else {
            return;
        };
        if voxel
            .into_iter()
            .zip(volume.shape)
            .any(|(coordinate, extent)| coordinate >= extent)
        {
            return;
        }
        let Some(cursor) = self.linked_cursor.as_mut() else {
            return;
        };
        cursor.set_voxel(volume.shape, voxel);
        self.viewer_state.slice_index = voxel[0];
        self.coronal_slice = voxel[1];
        self.sagittal_slice = voxel[2];
        self.bump_visual_revision();
        self.status_message = format!(
            "Linked cursor voxel=[{},{},{}]",
            voxel[0], voxel[1], voxel[2]
        );
    }

    pub(crate) fn on_oblique_click(&mut self, point: PatientPointMm) {
        match self.active_tool {
            ToolKind::MeasureLength => match self.tool_state.clone() {
                ToolState::PatientLength1 { p1 } => match PatientLength::try_new(p1, point) {
                    Ok(length) => {
                        self.status_message = format!("Length: {:.1} mm", length.length_mm());
                        self.annotations.push(Annotation::PatientLength(length));
                        self.tool_state = ToolState::Idle;
                    }
                    Err(error) => {
                        self.status_message = format!("Measurement rejected: {error}");
                        self.tool_state = ToolState::Idle;
                    }
                },
                _ => {
                    self.tool_state = ToolState::PatientLength1 { p1: point };
                    self.status_message = "Oblique length start selected".to_owned();
                }
            },
            ToolKind::Pan | ToolKind::Zoom | ToolKind::WindowLevel | ToolKind::Crosshair => {}
            unsupported => {
                self.status_message = format!(
                    "{} is not supported in the oblique view",
                    unsupported.label()
                );
            }
        }
    }
}
