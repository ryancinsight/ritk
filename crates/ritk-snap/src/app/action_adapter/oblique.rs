use super::*;
use crate::app::ObliqueViewport;
use crate::geometry::PatientPointMm;
use crate::presentation::{PointerGesture, PresentationEvent, ViewerAction};
use crate::render::{ReslicePlane, ResliceSample};

impl SnapApp {
    /// Apply viewer actions against a rendered physical oblique plane.
    ///
    /// Axis-bound view state is deliberately left untouched. The native panel
    /// owns pan, zoom, and plane navigation; this adapter handles the shared
    /// cursor, sampled intensity, and patient-space length tool.
    ///
    /// # Errors
    /// Returns [`ViewerInputError`] when the event batch is malformed, uses an
    /// unsupported pointer button, the displayed plane no longer matches the
    /// loaded volume, or a sampled patient point is invalid.
    pub(crate) fn apply_oblique_presentation_events(
        &mut self,
        events: &[PresentationEvent],
        viewport: &ObliqueViewport,
        plane: &ReslicePlane,
    ) -> Result<ViewerActionDisposition, ViewerInputError> {
        for event in events {
            if let Some(button) = event_button(event) {
                ensure_primary(button)?;
            }
        }
        viewport.validate_dimensions(plane.dimensions())?;
        if let Some(volume) = self.loaded.as_ref() {
            plane.validate_source(volume)?;
        }

        let mut next_dispatcher = self.presentation_dispatcher.clone();
        let actions = next_dispatcher.dispatch(events)?;
        for action in actions.iter() {
            validate_oblique_action(action)?;
        }

        self.presentation_dispatcher = next_dispatcher;
        let mut repaint = false;
        for action in actions.iter() {
            match self.apply_oblique_viewer_action(action, viewport, plane)? {
                ViewerActionDisposition::Continue { repaint: needed } => {
                    repaint |= needed;
                }
                ViewerActionDisposition::Exit => {
                    return Ok(ViewerActionDisposition::Exit);
                }
            }
        }
        Ok(ViewerActionDisposition::Continue { repaint })
    }

    fn apply_oblique_viewer_action(
        &mut self,
        action: &ViewerAction,
        viewport: &ObliqueViewport,
        plane: &ReslicePlane,
    ) -> Result<ViewerActionDisposition, ViewerInputError> {
        match action {
            ViewerAction::CloseRequested | ViewerAction::Destroyed => {
                Ok(ViewerActionDisposition::Exit)
            }
            ViewerAction::FocusChanged { focused: false } => {
                self.on_drag_end(None);
                Ok(ViewerActionDisposition::Continue { repaint: true })
            }
            ViewerAction::PointerMoved { position } => {
                let sample = self.sample_oblique_pointer(viewport, plane, *position)?;
                self.update_pointer_from_oblique_sample(sample);
                Ok(ViewerActionDisposition::Continue { repaint: true })
            }
            ViewerAction::PointerReleased {
                button,
                position,
                gesture,
            } => {
                ensure_primary(*button)?;
                if *gesture == PointerGesture::Click {
                    let sample = self.sample_oblique_pointer(viewport, plane, *position)?;
                    self.update_pointer_from_oblique_sample(sample);
                    if let Some(sample) = sample {
                        let patient = PatientPointMm::try_new(sample.patient())?;
                        self.update_linked_cursor_from_voxel(sample.nearest_voxel());
                        self.on_oblique_click(patient);
                    }
                    self.on_click_end();
                } else {
                    self.on_drag_end(viewport.map(*position).map(oblique_image_point));
                }
                Ok(ViewerActionDisposition::Continue { repaint: true })
            }
            ViewerAction::PointerPressed { button, position } => {
                ensure_primary(*button)?;
                if self.active_tool == crate::tools::kind::ToolKind::WindowLevel {
                    let position = viewport.map(*position).map(oblique_image_point);
                    self.on_drag_start(position);
                }
                Ok(ViewerActionDisposition::Continue { repaint: false })
            }
            ViewerAction::PointerDragged {
                button, current, ..
            } => {
                ensure_primary(*button)?;
                if matches!(
                    self.tool_state,
                    crate::tools::interaction::ToolState::WindowLevelDrag { .. }
                ) {
                    let position = viewport.map(*current).map(oblique_image_point);
                    self.on_drag(position);
                }
                Ok(ViewerActionDisposition::Continue { repaint: false })
            }
            ViewerAction::PointerCancelled { button, .. } => {
                ensure_primary(*button)?;
                self.on_drag_end(None);
                Ok(ViewerActionDisposition::Continue { repaint: true })
            }
            ViewerAction::KeyPressed {
                virtual_key,
                repeated,
            } if matches!(
                *virtual_key,
                0x25 | VIRTUAL_KEY_ARROW_UP | 0x27 | VIRTUAL_KEY_ARROW_DOWN
            ) =>
            {
                let _ = repeated;
                Ok(ViewerActionDisposition::Continue { repaint: false })
            }
            ViewerAction::KeyPressed {
                virtual_key,
                repeated,
            } => Ok(self.apply_virtual_key(*virtual_key, *repeated)),
            ViewerAction::KeyReleased { .. }
            | ViewerAction::FocusChanged { focused: true }
            | ViewerAction::PointerWheel { .. }
            | ViewerAction::TextInput { .. }
            | ViewerAction::TextComposition { .. }
            | ViewerAction::Resized { .. }
            | ViewerAction::DpiChanged { .. } => {
                Ok(ViewerActionDisposition::Continue { repaint: false })
            }
        }
    }

    fn sample_oblique_pointer(
        &self,
        viewport: &ObliqueViewport,
        plane: &ReslicePlane,
        position: crate::presentation::ViewportPoint,
    ) -> Result<Option<ResliceSample>, crate::render::ResliceError> {
        let Some(pixel) = viewport.map(position) else {
            return Ok(None);
        };
        let Some(volume) = self.loaded.as_ref() else {
            return Ok(None);
        };
        plane.sample_pixel(volume, pixel).map(Some)
    }

    fn update_pointer_from_oblique_sample(&mut self, sample: Option<ResliceSample>) {
        let Some(sample) = sample else {
            self.pointer_intensity = 0.0;
            self.pointer_suv = None;
            return;
        };
        self.pointer_intensity = sample.value();
        self.pointer_suv = self
            .loaded
            .as_ref()
            .and_then(|volume| Self::compute_suv_from_volume(volume, f64::from(sample.value())));
    }
}

fn oblique_image_point(pixel: [f64; 2]) -> crate::tools::interaction::ImagePoint {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the oblique viewport clips coordinates to its u32 presentation frame before the viewer's f32 interaction contract"
    )]
    let [column, row] = pixel.map(|coordinate| coordinate as f32);
    crate::tools::interaction::ImagePoint::new(column, row)
}

fn validate_oblique_action(action: &ViewerAction) -> Result<(), ViewerActionError> {
    match action {
        ViewerAction::PointerPressed { button, .. }
        | ViewerAction::PointerDragged { button, .. }
        | ViewerAction::PointerReleased { button, .. }
        | ViewerAction::PointerCancelled { button, .. } => ensure_primary(*button),
        _ => Ok(()),
    }
}
