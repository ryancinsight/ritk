//! Apply format-neutral presentation actions to RITK viewer state.
//!
//! The adapter is the RITK-owned side of the Métis host seam. Host coordinates
//! are mapped to image coordinates at the viewport boundary, then the existing
//! viewer transitions perform all domain work. No DICOM value or parser state
//! is part of this contract.

use super::state::SnapApp;
use crate::presentation::{
    ActionDispatchError, PointerButton, PointerGesture, PresentationEvent, ViewerAction,
    ViewportPoint,
};
use crate::tools::interaction::ImagePoint;
use crate::ui::{
    should_zoom_with_scroll, tool_kind_for_virtual_key, zoom_from_scroll, ViewTransform,
};
use thiserror::Error;

const PRIMARY_BUTTON: PointerButton = PointerButton::Left;
const VIRTUAL_KEY_PAGE_UP: u32 = 0x21;
const VIRTUAL_KEY_PAGE_DOWN: u32 = 0x22;
const VIRTUAL_KEY_END: u32 = 0x23;
const VIRTUAL_KEY_HOME: u32 = 0x24;
const VIRTUAL_KEY_ARROW_UP: u32 = 0x26;
const VIRTUAL_KEY_ARROW_DOWN: u32 = 0x28;

/// Geometry needed to map host client coordinates into one displayed slice.
///
/// The values are copied from the host's current image placement. Keeping the
/// mapping as a value lets native and browser hosts apply the same action
/// sequence without storing a GUI response or texture handle in viewer state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ViewerViewport {
    axis: usize,
    origin: egui::Pos2,
    texel_size: egui::Vec2,
    source_size: [usize; 2],
    transform: ViewTransform,
}

impl ViewerViewport {
    #[cfg(windows)]
    pub(crate) const fn axis(self) -> usize {
        self.axis
    }

    /// Construct a viewport mapping from validated image placement values.
    ///
    /// # Errors
    /// Returns an error when the axis, image dimensions or screen geometry is
    /// outside the finite positive range required by the inverse mapping.
    pub(crate) fn new(
        axis: usize,
        origin: egui::Pos2,
        texel_size: egui::Vec2,
        source_size: [usize; 2],
        transform: ViewTransform,
    ) -> Result<Self, ViewerViewportError> {
        if axis > 2 {
            return Err(ViewerViewportError::Axis { axis });
        }
        if source_size.contains(&0) {
            return Err(ViewerViewportError::EmptyImage { source_size });
        }
        if !origin.is_finite()
            || !texel_size.is_finite()
            || texel_size.x <= 0.0
            || texel_size.y <= 0.0
        {
            return Err(ViewerViewportError::InvalidScreenGeometry);
        }
        Ok(Self {
            axis,
            origin,
            texel_size,
            source_size,
            transform,
        })
    }

    fn screen_bounds(self) -> Option<[f64; 4]> {
        let [width, height] = self.transform.output_size(self.source_size);
        let min_x = f64::from(self.origin.x);
        let min_y = f64::from(self.origin.y);
        let extent_x = f64::from(self.texel_size.x) * width as f64;
        let extent_y = f64::from(self.texel_size.y) * height as f64;
        let max_x = min_x + extent_x;
        let max_y = min_y + extent_y;
        if !extent_x.is_finite()
            || !extent_y.is_finite()
            || !max_x.is_finite()
            || !max_y.is_finite()
        {
            return None;
        }
        Some([min_x, max_x, min_y, max_y])
    }

    fn map(self, point: ViewportPoint) -> Option<ImagePoint> {
        let x = point.x();
        let y = point.y();
        if !x.is_finite() || !y.is_finite() {
            return None;
        }
        let [min_x, max_x, min_y, max_y] = self.screen_bounds()?;
        if x < min_x || x > max_x || y < min_y || y > max_y {
            return None;
        }
        let output_size = self.transform.output_size(self.source_size);
        let output = [
            ((x - min_x) / f64::from(self.texel_size.x))
                .clamp(0.0, output_size[0] as f64 * 0.999_999),
            ((y - min_y) / f64::from(self.texel_size.y))
                .clamp(0.0, output_size[1] as f64 * 0.999_999),
        ];
        let [source_x, source_y] = self
            .transform
            .output_to_source_coordinates(output, self.source_size);
        if !source_x.is_finite()
            || !source_y.is_finite()
            || source_x < f64::from(f32::MIN)
            || source_x > f64::from(f32::MAX)
            || source_y < f64::from(f32::MIN)
            || source_y > f64::from(f32::MAX)
        {
            return None;
        }
        Some(ImagePoint::new(source_x as f32, source_y as f32))
    }
}

/// Error raised while validating a viewport action mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub(crate) enum ViewerViewportError {
    /// The viewport axis is outside the three orthogonal viewer axes.
    #[error("viewport axis {axis} is outside the supported range 0..=2")]
    Axis {
        /// Invalid axis value.
        axis: usize,
    },
    /// The source image has a zero dimension.
    #[error("viewport source dimensions {source_size:?} contain an empty axis")]
    EmptyImage {
        /// Invalid source dimensions.
        source_size: [usize; 2],
    },
    /// Origin or texel scale cannot represent a positive finite rectangle.
    #[error("viewport screen geometry must be finite with positive texel sizes")]
    InvalidScreenGeometry,
}

/// Result of applying one viewer action to the RITK application.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ViewerActionDisposition {
    /// Continue the host loop and optionally repaint the surface.
    Continue {
        /// Whether the caller should present a new frame.
        repaint: bool,
    },
    /// The host requested terminal surface teardown.
    Exit,
}

/// Failure applying a host action at the RITK boundary.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub(crate) enum ViewerActionError {
    /// A non-primary pointer button has no RITK-SNAP transition yet.
    #[error("pointer button {button:?} is not supported by the SnapApp adapter")]
    UnsupportedPointerButton {
        /// Button reported by the host.
        button: PointerButton,
    },
    /// A wheel delta cannot be represented by the viewer's `f32` zoom policy.
    #[error("wheel vertical delta {delta_y} exceeds the viewer precision range")]
    WheelDeltaOutOfRange {
        /// Signed vertical wheel displacement from the host.
        delta_y: f64,
    },
}

impl SnapApp {
    /// Reduce and apply a bounded batch of format-neutral host events.
    ///
    /// The dispatcher commits pointer state only after the whole batch is
    /// accepted. The action preflight below then rejects unsupported buttons
    /// before mutating [`SnapApp`], preserving the batch boundary at both
    /// presentation stages.
    ///
    /// # Errors
    /// Returns [`ViewerInputError`] when the host batch is malformed, an action
    /// uses an unsupported pointer button, or a zoom wheel delta exceeds the
    /// viewer's finite `f32` input range.
    pub(crate) fn apply_presentation_events(
        &mut self,
        events: &[PresentationEvent],
        viewport: Option<&ViewerViewport>,
    ) -> Result<ViewerActionDisposition, ViewerInputError> {
        for event in events {
            validate_presentation_event(event)?;
        }
        let actions = self.presentation_dispatcher.dispatch(events)?;
        for action in actions.iter() {
            validate_viewer_action(action)?;
        }
        let mut repaint = false;
        for action in actions.iter() {
            match self.apply_viewer_action(action, viewport)? {
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

    /// Cancel the active presentation gesture after a host loses its pointer.
    ///
    /// A native or browser host can terminate a drag without a final client
    /// coordinate. Clearing both reducer and viewer gesture state keeps the
    /// next press admissible and mirrors the focus-loss cancellation path.
    pub(crate) fn cancel_presentation_gesture(&mut self) {
        self.presentation_dispatcher.cancel_pointers();
        self.on_drag_end(None);
    }

    /// Apply one reduced host action to the RITK viewer state.
    ///
    /// Pointer actions require the viewport they target. Actions outside that
    /// rectangle are ignored just as a host widget ignores a pointer outside
    /// its hit region. Lifecycle actions do not alter loaded study state; the
    /// host loop owns surface teardown.
    ///
    /// # Errors
    /// Returns [`ViewerActionError::UnsupportedPointerButton`] when a pointer
    /// action uses a button without a corresponding viewer transition.
    pub(crate) fn apply_viewer_action(
        &mut self,
        action: &ViewerAction,
        viewport: Option<&ViewerViewport>,
    ) -> Result<ViewerActionDisposition, ViewerActionError> {
        match action {
            ViewerAction::CloseRequested | ViewerAction::Destroyed => {
                Ok(ViewerActionDisposition::Exit)
            }
            ViewerAction::FocusChanged { focused: false } => {
                self.on_drag_end(None);
                Ok(ViewerActionDisposition::Continue { repaint: true })
            }
            ViewerAction::FocusChanged { focused: true } => {
                Ok(ViewerActionDisposition::Continue { repaint: false })
            }
            ViewerAction::PointerMoved { position } => {
                let Some(viewport) = viewport else {
                    return Ok(ViewerActionDisposition::Continue { repaint: false });
                };
                let Some(image) = viewport.map(*position) else {
                    self.update_pointer_intensity(viewport.axis, None);
                    return Ok(ViewerActionDisposition::Continue { repaint: true });
                };
                self.update_pointer_intensity(viewport.axis, Some(image));
                Ok(ViewerActionDisposition::Continue { repaint: true })
            }
            ViewerAction::PointerPressed { button, position } => {
                ensure_primary(*button)?;
                let Some(viewport) = viewport else {
                    return Ok(ViewerActionDisposition::Continue { repaint: false });
                };
                let Some(image) = viewport.map(*position) else {
                    return Ok(ViewerActionDisposition::Continue { repaint: false });
                };
                if matches!(
                    self.active_tool,
                    crate::tools::kind::ToolKind::LabelPaint
                        | crate::tools::kind::ToolKind::LabelErase
                ) {
                    self.apply_label_at_pointer(viewport.axis, Some(image));
                }
                self.on_drag_start(Some(image));
                Ok(ViewerActionDisposition::Continue { repaint: true })
            }
            ViewerAction::PointerDragged {
                button, current, ..
            } => {
                ensure_primary(*button)?;
                let Some(viewport) = viewport else {
                    return Ok(ViewerActionDisposition::Continue { repaint: false });
                };
                let Some(image) = viewport.map(*current) else {
                    return Ok(ViewerActionDisposition::Continue { repaint: false });
                };
                if matches!(
                    self.active_tool,
                    crate::tools::kind::ToolKind::LabelPaint
                        | crate::tools::kind::ToolKind::LabelErase
                ) {
                    self.apply_label_at_pointer(viewport.axis, Some(image));
                }
                self.on_drag(Some(image));
                Ok(ViewerActionDisposition::Continue { repaint: true })
            }
            ViewerAction::PointerReleased {
                button,
                position,
                gesture,
            } => {
                ensure_primary(*button)?;
                let mapped = viewport.and_then(|viewport| viewport.map(*position));
                if *gesture == PointerGesture::Click {
                    if let Some(viewport) = viewport {
                        if let Some(image) = mapped {
                            self.update_linked_cursor_from_pointer(viewport.axis, Some(image));
                        }
                    }
                    self.on_click(mapped);
                    self.on_click_end();
                } else {
                    self.on_drag_end(mapped);
                }
                Ok(ViewerActionDisposition::Continue { repaint: true })
            }
            ViewerAction::PointerCancelled { button, .. } => {
                ensure_primary(*button)?;
                self.on_drag_end(None);
                Ok(ViewerActionDisposition::Continue { repaint: true })
            }
            ViewerAction::PointerWheel {
                position,
                delta,
                modifiers,
            } => {
                let Some(viewport) = viewport else {
                    return Ok(ViewerActionDisposition::Continue { repaint: false });
                };
                if viewport.map(*position).is_none() {
                    return Ok(ViewerActionDisposition::Continue { repaint: false });
                }
                if delta.x() == 0.0 && delta.y() == 0.0 {
                    return Ok(ViewerActionDisposition::Continue { repaint: false });
                }
                if delta.y() == 0.0 {
                    return Ok(ViewerActionDisposition::Continue { repaint: false });
                }
                if should_zoom_with_scroll(modifiers.ctrl() || modifiers.meta()) {
                    let scroll_y = viewer_scroll_value(delta.y())?;
                    self.zoom = zoom_from_scroll(self.zoom, scroll_y);
                    self.status_message = format!("Zoom: {:.0}%", self.zoom * 100.0);
                    Ok(ViewerActionDisposition::Continue { repaint: true })
                } else {
                    let step = if delta.y() > 0.0 { -1_i32 } else { 1 };
                    self.step_slice_for_axis(viewport.axis, step);
                    Ok(ViewerActionDisposition::Continue { repaint: true })
                }
            }
            ViewerAction::KeyPressed { virtual_key, .. } => {
                Ok(self.apply_virtual_key(*virtual_key))
            }
            ViewerAction::KeyReleased { .. }
            | ViewerAction::TextInput { .. }
            | ViewerAction::TextComposition { .. }
            | ViewerAction::Resized { .. }
            | ViewerAction::DpiChanged { .. } => {
                Ok(ViewerActionDisposition::Continue { repaint: false })
            }
        }
    }

    fn apply_virtual_key(&mut self, virtual_key: u32) -> ViewerActionDisposition {
        if let Some(tool) = tool_kind_for_virtual_key(virtual_key) {
            self.active_tool = tool;
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

/// Failure while reducing or applying one host event batch.
#[derive(Debug, Error)]
pub(crate) enum ViewerInputError {
    /// The presentation event sequence violated its bounded contract.
    #[error("presentation event dispatch failed: {0}")]
    Dispatch(#[from] ActionDispatchError),
    /// The reduced action had no corresponding RITK transition.
    #[error("viewer action application failed: {0}")]
    Action(#[from] ViewerActionError),
}

fn validate_presentation_event(event: &PresentationEvent) -> Result<(), ViewerActionError> {
    if let Some(button) = event_button(event) {
        ensure_primary(button)?;
    }
    if let PresentationEvent::PointerWheel {
        delta_y, modifiers, ..
    } = event
    {
        if delta_y.is_finite() && should_zoom_with_scroll(modifiers.ctrl() || modifiers.meta()) {
            viewer_scroll_value(*delta_y)?;
        }
    }
    Ok(())
}

fn validate_viewer_action(action: &ViewerAction) -> Result<(), ViewerActionError> {
    match action {
        ViewerAction::PointerPressed { button, .. }
        | ViewerAction::PointerDragged { button, .. }
        | ViewerAction::PointerReleased { button, .. }
        | ViewerAction::PointerCancelled { button, .. } => ensure_primary(*button),
        ViewerAction::PointerWheel {
            delta, modifiers, ..
        } if should_zoom_with_scroll(modifiers.ctrl() || modifiers.meta()) => {
            viewer_scroll_value(delta.y()).map(|_| ())
        }
        _ => Ok(()),
    }
}

fn event_button(event: &PresentationEvent) -> Option<PointerButton> {
    match event {
        PresentationEvent::PointerDown { button, .. }
        | PresentationEvent::PointerUp { button, .. } => Some(*button),
        _ => None,
    }
}

fn ensure_primary(button: PointerButton) -> Result<(), ViewerActionError> {
    if button == PRIMARY_BUTTON {
        Ok(())
    } else {
        Err(ViewerActionError::UnsupportedPointerButton { button })
    }
}

fn viewer_scroll_value(value: f64) -> Result<f32, ViewerActionError> {
    if !value.is_finite() || value < f64::from(f32::MIN) || value > f64::from(f32::MAX) {
        return Err(ViewerActionError::WheelDeltaOutOfRange { delta_y: value });
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the existing RITK zoom policy is f32; the finite host delta is checked against that contract before conversion"
    )]
    let value = value as f32;
    Ok(value)
}
