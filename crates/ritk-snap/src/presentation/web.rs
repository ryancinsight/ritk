//! Browser canvas adapter for RITK's format-neutral presentation frame.

use super::{PointerButton, PresentationEvent, PresentationFrame, PresentationModifiers};
use metis_web::{
    CanvasEvent, CanvasEventError, CanvasFrame, CanvasPointerEvent, CanvasPointerPhase,
    CanvasPointerType, CanvasSurface, CanvasWheelEvent, CanvasWheelUnit,
};
use std::io;
use thiserror::Error;

const WHEEL_LINE_PIXELS: f64 = 16.0;
const WHEEL_PAGE_PIXELS: f64 = 640.0;

/// Presents RITK-owned pixels through a Metis browser canvas.
pub struct WebCanvasPresenter {
    surface: CanvasSurface,
}

impl WebCanvasPresenter {
    /// Resolves a canvas in the current browser document.
    ///
    /// # Errors
    /// Returns a typed I/O error when the browser document, canvas element, or
    /// two-dimensional rendering context is unavailable.
    pub fn from_canvas_id(id: &str) -> io::Result<Self> {
        Ok(Self {
            surface: CanvasSurface::from_current_document(id)?,
        })
    }

    /// Resolves a canvas and retains bounded browser input listeners.
    ///
    /// # Errors
    /// Returns a typed I/O error when the browser document, canvas element,
    /// rendering context, or one of the listener registrations is unavailable.
    pub fn from_canvas_id_with_input(id: &str) -> io::Result<Self> {
        Ok(Self {
            surface: CanvasSurface::from_current_document_with_input(id)?,
        })
    }

    /// Returns the identifier of the resolved browser canvas.
    #[must_use]
    pub fn canvas_id(&self) -> String {
        self.surface.id()
    }

    /// Presents one RITK display frame without copying it in the Rust host.
    ///
    /// RITK retains ownership of the frame and its display semantics; Metis
    /// receives only the borrowed RGBA view at this boundary.
    ///
    /// # Errors
    /// Returns a typed I/O error when the frame violates the provider bounds
    /// or the browser rejects the upload.
    pub fn present(&self, frame: &PresentationFrame) -> io::Result<()> {
        self.surface.present(frame)
    }

    /// Takes the bounded browser input batch as RITK presentation events.
    ///
    /// Target-local CSS-pixel coordinates are preserved as the RITK client
    /// coordinates. Wheel line and page units are normalized at this host
    /// boundary so the action reducer receives one explicit displacement unit.
    /// Non-primary touch pointers are ignored because the current RITK
    /// dispatcher has one button-indexed gesture state; the browser surface
    /// still releases their capture when its event listener observes them.
    ///
    /// # Errors
    /// Returns a typed error when the Metis queue failed, a browser pointer
    /// button or wheel unit is outside the RITK contract, or a wheel delta is
    /// non-finite.
    pub fn take_events(&self) -> Result<Box<[PresentationEvent]>, WebCanvasInputError> {
        let events = self.surface.take_events()?;
        let mut translated = Vec::new();
        translated.try_reserve_exact(events.len()).map_err(|_| {
            WebCanvasInputError::Allocation {
                requested: events.len(),
            }
        })?;
        for event in events.iter().copied() {
            if let Some(event) = translate_event(event)? {
                translated.push(event);
            }
        }
        Ok(translated.into_boxed_slice())
    }
}

/// Failure translating one Metis canvas event into the RITK host contract.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum WebCanvasInputError {
    /// The bounded Metis event queue reported a terminal browser failure.
    #[error("Metis canvas input failed: {0}")]
    Canvas(#[from] CanvasEventError),
    /// A bounded translation allocation could not be reserved.
    #[error("unable to reserve {requested} browser presentation events")]
    Allocation {
        /// Number of output events requested.
        requested: usize,
    },
    /// A pointer button value is not representable by RITK.
    #[error("browser pointer button {button} is outside the supported range")]
    PointerButton {
        /// Browser button value.
        button: i16,
    },
    /// A browser wheel unit is not part of the normalized contract.
    #[error("browser wheel unit is not supported by the RITK host")]
    WheelUnit,
    /// A browser wheel delta is not finite.
    #[error("browser wheel delta is not finite")]
    NonFiniteWheel,
}

fn translate_event(event: CanvasEvent) -> Result<Option<PresentationEvent>, WebCanvasInputError> {
    match event {
        CanvasEvent::Pointer(pointer) => translate_pointer(pointer),
        CanvasEvent::Wheel(wheel) => translate_wheel(wheel).map(Some),
    }
}

fn translate_pointer(
    pointer: CanvasPointerEvent,
) -> Result<Option<PresentationEvent>, WebCanvasInputError> {
    if !pointer.is_primary() && matches!(pointer.pointer_type(), CanvasPointerType::Touch) {
        return Ok(None);
    }
    let x = f64::from(pointer.x());
    let y = f64::from(pointer.y());
    let button = match pointer.phase() {
        CanvasPointerPhase::Move => PointerButton::Left,
        CanvasPointerPhase::Cancel => button_or_left(pointer.button())?,
        CanvasPointerPhase::Down | CanvasPointerPhase::Up => button(pointer.button())?,
    };
    Ok(Some(match pointer.phase() {
        CanvasPointerPhase::Down => PresentationEvent::PointerDown { x, y, button },
        CanvasPointerPhase::Move => PresentationEvent::PointerMove { x, y },
        CanvasPointerPhase::Up => PresentationEvent::PointerUp { x, y, button },
        CanvasPointerPhase::Cancel => PresentationEvent::PointerCancel { x, y, button },
    }))
}

fn translate_wheel(wheel: CanvasWheelEvent) -> Result<PresentationEvent, WebCanvasInputError> {
    let scale = match wheel.unit() {
        CanvasWheelUnit::Pixel => 1.0,
        CanvasWheelUnit::Line => WHEEL_LINE_PIXELS,
        CanvasWheelUnit::Page => WHEEL_PAGE_PIXELS,
        _ => return Err(WebCanvasInputError::WheelUnit),
    };
    let delta_x = wheel.delta_x() * scale;
    let delta_y = wheel.delta_y() * scale;
    if !delta_x.is_finite() || !delta_y.is_finite() || !wheel.delta_z().is_finite() {
        return Err(WebCanvasInputError::NonFiniteWheel);
    }
    Ok(PresentationEvent::PointerWheel {
        x: f64::from(wheel.x()),
        y: f64::from(wheel.y()),
        delta_x,
        delta_y,
        modifiers: modifiers(wheel.modifiers()),
    })
}

fn button(value: i16) -> Result<PointerButton, WebCanvasInputError> {
    match value {
        0 => Ok(PointerButton::Left),
        1 => Ok(PointerButton::Middle),
        2 => Ok(PointerButton::Right),
        3 => Ok(PointerButton::X1),
        4 => Ok(PointerButton::X2),
        _ => Err(WebCanvasInputError::PointerButton { button: value }),
    }
}

fn button_or_left(value: i16) -> Result<PointerButton, WebCanvasInputError> {
    if value < 0 {
        Ok(PointerButton::Left)
    } else {
        button(value)
    }
}

fn modifiers(value: metis_web::CanvasModifiers) -> PresentationModifiers {
    PresentationModifiers::new(value.ctrl(), value.shift(), value.alt(), value.meta())
}

impl CanvasFrame for PresentationFrame {
    fn width(&self) -> u32 {
        PresentationFrame::width(self)
    }

    fn height(&self) -> u32 {
        PresentationFrame::height(self)
    }

    fn rgba(&self) -> &[u8] {
        PresentationFrame::rgba(self)
    }
}
