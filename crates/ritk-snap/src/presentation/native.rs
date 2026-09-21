//! Windows host adapter for one RITK presentation frame.

use super::{
    AccessibilityAction, AccessibilityActionRequest, CompositionPhase, PointerButton,
    PresentationEvent, PresentationFrame, PresentationModifiers, MAX_ACCESSIBILITY_VALUE_BYTES,
    MAX_COMPOSITION_UNITS, MAX_PRESENTATION_EVENTS,
};
use anyhow::{anyhow, bail, Result};
use metis_platform::native::{
    run_native_application, NativeApplication, NativeFlow, WindowConfig, WindowEvent,
    WindowVisibility, MAX_COMPOSITION_UNITS as PROVIDER_MAX_COMPOSITION_UNITS,
    MAX_WINDOW_EVENTS as PROVIDER_MAX_WINDOW_EVENTS,
};
use metis_platform::{Color, Framebuffer};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

const _: () = assert!(
    MAX_PRESENTATION_EVENTS == PROVIDER_MAX_WINDOW_EVENTS,
    "RITK presentation batch bound must match the Métis native queue bound"
);
const _: () = assert!(
    MAX_COMPOSITION_UNITS == PROVIDER_MAX_COMPOSITION_UNITS,
    "RITK composition bound must match the Métis native input bound"
);

/// Observable result of one native frame presentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeFrameOutcome {
    width: u32,
    height: u32,
    pixel_count: usize,
    frame_requests: usize,
    event_batches: usize,
    translated_events: usize,
}

impl NativeFrameOutcome {
    /// Width of the frame requested by the host.
    #[must_use]
    pub const fn width(self) -> u32 {
        self.width
    }

    /// Height of the frame requested by the host.
    #[must_use]
    pub const fn height(self) -> u32 {
        self.height
    }

    /// Number of RGBA pixels transferred to the host framebuffer.
    #[must_use]
    pub const fn pixel_count(self) -> usize {
        self.pixel_count
    }

    /// Number of framebuffer requests made by the host loop.
    #[must_use]
    pub const fn frame_requests(self) -> usize {
        self.frame_requests
    }

    /// Number of bounded event batches processed before exit.
    #[must_use]
    pub const fn event_batches(self) -> usize {
        self.event_batches
    }

    /// Number of native events translated at the RITK boundary.
    #[must_use]
    pub const fn translated_events(self) -> usize {
        self.translated_events
    }
}

/// Translates one bounded Métis/Moirai native event batch into viewer events.
///
/// The translation preserves all provider values while removing the host
/// provider types from the RITK-facing contract. It does not interpret paths,
/// DICOM metadata, volume state or application authority.
///
/// # Errors
/// Returns an error when the provider batch exceeds its declared event bound or
/// when the translated event storage cannot be reserved.
pub fn translate_native_events(events: &[WindowEvent]) -> Result<Box<[PresentationEvent]>> {
    if events.len() > MAX_PRESENTATION_EVENTS {
        bail!(
            "native event batch length {} exceeds host limit {}",
            events.len(),
            MAX_PRESENTATION_EVENTS
        );
    }
    let mut translated = Vec::new();
    translated
        .try_reserve_exact(events.len())
        .map_err(|_| anyhow!("unable to reserve translated native events"))?;
    for event in events {
        translated.push(match event {
            WindowEvent::CloseRequested => PresentationEvent::CloseRequested,
            WindowEvent::Destroyed => PresentationEvent::Destroyed,
            WindowEvent::FocusGained => PresentationEvent::FocusGained,
            WindowEvent::FocusLost => PresentationEvent::FocusLost,
            WindowEvent::AccessibilityAction { request } => {
                PresentationEvent::AccessibilityAction {
                    request: AccessibilityActionRequest {
                        target_node: request.target_node,
                        action: translate_accessibility_action(request.action)?,
                        value: translate_accessibility_value(request.value.as_deref())?,
                        delta: request.delta,
                    },
                }
            }
            WindowEvent::PointerMove { x, y } => PresentationEvent::PointerMove {
                x: f64::from(*x),
                y: f64::from(*y),
            },
            WindowEvent::PointerDown { x, y, button } => PresentationEvent::PointerDown {
                x: f64::from(*x),
                y: f64::from(*y),
                button: translate_pointer_button(*button),
            },
            WindowEvent::PointerUp { x, y, button } => PresentationEvent::PointerUp {
                x: f64::from(*x),
                y: f64::from(*y),
                button: translate_pointer_button(*button),
            },
            WindowEvent::PointerWheel {
                x,
                y,
                delta_x,
                delta_y,
                modifiers,
            } => PresentationEvent::PointerWheel {
                x: f64::from(*x),
                y: f64::from(*y),
                delta_x: f64::from(*delta_x),
                delta_y: f64::from(*delta_y),
                modifiers: translate_modifiers(*modifiers),
            },
            WindowEvent::KeyDown {
                virtual_key,
                repeated,
                modifiers,
            } => PresentationEvent::KeyDown {
                virtual_key: *virtual_key,
                repeated: *repeated,
                modifiers: translate_modifiers(*modifiers),
            },
            WindowEvent::KeyUp {
                virtual_key,
                modifiers,
            } => PresentationEvent::KeyUp {
                virtual_key: *virtual_key,
                modifiers: translate_modifiers(*modifiers),
            },
            WindowEvent::TextInput { character } => PresentationEvent::TextInput {
                character: *character,
            },
            WindowEvent::TextComposition { phase, text } => PresentationEvent::TextComposition {
                phase: translate_composition_phase(*phase),
                text: translate_composition_text(text)?,
            },
            WindowEvent::Resized { width, height } => PresentationEvent::Resized {
                width: *width,
                height: *height,
            },
            WindowEvent::DpiChanged { dpi } => PresentationEvent::DpiChanged { dpi: *dpi },
        });
    }
    Ok(translated.into_boxed_slice())
}

fn translate_pointer_button(button: metis_platform::native::MouseButton) -> PointerButton {
    match button {
        metis_platform::native::MouseButton::Left => PointerButton::Left,
        metis_platform::native::MouseButton::Right => PointerButton::Right,
        metis_platform::native::MouseButton::Middle => PointerButton::Middle,
        metis_platform::native::MouseButton::X1 => PointerButton::X1,
        metis_platform::native::MouseButton::X2 => PointerButton::X2,
    }
}

fn translate_modifiers(modifiers: metis_platform::native::ModifierState) -> PresentationModifiers {
    PresentationModifiers::new(
        modifiers.ctrl(),
        modifiers.shift(),
        modifiers.alt(),
        modifiers.meta(),
    )
}

fn translate_composition_phase(
    phase: metis_platform::native::CompositionPhase,
) -> CompositionPhase {
    match phase {
        metis_platform::native::CompositionPhase::Started => CompositionPhase::Started,
        metis_platform::native::CompositionPhase::Updated => CompositionPhase::Updated,
        metis_platform::native::CompositionPhase::Committed => CompositionPhase::Committed,
        metis_platform::native::CompositionPhase::Canceled => CompositionPhase::Canceled,
    }
}

fn translate_accessibility_action(
    action: metis_platform::native::AccessibilityAction,
) -> Result<AccessibilityAction> {
    match action {
        metis_platform::native::AccessibilityAction::Activate => Ok(AccessibilityAction::Activate),
        metis_platform::native::AccessibilityAction::Focus => Ok(AccessibilityAction::Focus),
        metis_platform::native::AccessibilityAction::SetValue => Ok(AccessibilityAction::SetValue),
        metis_platform::native::AccessibilityAction::Toggle => Ok(AccessibilityAction::Toggle),
        metis_platform::native::AccessibilityAction::AdjustValue => {
            Ok(AccessibilityAction::AdjustValue)
        }
        metis_platform::native::AccessibilityAction::Open => Ok(AccessibilityAction::Open),
        action => bail!("unsupported native accessibility action {action:?}"),
    }
}

fn translate_accessibility_value(value: Option<&str>) -> Result<Option<Box<str>>> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.len() > MAX_ACCESSIBILITY_VALUE_BYTES {
        bail!(
            "native accessibility action value length {} exceeds host limit {} bytes",
            value.len(),
            MAX_ACCESSIBILITY_VALUE_BYTES
        );
    }
    let mut owned = String::new();
    owned
        .try_reserve_exact(value.len())
        .map_err(|_| anyhow!("unable to reserve translated accessibility action value"))?;
    owned.push_str(value);
    Ok(Some(owned.into_boxed_str()))
}

fn translate_composition_text(text: &str) -> Result<Box<str>> {
    let utf16_units = text.encode_utf16().count();
    if utf16_units > MAX_COMPOSITION_UNITS {
        bail!(
            "native composition length {} exceeds host limit {} UTF-16 units",
            utf16_units,
            MAX_COMPOSITION_UNITS
        );
    }
    let mut owned = String::new();
    owned
        .try_reserve_exact(text.len())
        .map_err(|_| anyhow!("unable to reserve translated composition text"))?;
    owned.push_str(text);
    Ok(owned.into_boxed_str())
}

/// Presents one RITK frame through the bounded native host and exits.
///
/// The host receives only pixels. DICOM loading and display mapping have
/// already completed in RITK before this function is called.
///
/// # Errors
/// Returns a frame conversion, native surface, or host transition error.
pub fn run_native_frame(frame: PresentationFrame, title: &str) -> Result<NativeFrameOutcome> {
    let framebuffer = to_framebuffer(&frame)?;
    let config = WindowConfig::with_visibility(
        title,
        frame.width(),
        frame.height(),
        WindowVisibility::Hidden,
    )?;
    let observation = Arc::new(HostObservation::default());
    run_native_application(
        &config,
        SingleFrameApplication {
            framebuffer,
            observation: Arc::clone(&observation),
        },
        Duration::ZERO,
    )
    .map_err(|error| anyhow!("RITK presentation host failed: {error}"))?;
    let frame_requests = observation.frame_requests.load(Ordering::Relaxed);
    let event_batches = observation.event_batches.load(Ordering::Relaxed);
    let translated_events = observation.translated_events.load(Ordering::Relaxed);
    if observation.destroyed.load(Ordering::Relaxed) {
        bail!("native presentation surface was destroyed before close");
    }
    if frame_requests != 1 || event_batches != 1 {
        bail!(
            "native host completed an unexpected one-frame trace: {} frame requests, {} event batches",
            frame_requests,
            event_batches
        );
    }
    if translated_events == 0 {
        bail!("native host completed without a translated event");
    }
    Ok(NativeFrameOutcome {
        width: frame.width(),
        height: frame.height(),
        pixel_count: frame.rgba().len() / 4,
        frame_requests,
        event_batches,
        translated_events,
    })
}

fn to_framebuffer(frame: &PresentationFrame) -> Result<Framebuffer> {
    let mut framebuffer = Framebuffer::new(frame.width(), frame.height())
        .map_err(|error| anyhow!("allocate native presentation framebuffer: {error}"))?;
    let width = usize::try_from(frame.width()).map_err(|_| anyhow!("frame width exceeds usize"))?;
    for (index, pixel) in frame.rgba().chunks_exact(4).enumerate() {
        let x = i32::try_from(index % width).map_err(|_| anyhow!("frame x exceeds i32"))?;
        let y = i32::try_from(index / width).map_err(|_| anyhow!("frame y exceeds i32"))?;
        framebuffer.set_pixel(x, y, Color::rgba(pixel[0], pixel[1], pixel[2], pixel[3]));
    }
    if !frame.rgba().chunks_exact(4).remainder().is_empty() {
        return Err(anyhow!("presentation frame has an incomplete RGBA pixel"));
    }
    Ok(framebuffer)
}

struct SingleFrameApplication {
    framebuffer: Framebuffer,
    observation: Arc<HostObservation>,
}

#[derive(Default)]
struct HostObservation {
    frame_requests: AtomicUsize,
    event_batches: AtomicUsize,
    translated_events: AtomicUsize,
    destroyed: AtomicBool,
}

impl NativeApplication for SingleFrameApplication {
    type Error = std::io::Error;

    fn framebuffer(&self) -> &Framebuffer {
        self.observation
            .frame_requests
            .fetch_add(1, Ordering::Relaxed);
        &self.framebuffer
    }

    fn handle_events(
        &mut self,
        events: &[WindowEvent],
    ) -> std::result::Result<NativeFlow, Self::Error> {
        let translated = translate_native_events(events).map_err(|error| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, error.to_string())
        })?;
        self.observation
            .event_batches
            .fetch_add(1, Ordering::Relaxed);
        self.observation
            .translated_events
            .fetch_add(translated.len(), Ordering::Relaxed);
        if translated
            .iter()
            .any(|event| matches!(event, PresentationEvent::Destroyed))
        {
            self.observation.destroyed.store(true, Ordering::Relaxed);
        }
        Ok(NativeFlow::Exit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_events_preserve_provider_values() {
        let events = [
            WindowEvent::CloseRequested,
            WindowEvent::Destroyed,
            WindowEvent::FocusGained,
            WindowEvent::FocusLost,
            WindowEvent::AccessibilityAction {
                request: metis_platform::native::AccessibilityActionRequest {
                    target_node: 41,
                    action: metis_platform::native::AccessibilityAction::SetValue,
                    value: Some("patient".to_owned()),
                    delta: None,
                },
            },
            WindowEvent::PointerMove { x: -4, y: 8 },
            WindowEvent::PointerDown {
                x: 1,
                y: 2,
                button: metis_platform::native::MouseButton::X1,
            },
            WindowEvent::PointerUp {
                x: 3,
                y: 4,
                button: metis_platform::native::MouseButton::Right,
            },
            WindowEvent::PointerWheel {
                x: -5,
                y: 6,
                delta_x: -240,
                delta_y: 120,
                modifiers: metis_platform::native::ModifierState::NONE,
            },
            WindowEvent::KeyDown {
                virtual_key: 0x41,
                repeated: true,
                modifiers: metis_platform::native::ModifierState::NONE,
            },
            WindowEvent::KeyUp {
                virtual_key: 0x41,
                modifiers: metis_platform::native::ModifierState::NONE,
            },
            WindowEvent::TextInput { character: '中' },
            WindowEvent::TextComposition {
                phase: metis_platform::native::CompositionPhase::Updated,
                text: "A😀".to_owned(),
            },
            WindowEvent::Resized {
                width: 800,
                height: 600,
            },
            WindowEvent::DpiChanged { dpi: 144 },
        ];
        let translated = translate_native_events(&events).expect("translated events");
        assert_eq!(translated.len(), events.len());
        assert_eq!(
            translated.as_ref(),
            &[
                PresentationEvent::CloseRequested,
                PresentationEvent::Destroyed,
                PresentationEvent::FocusGained,
                PresentationEvent::FocusLost,
                PresentationEvent::AccessibilityAction {
                    request: AccessibilityActionRequest {
                        target_node: 41,
                        action: AccessibilityAction::SetValue,
                        value: Some("patient".into()),
                        delta: None,
                    },
                },
                PresentationEvent::PointerMove { x: -4.0, y: 8.0 },
                PresentationEvent::PointerDown {
                    x: 1.0,
                    y: 2.0,
                    button: PointerButton::X1,
                },
                PresentationEvent::PointerUp {
                    x: 3.0,
                    y: 4.0,
                    button: PointerButton::Right,
                },
                PresentationEvent::PointerWheel {
                    x: -5.0,
                    y: 6.0,
                    delta_x: -240.0,
                    delta_y: 120.0,
                    modifiers: PresentationModifiers::NONE,
                },
                PresentationEvent::KeyDown {
                    virtual_key: 0x41,
                    repeated: true,
                    modifiers: PresentationModifiers::NONE,
                },
                PresentationEvent::KeyUp {
                    virtual_key: 0x41,
                    modifiers: PresentationModifiers::NONE,
                },
                PresentationEvent::TextInput { character: '中' },
                PresentationEvent::TextComposition {
                    phase: CompositionPhase::Updated,
                    text: "A😀".into(),
                },
                PresentationEvent::Resized {
                    width: 800,
                    height: 600,
                },
                PresentationEvent::DpiChanged { dpi: 144 },
            ]
        );
    }

    #[test]
    fn native_coordinates_preserve_values_above_f32_integer_precision() {
        let events = [WindowEvent::PointerMove {
            x: 16_777_217,
            y: -16_777_217,
        }];
        let translated = translate_native_events(&events).expect("translated events");
        assert_eq!(
            translated.as_ref(),
            &[PresentationEvent::PointerMove {
                x: 16_777_217.0,
                y: -16_777_217.0,
            }]
        );
    }

    #[test]
    fn native_events_reject_oversized_batch() {
        let events = vec![WindowEvent::FocusGained; MAX_PRESENTATION_EVENTS + 1];
        let error = translate_native_events(&events).expect_err("oversized batch");
        assert!(error.to_string().contains("exceeds host limit"));
    }

    #[test]
    fn native_accessibility_value_bound_is_enforced() {
        let events = [WindowEvent::AccessibilityAction {
            request: metis_platform::native::AccessibilityActionRequest {
                target_node: 41,
                action: metis_platform::native::AccessibilityAction::SetValue,
                value: Some("x".repeat(MAX_ACCESSIBILITY_VALUE_BYTES + 1)),
                delta: None,
            },
        }];
        let error = translate_native_events(&events).expect_err("oversized action value");
        assert!(error.to_string().contains("action value length"));
    }

    #[test]
    fn native_events_bound_composition_utf16_units_and_preserve_supplementary_text() {
        let valid_text = "😀".repeat(MAX_COMPOSITION_UNITS / 2);
        let valid = [WindowEvent::TextComposition {
            phase: metis_platform::native::CompositionPhase::Updated,
            text: valid_text.clone(),
        }];
        let translated = translate_native_events(&valid).expect("boundary composition");
        assert_eq!(translated.len(), 1);
        assert_eq!(
            translated.as_ref(),
            &[PresentationEvent::TextComposition {
                phase: CompositionPhase::Updated,
                text: valid_text.into_boxed_str(),
            }]
        );

        let invalid = [WindowEvent::TextComposition {
            phase: metis_platform::native::CompositionPhase::Updated,
            text: format!("{}A", "😀".repeat(MAX_COMPOSITION_UNITS / 2)),
        }];
        let error = translate_native_events(&invalid).expect_err("over-budget composition");
        assert!(error.to_string().contains("UTF-16 units"));
    }

    #[test]
    fn frame_conversion_preserves_rgba_channels() {
        let frame = PresentationFrame::from_rgba(2, 1, &[0, 0, 0, 0, 200, 150, 100, 255])
            .expect("valid frame");
        let framebuffer = to_framebuffer(&frame).expect("framebuffer");
        assert_eq!(framebuffer.get_pixel(0, 0), Color::rgba(0, 0, 0, 0));
        assert_eq!(framebuffer.get_pixel(1, 0), Color::rgba(200, 150, 100, 255));
    }

    #[test]
    fn native_host_presents_one_frame_and_closes() {
        let frame = PresentationFrame::from_rgba(1, 1, &[12, 34, 56, 255]).expect("valid frame");
        let outcome =
            run_native_frame(frame, "RITK presentation frame test").expect("native frame");
        assert_eq!((outcome.width(), outcome.height()), (1, 1));
        assert_eq!(outcome.pixel_count(), 1);
        assert_eq!(outcome.frame_requests(), 1);
        assert_eq!(outcome.event_batches(), 1);
        assert!(outcome.translated_events() >= 1);
    }
}
