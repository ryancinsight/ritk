//! Format-neutral presentation outputs for RITK viewer hosts.
//!
//! The viewer owns decoding, geometry and display transforms. This module
//! converts those validated results into bounded pixels and translates host
//! input into typed events that Métis can use without receiving DICOM state.

mod actions;
#[cfg(any(target_arch = "wasm32", test))]
pub(crate) mod browser_coordinates;
#[cfg(any(target_arch = "wasm32", test))]
mod browser_policy;
mod events;
mod frame;
mod snapshot;
#[cfg(any(target_arch = "wasm32", test))]
mod viewport;
mod web_keys;

#[cfg(target_arch = "wasm32")]
mod web;

#[cfg(windows)]
mod native;
#[cfg(windows)]
mod native_session;

pub use actions::{
    ActionDispatchError, PointerDelta, PointerGesture, PresentationDispatcher, ViewerAction,
    ViewportPoint, WheelDelta,
};
pub use events::{
    AccessibilityAction, AccessibilityActionRequest, CompositionPhase, PointerButton,
    PresentationEvent, PresentationModifiers, MAX_ACCESSIBILITY_VALUE_BYTES, MAX_COMPOSITION_UNITS,
    MAX_PRESENTATION_EVENTS,
};
pub use frame::{PresentationFrame, PresentationSpacing};
pub use snapshot::{AnnotationKind, AnnotationSummary, PresentationSnapshot};

#[cfg(target_arch = "wasm32")]
pub use web::{WebCanvasInputError, WebCanvasPresenter};

#[cfg(windows)]
pub use native::{run_native_frame, translate_native_events, NativeFrameOutcome};
#[cfg(windows)]
pub use native_session::{run_native_viewer, NativeViewerOutcome};
