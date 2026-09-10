//! Format-neutral presentation outputs for RITK viewer hosts.
//!
//! The viewer owns decoding, geometry and display transforms. This module
//! converts those validated results into bounded pixels and translates host
//! input into typed events that Métis can use without receiving DICOM state.

mod events;
mod frame;

#[cfg(windows)]
mod native;

pub use events::{CompositionPhase, PointerButton, PresentationEvent};
pub use frame::PresentationFrame;

#[cfg(windows)]
pub use native::{run_native_frame, translate_native_events, NativeFrameOutcome};
