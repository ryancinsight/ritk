//! Format-neutral presentation outputs for RITK viewer hosts.
//!
//! The viewer owns decoding, geometry and display transforms. This module
//! converts those validated results into bounded pixels that a host such as
//! Métis can present without receiving DICOM state.

mod frame;

#[cfg(windows)]
mod native;

pub use frame::PresentationFrame;

#[cfg(windows)]
pub use native::{run_native_frame, NativeFrameOutcome};
