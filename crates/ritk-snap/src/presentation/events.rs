//! Format-neutral input and lifecycle events for viewer hosts.

/// Maximum number of presentation events accepted in one host batch.
///
/// The value matches the bounded event queues in the Métis virtual and native
/// surfaces. Keeping the bound on the RITK contract lets browser and native
/// hosts share the same action-dispatch allocation limit.
pub const MAX_PRESENTATION_EVENTS: usize = 1_024;

/// Maximum UTF-16 code units retained for one text-composition update.
///
/// Native Métis input applies the same limit before translation. The shared
/// contract also applies it to browser events so a host cannot bypass the
/// bound by constructing a presentation event directly.
pub const MAX_COMPOSITION_UNITS: usize = 4_096;

/// Mouse button carried by a host pointer event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PointerButton {
    /// Primary pointer button.
    Left,
    /// Secondary pointer button.
    Right,
    /// Auxiliary pointer button.
    Middle,
    /// First extended pointer button.
    X1,
    /// Second extended pointer button.
    X2,
}

/// Phase of a native text-composition transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CompositionPhase {
    /// Composition started.
    Started,
    /// Uncommitted preedit text changed.
    Updated,
    /// Text was committed to the focused control.
    Committed,
    /// Uncommitted composition text was canceled.
    Canceled,
}

/// Input or lifecycle event delivered to the RITK viewer boundary.
///
/// The event carries coordinates, controls and lifecycle state only. It never
/// carries a path, DICOM object, volume, metadata record or host authority.
/// Pointer coordinates use `f64` so native `i32` positions and browser client
/// coordinates share one lossless host representation; the action reducer
/// rejects non-finite values.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum PresentationEvent {
    /// The host requested application shutdown.
    CloseRequested,
    /// The host surface finished destruction.
    Destroyed,
    /// The viewer gained keyboard focus.
    FocusGained,
    /// The viewer lost keyboard focus.
    FocusLost,
    /// The pointer moved in client coordinates.
    PointerMove {
        /// Horizontal client coordinate in display pixels.
        x: f64,
        /// Vertical client coordinate in display pixels.
        y: f64,
    },
    /// A pointer button was pressed in client coordinates.
    PointerDown {
        /// Horizontal client coordinate in display pixels.
        x: f64,
        /// Vertical client coordinate in display pixels.
        y: f64,
        /// Pressed button.
        button: PointerButton,
    },
    /// A pointer button was released in client coordinates.
    PointerUp {
        /// Horizontal client coordinate in display pixels.
        x: f64,
        /// Vertical client coordinate in display pixels.
        y: f64,
        /// Released button.
        button: PointerButton,
    },
    /// A virtual key was pressed.
    KeyDown {
        /// Host virtual-key value.
        virtual_key: u32,
        /// The key message is an auto-repeat.
        repeated: bool,
    },
    /// A virtual key was released.
    KeyUp {
        /// Host virtual-key value.
        virtual_key: u32,
    },
    /// Unicode text was produced by the host.
    TextInput {
        /// One Unicode scalar value.
        character: char,
    },
    /// A bounded text-composition phase changed.
    TextComposition {
        /// Composition lifecycle phase.
        phase: CompositionPhase,
        /// Preedit or committed UTF-8 text.
        text: Box<str>,
    },
    /// The host client size changed.
    Resized {
        /// Horizontal client extent.
        width: u32,
        /// Vertical client extent.
        height: u32,
    },
    /// The host crossed a display scale boundary.
    DpiChanged {
        /// Effective horizontal DPI.
        dpi: u32,
    },
}
