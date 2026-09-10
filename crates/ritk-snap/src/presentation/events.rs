//! Format-neutral input and lifecycle events for viewer hosts.

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
#[derive(Debug, Clone, PartialEq, Eq)]
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
        /// Horizontal client coordinate.
        x: i32,
        /// Vertical client coordinate.
        y: i32,
    },
    /// A pointer button was pressed in client coordinates.
    PointerDown {
        /// Horizontal client coordinate.
        x: i32,
        /// Vertical client coordinate.
        y: i32,
        /// Pressed button.
        button: PointerButton,
    },
    /// A pointer button was released in client coordinates.
    PointerUp {
        /// Horizontal client coordinate.
        x: i32,
        /// Vertical client coordinate.
        y: i32,
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
