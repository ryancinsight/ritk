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

/// Maximum UTF-8 bytes retained for one native accessibility action value.
///
/// The value is copied from the host request at the native trust boundary and
/// is bounded independently of the provider's queue size. Actions outside
/// this bound are rejected before they reach viewer state.
pub const MAX_ACCESSIBILITY_VALUE_BYTES: usize = 4_096;

const MODIFIER_CTRL: u8 = 0b0001;
const MODIFIER_SHIFT: u8 = 0b0010;
const MODIFIER_ALT: u8 = 0b0100;
const MODIFIER_META: u8 = 0b1000;

/// Modifier-key state captured with one presentation event.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PresentationModifiers {
    bits: u8,
}

impl PresentationModifiers {
    /// No modifier keys are pressed.
    pub const NONE: Self = Self { bits: 0 };

    /// Construct a modifier snapshot from its logical key states.
    #[must_use]
    pub const fn new(ctrl: bool, shift: bool, alt: bool, meta: bool) -> Self {
        let mut bits = 0;
        if ctrl {
            bits |= MODIFIER_CTRL;
        }
        if shift {
            bits |= MODIFIER_SHIFT;
        }
        if alt {
            bits |= MODIFIER_ALT;
        }
        if meta {
            bits |= MODIFIER_META;
        }
        Self { bits }
    }

    /// Returns whether Control was held for the event.
    #[must_use]
    pub const fn ctrl(self) -> bool {
        self.bits & MODIFIER_CTRL != 0
    }

    /// Returns whether Shift was held for the event.
    #[must_use]
    pub const fn shift(self) -> bool {
        self.bits & MODIFIER_SHIFT != 0
    }

    /// Returns whether Alt was held for the event.
    #[must_use]
    pub const fn alt(self) -> bool {
        self.bits & MODIFIER_ALT != 0
    }

    /// Returns whether the platform meta key was held for the event.
    #[must_use]
    pub const fn meta(self) -> bool {
        self.bits & MODIFIER_META != 0
    }
}

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

/// Operation requested by an assistive-technology client.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AccessibilityAction {
    /// Activate the target control.
    Activate,
    /// Move keyboard focus to the target control.
    Focus,
    /// Replace the target control's value.
    SetValue,
    /// Toggle the target control.
    Toggle,
    /// Increment or decrement a bounded target value.
    AdjustValue,
    /// Open the target selection control.
    Open,
}

/// One bounded assistive-technology action delivered by a native host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccessibilityActionRequest {
    /// Stable target node identity supplied by the host semantics tree.
    pub target_node: u64,
    /// Requested operation.
    pub action: AccessibilityAction,
    /// Replacement value when the operation carries text.
    pub value: Option<Box<str>>,
    /// Signed numeric adjustment when the operation increments or decrements.
    pub delta: Option<i8>,
}

/// Input or lifecycle event delivered to the RITK viewer boundary.
///
/// The event carries coordinates, controls and lifecycle state only. It never
/// carries a path, DICOM object, volume, metadata record or host authority.
/// Pointer coordinates use the host viewport's basis: native display pixels
/// or browser content fractions. Both retain `f64` precision until image
/// mapping; the action reducer rejects non-finite values. Wheel displacement
/// units are independent of this position basis.
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
    /// An assistive-technology client requested an operation on a native node.
    ///
    /// RITK currently does not install a native accessibility tree. The
    /// presentation reducer therefore surfaces this event as an unsupported
    /// action instead of silently discarding a host request.
    AccessibilityAction {
        /// Bounded action request supplied by the host semantics tree.
        request: AccessibilityActionRequest,
    },
    /// The pointer moved in client coordinates.
    PointerMove {
        /// Horizontal coordinate in the host viewport basis.
        x: f64,
        /// Vertical coordinate in the host viewport basis.
        y: f64,
    },
    /// A pointer button was pressed in client coordinates.
    PointerDown {
        /// Horizontal coordinate in the host viewport basis.
        x: f64,
        /// Vertical coordinate in the host viewport basis.
        y: f64,
        /// Pressed button.
        button: PointerButton,
    },
    /// A pointer button was released in client coordinates.
    PointerUp {
        /// Horizontal coordinate in the host viewport basis.
        x: f64,
        /// Vertical coordinate in the host viewport basis.
        y: f64,
        /// Released button.
        button: PointerButton,
    },
    /// A browser or native host canceled an active pointer without a release.
    PointerCancel {
        /// Horizontal coordinate in the host viewport basis.
        x: f64,
        /// Vertical coordinate in the host viewport basis.
        y: f64,
        /// Canceled button.
        button: PointerButton,
    },
    /// A wheel rotated at client coordinates.
    ///
    /// The deltas retain the host's signed units as finite `f64` values. A
    /// native Win32 detent is represented exactly; browser hosts may pass
    /// pixel, line or page values after applying their own unit policy.
    PointerWheel {
        /// Horizontal coordinate in the host viewport basis.
        x: f64,
        /// Vertical coordinate in the host viewport basis.
        y: f64,
        /// Signed horizontal wheel delta in host units.
        delta_x: f64,
        /// Signed vertical wheel delta in host units.
        delta_y: f64,
        /// Modifier keys held when the wheel event was received.
        modifiers: PresentationModifiers,
    },
    /// A virtual key was pressed.
    KeyDown {
        /// Host virtual-key value.
        virtual_key: u32,
        /// The key message is an auto-repeat.
        repeated: bool,
        /// Modifier keys held when the key message was received.
        modifiers: PresentationModifiers,
    },
    /// A virtual key was released.
    KeyUp {
        /// Host virtual-key value.
        virtual_key: u32,
        /// Modifier keys still held after this key was released.
        modifiers: PresentationModifiers,
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
