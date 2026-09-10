//! Stateful reduction of host events into viewer actions.
//!
//! The presentation event contract is deliberately close to a host protocol:
//! it reports coordinates and lifecycle values without deciding what a viewer
//! gesture means. This module owns that reduction. It keeps button state in a
//! fixed array, emits deterministic gesture actions, and commits state only
//! after a complete bounded batch succeeds.

use super::{
    CompositionPhase, PointerButton, PresentationEvent, MAX_COMPOSITION_UNITS,
    MAX_PRESENTATION_EVENTS,
};
use thiserror::Error;

const POINTER_BUTTONS: [PointerButton; 5] = [
    PointerButton::Left,
    PointerButton::Right,
    PointerButton::Middle,
    PointerButton::X1,
    PointerButton::X2,
];
const MAX_ACTIONS_PER_EVENT: usize = 6;

/// A client-space point carried by a viewer action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ViewportPoint {
    x: i32,
    y: i32,
}

impl ViewportPoint {
    /// Construct a point from signed client coordinates.
    #[must_use]
    pub const fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }

    /// Horizontal client coordinate.
    #[must_use]
    pub const fn x(self) -> i32 {
        self.x
    }

    /// Vertical client coordinate.
    #[must_use]
    pub const fn y(self) -> i32 {
        self.y
    }

    fn delta_to(self, to: Self) -> Result<PointerDelta, ActionDispatchError> {
        let x =
            to.x.checked_sub(self.x)
                .ok_or(ActionDispatchError::CoordinateDeltaOverflow { from: self, to })?;
        let y =
            to.y.checked_sub(self.y)
                .ok_or(ActionDispatchError::CoordinateDeltaOverflow { from: self, to })?;
        Ok(PointerDelta { x, y })
    }
}

/// A checked client-space displacement between two pointer positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PointerDelta {
    x: i32,
    y: i32,
}

impl PointerDelta {
    /// Horizontal displacement.
    #[must_use]
    pub const fn x(self) -> i32 {
        self.x
    }

    /// Vertical displacement.
    #[must_use]
    pub const fn y(self) -> i32 {
        self.y
    }

    fn is_zero(self) -> bool {
        self.x == 0 && self.y == 0
    }
}

/// Gesture classification attached to a completed pointer press.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointerGesture {
    /// The button was released without changing its pressed position.
    Click,
    /// The button moved while pressed or was released at another position.
    Drag,
}

/// Viewer-facing action reduced from one or more presentation events.
///
/// Pointer movement emits [`ViewerAction::PointerMoved`] for the host
/// position. A pressed button additionally emits a [`ViewerAction::PointerDragged`]
/// action for each non-zero movement, in the fixed [`PointerButton`] order.
/// Release emits one action with a [`PointerGesture`] classification, so a
/// viewer does not have to infer clicks from a second event stream.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ViewerAction {
    /// The host requested application shutdown.
    CloseRequested,
    /// The host surface was destroyed.
    Destroyed,
    /// Keyboard focus changed.
    FocusChanged {
        /// Whether the viewer now owns keyboard focus.
        focused: bool,
    },
    /// The pointer moved in client coordinates.
    PointerMoved {
        /// Current client position.
        position: ViewportPoint,
    },
    /// A pointer button was pressed.
    PointerPressed {
        /// Pressed button.
        button: PointerButton,
        /// Press position.
        position: ViewportPoint,
    },
    /// A pressed pointer button moved by a non-zero checked displacement.
    PointerDragged {
        /// Dragging button.
        button: PointerButton,
        /// Position where the press began.
        start: ViewportPoint,
        /// Position after this movement.
        current: ViewportPoint,
        /// Displacement from the preceding movement position.
        delta: PointerDelta,
    },
    /// A pointer button was released and classified as a click or drag.
    PointerReleased {
        /// Released button.
        button: PointerButton,
        /// Release position.
        position: ViewportPoint,
        /// Completed gesture classification.
        gesture: PointerGesture,
    },
    /// A focus loss canceled an in-progress pointer gesture.
    PointerCancelled {
        /// Canceled button.
        button: PointerButton,
        /// Last position known for the gesture.
        position: ViewportPoint,
    },
    /// A virtual key was pressed.
    KeyPressed {
        /// Host virtual-key value.
        virtual_key: u32,
        /// Whether the host reported an auto-repeat.
        repeated: bool,
    },
    /// A virtual key was released.
    KeyReleased {
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

/// Failure while reducing a bounded presentation event batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum ActionDispatchError {
    /// The input batch exceeds the shared host bound.
    #[error("presentation event batch length {actual} exceeds limit {limit}")]
    BatchTooLarge {
        /// Number of events supplied by the host.
        actual: usize,
        /// Maximum accepted event count.
        limit: usize,
    },
    /// The deterministic action bound would be exceeded.
    #[error("viewer action batch exceeds limit {limit}")]
    ActionBatchTooLarge {
        /// Number of actions already produced.
        actual: usize,
        /// Maximum action count for this input batch.
        limit: usize,
    },
    /// A bounded output or text allocation could not be reserved.
    #[error("unable to reserve {requested} bounded viewer entries")]
    AllocationFailure {
        /// Number of entries or bytes requested.
        requested: usize,
    },
    /// The event sequence contains two presses for one button.
    #[error("pointer button {button:?} was pressed twice without release")]
    DuplicatePointerPress {
        /// Button whose state is already pressed.
        button: PointerButton,
    },
    /// The event sequence releases a button that is not pressed.
    #[error("pointer button {button:?} was released without a press")]
    PointerReleaseWithoutPress {
        /// Button whose state is absent.
        button: PointerButton,
    },
    /// A checked movement cannot be represented by signed client coordinates.
    #[error("pointer delta from {from:?} to {to:?} overflows i32")]
    CoordinateDeltaOverflow {
        /// Previous pointer position.
        from: ViewportPoint,
        /// New pointer position.
        to: ViewportPoint,
    },
    /// A text-composition update exceeds the shared UTF-16 bound.
    #[error("text composition length {actual} exceeds limit {limit} UTF-16 units")]
    CompositionTooLong {
        /// Number of UTF-16 code units supplied by the host.
        actual: usize,
        /// Maximum accepted UTF-16 code units.
        limit: usize,
    },
}

#[derive(Debug, Clone, Copy)]
struct PointerPress {
    origin: ViewportPoint,
    last: ViewportPoint,
    moved: bool,
}

/// Stateful presentation-event reducer for one viewer surface.
#[derive(Debug, Clone, Default)]
pub struct PresentationDispatcher {
    pointers: [Option<PointerPress>; POINTER_BUTTONS.len()],
}

impl PresentationDispatcher {
    /// Construct a dispatcher with no pressed buttons.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            pointers: [None; POINTER_BUTTONS.len()],
        }
    }

    /// Reduce one complete bounded host batch into viewer actions.
    ///
    /// State changes are committed only after every event and action has been
    /// accepted. A failed batch therefore leaves the dispatcher ready to
    /// retry or report the same input without a partial gesture state.
    ///
    /// # Errors
    /// Returns [`ActionDispatchError`] for an oversized batch, malformed
    /// pointer sequence, overflowing coordinate delta, overlong composition,
    /// or bounded allocation failure.
    pub fn dispatch(
        &mut self,
        events: &[PresentationEvent],
    ) -> Result<Box<[ViewerAction]>, ActionDispatchError> {
        if events.len() > MAX_PRESENTATION_EVENTS {
            return Err(ActionDispatchError::BatchTooLarge {
                actual: events.len(),
                limit: MAX_PRESENTATION_EVENTS,
            });
        }
        let action_limit = events.len().checked_mul(MAX_ACTIONS_PER_EVENT).ok_or(
            ActionDispatchError::AllocationFailure {
                requested: usize::MAX,
            },
        )?;
        let mut actions = Vec::new();
        actions.try_reserve_exact(action_limit).map_err(|_| {
            ActionDispatchError::AllocationFailure {
                requested: action_limit,
            }
        })?;
        let mut next = self.clone();
        for event in events {
            next.apply_event(event, &mut actions, action_limit)?;
        }
        *self = next;
        Ok(actions.into_boxed_slice())
    }
}

mod reduce;
#[cfg(test)]
mod tests;
