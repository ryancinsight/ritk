//! The set of pointer buttons a native host surface currently holds.

use crate::presentation::PointerButton;

/// Pointer buttons held since their press, independent of event ordering.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct PressedButtons(u8);

impl PressedButtons {
    pub(super) const EMPTY: Self = Self(0);

    const fn bit(button: PointerButton) -> u8 {
        match button {
            PointerButton::Left => 1,
            PointerButton::Right => 2,
            PointerButton::Middle => 4,
            PointerButton::X1 => 8,
            PointerButton::X2 => 16,
        }
    }

    pub(super) const fn contains(self, button: PointerButton) -> bool {
        self.0 & Self::bit(button) != 0
    }

    pub(super) const fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub(super) const fn insert(&mut self, button: PointerButton) {
        self.0 |= Self::bit(button);
    }

    pub(super) const fn remove(&mut self, button: PointerButton) {
        self.0 &= !Self::bit(button);
    }
}
