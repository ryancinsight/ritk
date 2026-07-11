//! Viewer navigation state.

use serde::{Deserialize, Serialize};

/// Navigation state for a volume viewer.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ViewerState {
    /// Current slice index along the depth axis.
    pub slice_index: usize,
    /// Window center for intensity display.
    pub window_center: Option<f32>,
    /// Window width for intensity display.
    pub window_width: Option<f32>,
}

impl ViewerState {
    /// Create a default state at the first slice.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            slice_index: 0,
            window_center: None,
            window_width: None,
        }
    }
}

impl Default for ViewerState {
    fn default() -> Self {
        Self::new()
    }
}

/// Default window centre used when no explicit value is present.
pub(crate) const DEFAULT_WINDOW_CENTER: f32 = 128.0;

/// Default window width used when no explicit value is present.
pub(crate) const DEFAULT_WINDOW_WIDTH: f32 = 256.0;
