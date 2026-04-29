//! Viewer UI components.
//!
//! # Sub-modules
//!
//! | Module           | Contents                                                |
//! |------------------|---------------------------------------------------------|
//! | [`layout`]       | [`LayoutMode`] and [`ViewportId`] enumerations.        |
//! | [`viewport`]     | [`ViewportState`] and [`ViewportPanel`] — the core MPR  |
//! |                  | slice display widget.                                   |
//! | [`toolbar`]      | [`ToolbarState`] and [`ToolbarPanel`] — top toolbar.    |
//! | [`sidebar`]      | [`SidebarPanel`] — series browser + metadata tab.       |
//! | [`overlay`]      | [`OverlayRenderer`] — DICOM 4-corner text overlays.     |
//! | [`measurements`] | [`MeasurementLayer`] — annotation drawing helpers.     |
//! | [`window_presets`] | [`WindowPreset`] with standard CT/MR presets.         |

pub mod layout;
pub mod measurements;
pub mod overlay;
pub mod sidebar;
pub mod toolbar;
pub mod viewport;
pub mod window_presets;

pub use layout::{LayoutMode, ViewportId};
pub use measurements::MeasurementLayer;
pub use overlay::OverlayRenderer;
pub use sidebar::SidebarPanel;
pub use toolbar::{ToolbarPanel, ToolbarState};
pub use viewport::{ViewportPanel, ViewportState};
pub use window_presets::WindowPreset;
