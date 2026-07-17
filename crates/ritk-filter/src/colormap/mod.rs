//! Scalar-to-RGB colormap and label overlays module.

pub mod overlay;
pub mod scalar;

pub use overlay::{LabelMapContourOverlayFilter, LabelOverlayFilter, LabelToRGBFilter};
pub use scalar::{Colormap, ScalarToRGBColormapFilter};
