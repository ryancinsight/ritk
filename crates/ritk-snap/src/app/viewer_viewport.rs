//! Coordinate mapping for one displayed viewer slice.

use crate::presentation::ViewportPoint;
use crate::tools::interaction::{ImagePoint, ViewportOffset};
use crate::ui::ViewTransform;
use thiserror::Error;

/// Geometry needed to map host client coordinates into one displayed slice.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ViewerViewport {
    axis: usize,
    origin: [f64; 2],
    texel_size: [f64; 2],
    source_size: [usize; 2],
    transform: ViewTransform,
    zoom: f64,
    pan: [f64; 2],
}

impl ViewerViewport {
    /// Return the orthogonal viewer axis covered by this viewport.
    pub(crate) const fn axis(self) -> usize {
        self.axis
    }

    /// Construct a viewport mapping from validated image placement values.
    ///
    /// # Errors
    /// Returns an error when the axis, image dimensions or screen geometry is
    /// outside the finite positive range required by the inverse mapping.
    #[cfg(any(not(target_arch = "wasm32"), test))]
    pub(crate) fn new(
        axis: usize,
        origin: [f64; 2],
        texel_size: [f64; 2],
        source_size: [usize; 2],
        transform: ViewTransform,
    ) -> Result<Self, ViewerViewportError> {
        Self::new_with_zoom_pan(
            axis,
            origin,
            texel_size,
            source_size,
            transform,
            1.0,
            ViewportOffset::new(0.0, 0.0),
        )
    }

    /// Construct a viewport mapping with the viewer's zoom and pan state.
    ///
    /// The transform is applied in displayed output coordinates before the
    /// orientation inverse. Browser raster presentation uses the same
    /// equation, keeping pointer actions aligned with the transformed pixels.
    pub(crate) fn new_with_zoom_pan(
        axis: usize,
        origin: [f64; 2],
        texel_size: [f64; 2],
        source_size: [usize; 2],
        transform: ViewTransform,
        zoom: f32,
        pan: ViewportOffset,
    ) -> Result<Self, ViewerViewportError> {
        if axis > 2 {
            return Err(ViewerViewportError::Axis { axis });
        }
        if source_size.contains(&0) {
            return Err(ViewerViewportError::EmptyImage { source_size });
        }
        if !origin.iter().all(|value| value.is_finite())
            || !texel_size.iter().all(|value| value.is_finite())
            || texel_size[0] <= 0.0
            || texel_size[1] <= 0.0
        {
            return Err(ViewerViewportError::InvalidScreenGeometry);
        }
        if !zoom.is_finite() || zoom <= 0.0 || !pan.x().is_finite() || !pan.y().is_finite() {
            return Err(ViewerViewportError::InvalidViewTransform);
        }
        Ok(Self {
            axis,
            origin,
            texel_size,
            source_size,
            transform,
            zoom: f64::from(zoom),
            pan: [f64::from(pan.x()), f64::from(pan.y())],
        })
    }

    fn screen_bounds(self) -> Option<[f64; 4]> {
        let [width, height] = self.transform.output_size(self.source_size);
        let min_x = self.origin[0];
        let min_y = self.origin[1];
        let extent_x = self.texel_size[0] * width as f64;
        let extent_y = self.texel_size[1] * height as f64;
        let max_x = min_x + extent_x;
        let max_y = min_y + extent_y;
        if !extent_x.is_finite()
            || !extent_y.is_finite()
            || !max_x.is_finite()
            || !max_y.is_finite()
        {
            return None;
        }
        Some([min_x, max_x, min_y, max_y])
    }

    pub(crate) fn map(self, point: ViewportPoint) -> Option<ImagePoint> {
        let x = point.x();
        let y = point.y();
        if !x.is_finite() || !y.is_finite() {
            return None;
        }
        let [min_x, max_x, min_y, max_y] = self.screen_bounds()?;
        if x < min_x || x > max_x || y < min_y || y > max_y {
            return None;
        }
        let output_size = self.transform.output_size(self.source_size);
        let output = [
            ((x - min_x) / self.texel_size[0]).clamp(0.0, output_size[0] as f64 * 0.999_999),
            ((y - min_y) / self.texel_size[1]).clamp(0.0, output_size[1] as f64 * 0.999_999),
        ];
        let center = [output_size[0] as f64 / 2.0, output_size[1] as f64 / 2.0];
        let output = [
            ((output[0] - center[0] - self.pan[0]) / self.zoom) + center[0],
            ((output[1] - center[1] - self.pan[1]) / self.zoom) + center[1],
        ];
        if output[0] < 0.0
            || output[0] >= output_size[0] as f64
            || output[1] < 0.0
            || output[1] >= output_size[1] as f64
        {
            return None;
        }
        let [source_x, source_y] = self
            .transform
            .output_to_source_coordinates(output, self.source_size);
        if !source_x.is_finite()
            || !source_y.is_finite()
            || source_x < f64::from(f32::MIN)
            || source_x > f64::from(f32::MAX)
            || source_y < f64::from(f32::MIN)
            || source_y > f64::from(f32::MAX)
        {
            return None;
        }
        Some(ImagePoint::new(source_x as f32, source_y as f32))
    }
}

/// Error raised while validating a viewport action mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub(crate) enum ViewerViewportError {
    /// The viewport axis is outside the three orthogonal viewer axes.
    #[error("viewport axis {axis} is outside the supported range 0..=2")]
    Axis {
        /// Invalid axis value.
        axis: usize,
    },
    /// The source image has a zero dimension.
    #[error("viewport source dimensions {source_size:?} contain an empty axis")]
    EmptyImage {
        /// Invalid source dimensions.
        source_size: [usize; 2],
    },
    /// Origin or texel scale cannot represent a positive finite rectangle.
    #[error("viewport screen geometry must be finite with positive texel sizes")]
    InvalidScreenGeometry,
    /// Zoom or pan contains a non-finite or non-positive value.
    #[error("viewport zoom and pan must be finite with positive zoom")]
    InvalidViewTransform,
}
