//! Coordinate mapping for one displayed viewer slice.

use crate::app::screen_image_geometry::ScreenImageGeometry;
use crate::presentation::ViewportPoint;
use crate::tools::interaction::{ImagePoint, ViewportOffset};
use crate::ui::ViewTransform;
use thiserror::Error;

// Every integer through this index is exactly representable in ImagePoint.
const MAX_EXACT_PIXEL_INDEX: u32 = 1_u32 << f32::MANTISSA_DIGITS;
// A half-open image extent includes the last exact index and its far edge.
const MAX_IMAGE_EXTENT: u32 = MAX_EXACT_PIXEL_INDEX + 1;

/// Geometry needed to map host client coordinates into one displayed slice.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ViewerViewport {
    axis: usize,
    geometry: ScreenImageGeometry,
    source_size: [usize; 2],
    source_dimensions: [u32; 2],
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
        if !zoom.is_finite() || zoom <= 0.0 || !pan.x().is_finite() || !pan.y().is_finite() {
            return Err(ViewerViewportError::InvalidViewTransform);
        }
        let output_size = transform.output_size(source_size);
        let [Ok(width), Ok(height)] = output_size.map(u32::try_from) else {
            return Err(ViewerViewportError::InvalidScreenGeometry);
        };
        if [width, height]
            .into_iter()
            .any(|dimension| dimension > MAX_IMAGE_EXTENT)
        {
            return Err(ViewerViewportError::InvalidScreenGeometry);
        }
        let [Ok(source_width), Ok(source_height)] = source_size.map(u32::try_from) else {
            return Err(ViewerViewportError::InvalidScreenGeometry);
        };
        let geometry = ScreenImageGeometry::try_new(origin, texel_size, [width, height])
            .ok_or(ViewerViewportError::InvalidScreenGeometry)?;
        Ok(Self {
            axis,
            geometry,
            source_size,
            source_dimensions: [source_width, source_height],
            transform,
            zoom: f64::from(zoom),
            pan: [f64::from(pan.x()), f64::from(pan.y())],
        })
    }

    pub(crate) fn map(self, point: ViewportPoint) -> Option<ImagePoint> {
        let output = self.geometry.map_edge(point)?;
        let output_size = self.geometry.dimensions().map(f64::from);
        let center = [output_size[0] / 2.0, output_size[1] / 2.0];
        let output = [
            ((output[0] - center[0] - self.pan[0]) / self.zoom) + center[0],
            ((output[1] - center[1] - self.pan[1]) / self.zoom) + center[1],
        ];
        if output[0] < 0.0
            || output[0] >= output_size[0]
            || output[1] < 0.0
            || output[1] >= output_size[1]
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
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the pointer-action contract stores f32 image coordinates; the dimensions bound the pixel index"
        )]
        let image_x = source_x as f32;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the pointer-action contract stores f32 image coordinates; the dimensions bound the pixel index"
        )]
        let image_y = source_y as f32;
        if f64::from(image_x) >= f64::from(self.source_dimensions[0])
            || f64::from(image_y) >= f64::from(self.source_dimensions[1])
        {
            return None;
        }
        Some(ImagePoint::new(image_x, image_y))
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
    /// Geometry is invalid or a raster dimension exceeds the mapping range.
    #[error("viewport screen geometry must be finite, positive, and fit its mapping range")]
    InvalidScreenGeometry,
    /// Zoom or pan contains a non-finite or non-positive value.
    #[error("viewport zoom and pan must be finite with positive zoom")]
    InvalidViewTransform,
}
