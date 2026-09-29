//! Pointer mapping for a displayed oblique reslice frame.

use crate::presentation::ViewportPoint;
use thiserror::Error;

/// Maps native screen positions to continuous reslice pixel coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ObliqueViewport {
    origin: [f64; 2],
    pixel_size: [f64; 2],
    dimensions: [u32; 2],
    panel_bounds: [f64; 4],
}

impl ObliqueViewport {
    /// Build a mapping from a rendered image rectangle and frame dimensions.
    ///
    /// # Errors
    /// Returns an error when the image rectangle is non-finite, has a
    /// non-positive pixel size, or has an empty frame dimension.
    pub(crate) fn try_new(
        origin: [f64; 2],
        pixel_size: [f64; 2],
        dimensions: [u32; 2],
        panel_bounds: [f64; 4],
    ) -> Result<Self, ObliqueViewportError> {
        if dimensions.contains(&0) {
            return Err(ObliqueViewportError::EmptyFrame { dimensions });
        }
        if !origin.into_iter().all(f64::is_finite)
            || !pixel_size.into_iter().all(f64::is_finite)
            || pixel_size.into_iter().any(|value| value <= 0.0)
            || !panel_bounds.into_iter().all(f64::is_finite)
            || panel_bounds[0] >= panel_bounds[2]
            || panel_bounds[1] >= panel_bounds[3]
        {
            return Err(ObliqueViewportError::InvalidImageGeometry);
        }
        let viewport = Self {
            origin,
            pixel_size,
            dimensions,
            panel_bounds,
        };
        if viewport.bounds().is_none() {
            return Err(ObliqueViewportError::InvalidImageGeometry);
        }
        Ok(viewport)
    }

    /// Map a screen position to a continuous `[column, row]` pixel coordinate.
    pub(crate) fn map(self, point: ViewportPoint) -> Option<[f64; 2]> {
        let [x, y] = [point.x(), point.y()];
        if !x.is_finite() || !y.is_finite() {
            return None;
        }
        let [left, right, top, bottom] = self.bounds()?;
        let [panel_left, panel_top, panel_right, panel_bottom] = self.panel_bounds;
        if x < left
            || x >= right
            || y < top
            || y >= bottom
            || x < panel_left
            || x >= panel_right
            || y < panel_top
            || y >= panel_bottom
        {
            return None;
        }
        Some([
            ((x - left) / self.pixel_size[0] - 0.5).clamp(0.0, f64::from(self.dimensions[0] - 1)),
            ((y - top) / self.pixel_size[1] - 0.5).clamp(0.0, f64::from(self.dimensions[1] - 1)),
        ])
    }

    pub(crate) fn validate_dimensions(
        self,
        dimensions: [usize; 2],
    ) -> Result<(), ObliqueViewportError> {
        let [width, height] = dimensions;
        if u32::try_from(width).ok() == Some(self.dimensions[0])
            && u32::try_from(height).ok() == Some(self.dimensions[1])
        {
            return Ok(());
        }
        Err(ObliqueViewportError::FrameDimensionsMismatch {
            viewport: self.dimensions,
            plane: dimensions,
        })
    }

    #[cfg(test)]
    pub(crate) fn image_bounds(self) -> [f64; 4] {
        self.bounds()
            .expect("invariant: validated oblique viewport has finite image bounds")
    }

    /// Return the inclusive pixel-coordinate bounds visible inside this pane.
    pub(crate) fn visible_pixel_bounds(self) -> Option<[f64; 4]> {
        let [image_left, image_right, image_top, image_bottom] = self.bounds()?;
        let [panel_left, panel_top, panel_right, panel_bottom] = self.panel_bounds;
        let left = image_left.max(panel_left).ceil();
        let right = image_right.min(panel_right).ceil() - 1.0;
        let top = image_top.max(panel_top).ceil();
        let bottom = image_bottom.min(panel_bottom).ceil() - 1.0;
        (left <= right && top <= bottom).then_some([left, right, top, bottom])
    }

    pub(crate) fn screen_point(self, pixel: [f64; 2]) -> Option<[f64; 2]> {
        if !pixel.into_iter().all(f64::is_finite)
            || pixel[0] < 0.0
            || pixel[1] < 0.0
            || pixel[0] > f64::from(self.dimensions[0] - 1)
            || pixel[1] > f64::from(self.dimensions[1] - 1)
        {
            return None;
        }
        Some([
            self.origin[0] + (pixel[0] + 0.5) * self.pixel_size[0],
            self.origin[1] + (pixel[1] + 0.5) * self.pixel_size[1],
        ])
    }

    fn bounds(self) -> Option<[f64; 4]> {
        let right = self.origin[0] + f64::from(self.dimensions[0]) * self.pixel_size[0];
        let bottom = self.origin[1] + f64::from(self.dimensions[1]) * self.pixel_size[1];
        (right.is_finite() && bottom.is_finite()).then_some([
            self.origin[0],
            right,
            self.origin[1],
            bottom,
        ])
    }
}

/// Invalid oblique image placement supplied by the native composition layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub(crate) enum ObliqueViewportError {
    /// The rendered reslice has an empty dimension.
    #[error("oblique frame dimensions {dimensions:?} must be non-zero")]
    EmptyFrame {
        /// Rejected frame dimensions `[width, height]`.
        dimensions: [u32; 2],
    },
    /// The screen mapping and physical plane describe different pixel grids.
    #[error("oblique viewport dimensions {viewport:?} do not match plane dimensions {plane:?}")]
    FrameDimensionsMismatch {
        /// Displayed pixel dimensions `[width, height]`.
        viewport: [u32; 2],
        /// Physical plane dimensions `[width, height]`.
        plane: [usize; 2],
    },
    /// The image or panel geometry is outside finite positive screen space.
    #[error("oblique image and panel geometry must be finite with positive dimensions")]
    InvalidImageGeometry,
}

#[cfg(test)]
mod tests {
    use super::{ObliqueViewport, ObliqueViewportError};
    use crate::presentation::ViewportPoint;

    #[test]
    fn pixel_centres_map_to_exact_reslice_coordinates() {
        let viewport =
            ObliqueViewport::try_new([10.0, 20.0], [2.0, 4.0], [5, 3], [10.0, 20.0, 20.0, 32.0])
                .expect("finite non-empty oblique image");
        assert_eq!(
            viewport.map(ViewportPoint::new(17.0, 30.0)),
            Some([3.0, 2.0])
        );
        assert_eq!(
            viewport.map(ViewportPoint::new(11.0, 22.0)),
            Some([0.0, 0.0])
        );
        assert_eq!(
            viewport.map(ViewportPoint::new(19.9, 31.9)),
            Some([4.0, 2.0])
        );
    }

    #[test]
    fn pointer_mapping_rejects_padding_and_non_finite_positions() {
        let viewport =
            ObliqueViewport::try_new([10.0, 20.0], [2.0, 4.0], [5, 3], [10.0, 20.0, 18.0, 32.0])
                .expect("finite non-empty oblique image");
        assert_eq!(viewport.map(ViewportPoint::new(9.9, 24.0)), None);
        assert_eq!(viewport.map(ViewportPoint::new(20.0, 24.0)), None);
        assert_eq!(viewport.map(ViewportPoint::new(12.0, f64::NAN)), None);
    }

    #[test]
    fn pointer_mapping_rejects_image_pixels_clipped_by_the_pane() {
        let viewport =
            ObliqueViewport::try_new([8.0, 18.0], [2.0, 4.0], [5, 3], [10.0, 20.0, 18.0, 30.0])
                .expect("finite image extending beyond its panel");
        assert_eq!(viewport.map(ViewportPoint::new(9.0, 22.0)), None);
        assert_eq!(viewport.map(ViewportPoint::new(17.0, 19.0)), None);
        assert_eq!(
            viewport.map(ViewportPoint::new(11.0, 20.0)),
            Some([1.0, 0.0])
        );
        assert_eq!(
            viewport.visible_pixel_bounds(),
            Some([10.0, 17.0, 20.0, 29.0])
        );
    }

    #[test]
    fn invalid_frame_geometry_is_rejected_at_construction() {
        assert!(matches!(
            ObliqueViewport::try_new([0.0, 0.0], [1.0, 1.0], [0, 1], [0.0, 0.0, 1.0, 1.0]),
            Err(ObliqueViewportError::EmptyFrame { .. })
        ));
        assert_eq!(
            ObliqueViewport::try_new(
                [0.0, 0.0],
                [1.0, f64::INFINITY],
                [1, 1],
                [0.0, 0.0, 1.0, 1.0],
            ),
            Err(ObliqueViewportError::InvalidImageGeometry)
        );
        assert_eq!(
            ObliqueViewport::try_new([0.0, 0.0], [1.0, 1.0], [1, 1], [0.0, 0.0, 0.0, 1.0],),
            Err(ObliqueViewportError::InvalidImageGeometry)
        );
    }

    #[test]
    fn mapper_rejects_dimensions_from_another_plane() {
        let viewport =
            ObliqueViewport::try_new([0.0, 0.0], [1.0, 1.0], [5, 3], [0.0, 0.0, 10.0, 10.0])
                .expect("finite non-empty oblique image");
        assert_eq!(
            viewport.validate_dimensions([5, 4]),
            Err(ObliqueViewportError::FrameDimensionsMismatch {
                viewport: [5, 3],
                plane: [5, 4],
            })
        );
    }
}
