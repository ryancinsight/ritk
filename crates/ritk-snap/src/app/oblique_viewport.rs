//! Screen mapping for a rendered oblique plane.

use crate::app::screen_image_geometry::ScreenImageGeometry;
use crate::presentation::ViewportPoint;
use thiserror::Error;

/// Maps host screen positions to continuous reslice pixel coordinates.
///
/// The origin and pixel size describe the image rectangle produced by the
/// current layout. Rebuilding this value after a resize, pan, or zoom keeps
/// input mapping tied to the rectangle that was actually rendered.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ObliqueViewport {
    geometry: ScreenImageGeometry,
    dimensions: [u32; 2],
    panel_bounds: [f64; 4],
}

impl ObliqueViewport {
    /// Build a mapping from the rendered image rectangle and frame dimensions.
    ///
    /// `origin` is the image's upper-left edge in host coordinates;
    /// `pixel_size` is one reslice pixel's host-space width and height;
    /// `panel_bounds` clips pointer input and overlays to the pane.
    ///
    /// # Errors
    /// Returns an error when dimensions are empty, geometry is non-finite or
    /// non-positive, or the pixel centres cannot be represented distinctly
    /// from one another and the image edges.
    pub(crate) fn try_new(
        origin: [f64; 2],
        pixel_size: [f64; 2],
        dimensions: [u32; 2],
        panel_bounds: [f64; 4],
    ) -> Result<Self, ObliqueViewportError> {
        if dimensions.contains(&0) {
            return Err(ObliqueViewportError::EmptyFrame { dimensions });
        }
        if !panel_bounds.into_iter().all(f64::is_finite)
            || panel_bounds[0] >= panel_bounds[2]
            || panel_bounds[1] >= panel_bounds[3]
        {
            return Err(ObliqueViewportError::InvalidImageGeometry);
        }

        let geometry = ScreenImageGeometry::try_new(origin, pixel_size, dimensions)
            .ok_or(ObliqueViewportError::InvalidImageGeometry)?;
        let [_, right, _, bottom] = geometry.bounds();
        let Some(first_center) = geometry.pixel_center_to_screen([0.0, 0.0]) else {
            return Err(ObliqueViewportError::InvalidImageGeometry);
        };
        let Some(last_center) = geometry
            .pixel_center_to_screen([f64::from(dimensions[0] - 1), f64::from(dimensions[1] - 1)])
        else {
            return Err(ObliqueViewportError::InvalidImageGeometry);
        };
        if right <= origin[0]
            || bottom <= origin[1]
            || first_center[0] <= origin[0]
            || first_center[1] <= origin[1]
            || last_center[0] >= right
            || last_center[1] >= bottom
            || !pixel_centres_are_distinct(
                first_center[0],
                last_center[0],
                origin[0],
                geometry.pixel_size()[0],
                dimensions[0],
            )
            || !pixel_centres_are_distinct(
                first_center[1],
                last_center[1],
                origin[1],
                geometry.pixel_size()[1],
                dimensions[1],
            )
        {
            return Err(ObliqueViewportError::InvalidImageGeometry);
        }
        Ok(Self {
            geometry,
            dimensions,
            panel_bounds,
        })
    }

    /// Map a host position to continuous `[column, row]` reslice coordinates.
    ///
    /// A pixel centre maps to its integer index. Positions in the first or
    /// last half-pixel are clamped to the edge pixel, while pane padding and
    /// points outside the rendered image return `None`.
    pub(crate) fn map(self, point: ViewportPoint) -> Option<[f64; 2]> {
        let [x, y] = [point.x(), point.y()];
        if !x.is_finite() || !y.is_finite() {
            return None;
        }
        let [panel_left, panel_top, panel_right, panel_bottom] = self.panel_bounds;
        if x < panel_left || x >= panel_right || y < panel_top || y >= panel_bottom {
            return None;
        }
        self.geometry.map_pixel_center(point)
    }

    /// Ensure this mapping describes the supplied physical plane's pixel grid.
    ///
    /// # Errors
    /// Returns [`ObliqueViewportError::FrameDimensionsMismatch`] when the
    /// rendered frame and plane have different dimensions.
    pub(crate) fn validate_dimensions(
        self,
        dimensions: [usize; 2],
    ) -> Result<(), ObliqueViewportError> {
        if u32::try_from(dimensions[0]).ok() == Some(self.dimensions[0])
            && u32::try_from(dimensions[1]).ok() == Some(self.dimensions[1])
        {
            return Ok(());
        }
        Err(ObliqueViewportError::FrameDimensionsMismatch {
            viewport: self.dimensions,
            plane: dimensions,
        })
    }

    /// Return visible integer host-pixel centres as `[left, right, top, bottom]`.
    pub(crate) fn visible_pixel_bounds(self) -> Option<[f64; 4]> {
        let [image_left, image_right, image_top, image_bottom] = self.geometry.bounds();
        let [panel_left, panel_top, panel_right, panel_bottom] = self.panel_bounds;
        let left = image_left.max(panel_left).ceil();
        let right = last_integer_before(image_right.min(panel_right))?;
        let top = image_top.max(panel_top).ceil();
        let bottom = last_integer_before(image_bottom.min(panel_bottom))?;
        (left <= right && top <= bottom).then_some([left, right, top, bottom])
    }

    /// Map a continuous reslice coordinate to its rendered pixel centre.
    pub(crate) fn screen_point(self, pixel: [f64; 2]) -> Option<[f64; 2]> {
        self.geometry.pixel_center_to_screen(pixel)
    }
}

/// Return whether adjacent centres remain distinct after binary64 rounding.
fn pixel_centres_are_distinct(first: f64, last: f64, origin: f64, step: f64, count: u32) -> bool {
    if count <= 1 {
        return true;
    }
    if first >= last {
        return false;
    }
    let first_spacing = first.next_up() - first;
    let last_spacing = last - last.next_down();
    let spacing = first_spacing.max(last_spacing);
    step > spacing || (step == spacing && first - origin == step * 0.5)
}

/// Return the greatest representable integer strictly below a half-open edge.
fn last_integer_before(edge: f64) -> Option<f64> {
    let previous = edge.next_down();
    previous.is_finite().then_some(previous.floor())
}

/// Invalid image placement or rendered-frame geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub(crate) enum ObliqueViewportError {
    /// The rendered reslice has an empty dimension.
    #[error("oblique frame dimensions {dimensions:?} must be non-zero")]
    EmptyFrame {
        /// Rejected frame dimensions `[width, height]`.
        dimensions: [u32; 2],
    },
    /// The rendered frame and physical plane use different pixel grids.
    #[error("oblique viewport dimensions {viewport:?} do not match plane dimensions {plane:?}")]
    FrameDimensionsMismatch {
        /// Displayed pixel dimensions `[width, height]`.
        viewport: [u32; 2],
        /// Physical plane dimensions `[width, height]`.
        plane: [usize; 2],
    },
    /// The image or pane geometry is non-finite, empty, or unrepresentable.
    #[error("oblique image and pane geometry must be finite and have positive dimensions")]
    InvalidImageGeometry,
}

#[cfg(test)]
#[path = "oblique_viewport/tests.rs"]
mod tests;
