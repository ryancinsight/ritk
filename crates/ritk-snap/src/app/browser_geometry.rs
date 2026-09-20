//! Physical geometry for RITK-owned browser canvases.

use super::viewer_viewport::ViewerViewport;
use crate::presentation::PresentationSpacing;
use crate::tools::interaction::ViewportOffset;
use crate::ui::ViewTransform;
use std::io;

/// A positive finite CSS width-to-height ratio for one rendered slice.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct PhysicalCanvasAspect(f64);

impl PhysicalCanvasAspect {
    /// Computes the aspect from frame-ordered row and column distances.
    ///
    /// Pixel and spacing factors are normalized independently before they are
    /// multiplied. This preserves common-unit scale invariance and prevents
    /// otherwise-valid large sample distances from overflowing intermediate
    /// physical extents.
    pub(super) fn from_display_spacing(
        display_spacing: PresentationSpacing,
        width: u32,
        height: u32,
    ) -> io::Result<Self> {
        if width == 0 || height == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "browser slice dimensions must be positive",
            ));
        }
        let [row_spacing, column_spacing] = display_spacing.values();
        let spacing_scale = row_spacing.max(column_spacing);
        let dimension_scale = f64::from(width.max(height));
        let physical_width =
            (f64::from(width) / dimension_scale) * (column_spacing / spacing_scale);
        let physical_height = (f64::from(height) / dimension_scale) * (row_spacing / spacing_scale);
        let ratio = physical_width / physical_height;
        if !physical_width.is_finite()
            || physical_width <= 0.0
            || !physical_height.is_finite()
            || physical_height <= 0.0
            || !ratio.is_finite()
            || ratio <= 0.0
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "browser slice physical aspect is not representable",
            ));
        }
        Ok(Self(ratio))
    }

    pub(super) fn attribute_value(self) -> String {
        self.0.to_string()
    }
}

/// Maps a content-coordinate extent onto the current backing frame.
///
/// Browser events use fractions of their measured content box, so their
/// matching display extent is `[1.0, 1.0]`, independent of CSS transforms.
#[cfg(test)]
pub(super) fn viewport_for_display(
    axis: usize,
    display_size: [f64; 2],
    frame_size: [u32; 2],
) -> io::Result<ViewerViewport> {
    viewport_for_display_with_zoom_pan(
        axis,
        display_size,
        frame_size,
        1.0,
        ViewportOffset::new(0.0, 0.0),
    )
}

/// Maps browser content coordinates through the current viewer zoom and pan.
pub(super) fn viewport_for_display_with_zoom_pan(
    axis: usize,
    display_size: [f64; 2],
    frame_size: [u32; 2],
    zoom: f32,
    pan: ViewportOffset,
) -> io::Result<ViewerViewport> {
    let [frame_width, frame_height] = frame_size;
    if frame_width == 0 || frame_height == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "browser frame dimensions must be positive",
        ));
    }
    let width = usize::try_from(frame_width)
        .map_err(|_| io::Error::other("browser frame width exceeds host range"))?;
    let height = usize::try_from(frame_height)
        .map_err(|_| io::Error::other("browser frame height exceeds host range"))?;
    ViewerViewport::new_with_zoom_pan(
        axis,
        [0.0, 0.0],
        [
            display_size[0] / f64::from(frame_width),
            display_size[1] / f64::from(frame_height),
        ],
        [width, height],
        ViewTransform::default(),
        zoom,
        pan,
    )
    .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::{viewport_for_display, viewport_for_display_with_zoom_pan, PhysicalCanvasAspect};
    use crate::presentation::PresentationSpacing;
    use crate::tools::interaction::ViewportOffset;
    use std::io;

    fn ratio(display_spacing: [f64; 2], dimensions: [u32; 2]) -> f64 {
        let display_spacing = PresentationSpacing::try_new(display_spacing[0], display_spacing[1])
            .expect("valid display spacing");
        PhysicalCanvasAspect::from_display_spacing(display_spacing, dimensions[0], dimensions[1])
            .expect("valid physical canvas geometry")
            .0
    }

    fn assert_close(actual: f64, expected: f64) {
        // Construction performs at most eight rounded f64 arithmetic steps.
        let bound = 8.0 * f64::EPSILON * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= bound,
            "actual {actual}, expected {expected}, bound {bound}"
        );
    }

    fn assert_invalid(result: io::Result<PhysicalCanvasAspect>) {
        assert_eq!(
            result.expect_err("invalid physical canvas geometry").kind(),
            io::ErrorKind::InvalidInput
        );
    }

    #[test]
    fn frame_spacing_preserves_physical_extents() {
        assert_close(ratio([0.5, 0.5], [512, 512]), 1.0);
        assert_close(ratio([2.5, 0.5], [512, 94]), 256.0 / 235.0);
    }

    #[test]
    fn isotropic_spacing_preserves_pixel_aspect() {
        assert_close(ratio([1.0; 2], [512, 94]), 512.0 / 94.0);
        let square = PhysicalCanvasAspect::from_display_spacing(
            PresentationSpacing::try_new(1.0, 1.0).expect("unit spacing"),
            2,
            2,
        )
        .expect("valid square canvas geometry");
        assert_eq!(square.attribute_value(), "1");
    }

    #[test]
    fn each_axis_selects_its_row_and_column_spacing() {
        assert_close(ratio([3.0, 5.0], [10, 4]), 50.0 / 12.0);
        assert_close(ratio([2.0, 5.0], [10, 4]), 50.0 / 8.0);
        assert_close(ratio([2.0, 3.0], [10, 4]), 30.0 / 8.0);
    }

    #[test]
    fn common_spacing_scale_cannot_change_aspect() {
        let base = [2.0, 5.0];
        let expected = ratio(base, [512, 94]);
        for exponent in [-1000, -30, 0, 1000] {
            let scale = 2.0_f64.powi(exponent);
            assert_close(ratio(base.map(|value| value * scale), [512, 94]), expected);
        }
    }

    #[test]
    fn invalid_dimensions_are_rejected() {
        let spacing = PresentationSpacing::try_new(1.0, 1.0).expect("unit spacing");
        assert_invalid(PhysicalCanvasAspect::from_display_spacing(spacing, 0, 2));
        assert_invalid(PhysicalCanvasAspect::from_display_spacing(spacing, 2, 0));
    }

    #[test]
    fn display_viewport_rejects_invalid_frame_and_css_dimensions() {
        for frame_size in [[0, 4], [8, 0]] {
            assert_eq!(
                viewport_for_display(0, [80.0, 40.0], frame_size)
                    .expect_err("empty backing frame must be rejected")
                    .kind(),
                io::ErrorKind::InvalidInput
            );
        }
        for display_size in [
            [0.0, 40.0],
            [80.0, -1.0],
            [f64::NAN, 40.0],
            [80.0, f64::INFINITY],
        ] {
            assert_eq!(
                viewport_for_display(0, display_size, [8, 4])
                    .expect_err("invalid CSS dimensions must be rejected")
                    .kind(),
                io::ErrorKind::InvalidInput
            );
        }
    }

    #[test]
    fn transformed_display_viewport_rejects_invalid_zoom_and_pan() {
        for (zoom, pan) in [
            (0.0, ViewportOffset::new(0.0, 0.0)),
            (f32::NAN, ViewportOffset::new(0.0, 0.0)),
            (1.0, ViewportOffset::new(f32::INFINITY, 0.0)),
        ] {
            assert_eq!(
                viewport_for_display_with_zoom_pan(0, [1.0; 2], [8, 4], zoom, pan)
                    .expect_err("invalid transformed viewport must be rejected")
                    .kind(),
                io::ErrorKind::InvalidInput
            );
        }
    }
}
