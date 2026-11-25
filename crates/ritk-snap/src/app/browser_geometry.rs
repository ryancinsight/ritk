//! Physical geometry for RITK-owned browser canvases.

use super::viewer_viewport::ViewerViewport;
use crate::presentation::PresentationSpacing;
use crate::tools::interaction::ViewportOffset;
use crate::ui::ViewTransform;
use metis_web::DisplaySpacing;
use std::io;

/// A positive finite CSS width-to-height ratio for one rendered slice.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct PhysicalCanvasAspect(f64);

impl PhysicalCanvasAspect {
    /// Computes the aspect from frame-ordered row and column distances.
    ///
    /// RITK supplies frame-ordered spacing after admitting the DICOM geometry;
    /// Metis computes the format-neutral physical ratio.
    pub(super) fn from_display_spacing(
        display_spacing: PresentationSpacing,
        width: u32,
        height: u32,
    ) -> io::Result<Self> {
        let [row_spacing, column_spacing] = display_spacing.values();
        let ratio = DisplaySpacing::try_new(row_spacing, column_spacing)?.aspect(width, height)?;
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
    .map(|mut viewport| {
        viewport.display_units = [
            f64::from(frame_width) / display_size[0],
            f64::from(frame_height) / display_size[1],
        ];
        viewport
    })
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

    fn assert_same_value(actual: f64, expected: f64) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }

    fn assert_invalid(result: io::Result<PhysicalCanvasAspect>) {
        assert_eq!(
            result.expect_err("invalid physical canvas geometry").kind(),
            io::ErrorKind::InvalidInput
        );
    }

    #[test]
    fn frame_spacing_preserves_physical_extents() {
        assert_same_value(ratio([0.5, 0.5], [512, 512]), 1.0);
        assert_same_value(ratio([2.5, 0.5], [512, 94]), 256.0 / 235.0);
    }

    #[test]
    fn isotropic_spacing_preserves_pixel_aspect() {
        assert_same_value(ratio([1.0; 2], [512, 94]), 512.0 / 94.0);
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
        assert_same_value(ratio([3.0, 5.0], [10, 4]), 50.0 / 12.0);
        assert_same_value(ratio([2.0, 5.0], [10, 4]), 50.0 / 8.0);
        assert_same_value(ratio([2.0, 3.0], [10, 4]), 30.0 / 8.0);
    }

    #[test]
    fn common_spacing_scale_cannot_change_aspect() {
        let base = [2.0, 5.0];
        let expected = ratio(base, [512, 94]);
        for exponent in [-1000, -30, 0, 1000] {
            let scale = 2.0_f64.powi(exponent);
            assert_same_value(ratio(base.map(|value| value * scale), [512, 94]), expected);
        }
    }

    #[test]
    fn extreme_spacing_preserves_minimum_subnormal_aspect() {
        let row = f64::from_bits(0x7fe0_0000_0000_0000);
        let column = f64::from_bits(0x3cb0_0000_0000_0000);
        // (2 * 2^-52) / (1 * 2^1023) = 2^-1074 exactly.
        assert_eq!(ratio([row, column], [2, 1]).to_bits(), 1);
    }

    #[test]
    fn invalid_dimensions_are_rejected() {
        let spacing = PresentationSpacing::try_new(1.0, 1.0).expect("unit spacing");
        assert_invalid(PhysicalCanvasAspect::from_display_spacing(spacing, 0, 2));
        assert_invalid(PhysicalCanvasAspect::from_display_spacing(spacing, 2, 0));
    }

    #[test]
    fn unrepresentable_physical_aspects_are_invalid_input() {
        for spacing in [
            PresentationSpacing::try_new(f64::from_bits(1), f64::MAX)
                .expect("valid overflow-boundary spacing"),
            PresentationSpacing::try_new(f64::MAX, f64::from_bits(1))
                .expect("valid underflow-boundary spacing"),
        ] {
            assert_invalid(PhysicalCanvasAspect::from_display_spacing(spacing, 1, 1));
        }
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
