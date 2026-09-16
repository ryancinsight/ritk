//! Physical geometry for RITK-owned browser canvases.

use super::action_adapter::ViewerViewport;
use crate::ui::ViewTransform;
use std::io;

/// A positive finite CSS width-to-height ratio for one rendered slice.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct PhysicalCanvasAspect(f64);

impl PhysicalCanvasAspect {
    /// Computes the physical slice aspect from `[dz, dy, dx]` sample spacing.
    ///
    /// Pixel and spacing factors are normalized independently before they are
    /// multiplied. This preserves common-unit scale invariance and prevents
    /// otherwise-valid large sample distances from overflowing intermediate
    /// physical extents.
    pub(super) fn new(spacing: [f64; 3], axis: usize, width: u32, height: u32) -> io::Result<Self> {
        if width == 0 || height == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "browser slice dimensions must be positive",
            ));
        }
        if !spacing
            .iter()
            .all(|distance| distance.is_finite() && *distance > 0.0)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "browser slice spacing must be positive and finite",
            ));
        }

        let [dz, dy, dx] = spacing;
        let [row_spacing, column_spacing] = match axis {
            0 => [dy, dx],
            1 => [dz, dx],
            2 => [dz, dy],
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("browser slice axis {axis} is outside 0..=2"),
                ));
            }
        };

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
pub(super) fn viewport_for_display(
    axis: usize,
    display_size: [f64; 2],
    frame_size: [u32; 2],
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
    ViewerViewport::new(
        axis,
        [0.0, 0.0],
        [
            display_size[0] / f64::from(frame_width),
            display_size[1] / f64::from(frame_height),
        ],
        [width, height],
        ViewTransform::default(),
    )
    .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::{viewport_for_display, PhysicalCanvasAspect};
    use std::io;

    fn ratio(spacing: [f64; 3], axis: usize, dimensions: [u32; 2]) -> f64 {
        PhysicalCanvasAspect::new(spacing, axis, dimensions[0], dimensions[1])
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
    fn anisotropic_mri_planes_use_physical_extents() {
        let spacing = [2.5, 0.5, 0.5];
        assert_close(ratio(spacing, 0, [512, 512]), 1.0);
        assert_close(ratio(spacing, 1, [512, 94]), 256.0 / 235.0);
        assert_close(ratio(spacing, 2, [512, 94]), 256.0 / 235.0);
    }

    #[test]
    fn isotropic_spacing_preserves_pixel_aspect() {
        assert_close(ratio([1.0; 3], 0, [512, 94]), 512.0 / 94.0);
        let square =
            PhysicalCanvasAspect::new([1.0; 3], 0, 2, 2).expect("valid square canvas geometry");
        assert_eq!(square.attribute_value(), "1");
    }

    #[test]
    fn each_axis_selects_its_row_and_column_spacing() {
        let spacing = [2.0, 3.0, 5.0];
        assert_close(ratio(spacing, 0, [10, 4]), 50.0 / 12.0);
        assert_close(ratio(spacing, 1, [10, 4]), 50.0 / 8.0);
        assert_close(ratio(spacing, 2, [10, 4]), 30.0 / 8.0);
    }

    #[test]
    fn common_spacing_scale_cannot_change_aspect() {
        let base = [2.0, 3.0, 5.0];
        let expected = ratio(base, 1, [512, 94]);
        for exponent in [-1000, -30, 0, 1000] {
            let scale = 2.0_f64.powi(exponent);
            assert_close(
                ratio(base.map(|value| value * scale), 1, [512, 94]),
                expected,
            );
        }
    }

    #[test]
    fn invalid_spacing_dimensions_and_axis_are_rejected() {
        for invalid in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert_invalid(PhysicalCanvasAspect::new([invalid, 1.0, 1.0], 0, 2, 2));
        }
        assert_invalid(PhysicalCanvasAspect::new([1.0; 3], 0, 0, 2));
        assert_invalid(PhysicalCanvasAspect::new([1.0; 3], 0, 2, 0));
        assert_invalid(PhysicalCanvasAspect::new([1.0; 3], 3, 2, 2));
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
}
