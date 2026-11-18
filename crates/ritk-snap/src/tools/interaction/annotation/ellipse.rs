use super::{Annotation, MeasurementError};

impl Annotation {
    /// Compute ellipse statistics after validating physical spacing and the
    /// f32-backed derived area.
    #[expect(
        clippy::type_complexity,
        reason = "the checked API mirrors the established ellipse statistics tuple"
    )]
    pub fn compute_roi_ellipse_stats_checked(
        p1: [f32; 2],
        p2: [f32; 2],
        pixels: &[f32],
        width: usize,
        height: usize,
        spacing: [f64; 2],
    ) -> Result<([f32; 2], [f32; 2], f32, f32, f32, f32, f32), MeasurementError> {
        let spacing = Self::validate_spacing(spacing)?;
        let result = Self::compute_roi_ellipse_stats(p1, p2, pixels, width, height, spacing);
        let finite = result
            .0
            .iter()
            .chain(result.1.iter())
            .all(|value| value.is_finite())
            && result.2.is_finite()
            && result.3.is_finite()
            && result.4.is_finite()
            && result.5.is_finite()
            && result.6.is_finite();
        finite
            .then_some(result)
            .ok_or(MeasurementError::NonFiniteResult {
                kind: "ellipse ROI",
            })
    }
}
