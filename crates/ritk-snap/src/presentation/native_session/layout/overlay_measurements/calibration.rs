#[derive(Clone, Copy)]
pub(super) struct ImageMeasurementScale {
    linear: f64,
}

impl ImageMeasurementScale {
    pub(super) fn new(linear: f64) -> Self {
        Self { linear }
    }

    fn length(self, raw_mm: f32) -> f64 {
        f64::from(raw_mm) * self.linear
    }

    fn area(self, raw_mm2: f32) -> f64 {
        f64::from(raw_mm2) * self.linear * self.linear
    }

    pub(super) fn length_label(self, raw_mm: f32) -> String {
        format!("{:.1} mm", self.length(raw_mm))
    }

    pub(super) fn deviation_label(self, raw_mm: f32, angle_deg: f32) -> String {
        format!("{:.1} mm {angle_deg:.1} deg", self.length(raw_mm))
    }

    pub(super) fn roi_label(self, mean: f32, std_dev: f32, raw_mm2: f32) -> String {
        format!(
            "Mean {mean:.1} SD {std_dev:.1} {:.1} mm2",
            self.area(raw_mm2)
        )
    }

    pub(super) fn closed_polygon_label(self, raw_mm2: f32, raw_mm: f32) -> String {
        format!(
            "{:.1} mm2  {:.1} mm",
            self.area(raw_mm2),
            self.length(raw_mm)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::ImageMeasurementScale;

    #[test]
    fn finite_annotation_extremes_fit_the_display_range() {
        let minimum = f32::from_bits(1);
        let maximum = f32::MAX;
        let upper =
            ImageMeasurementScale::new(f64::from(maximum) / f64::from(minimum)).area(maximum);
        let lower =
            ImageMeasurementScale::new(f64::from(minimum) / f64::from(maximum)).area(minimum);

        assert!(upper.is_finite());
        assert!(upper.log2() < 682.0);
        assert!(lower > 0.0);
        assert!(lower.log2() > -703.0);
    }
}
