#[derive(Clone, Copy)]
pub(crate) struct ImageMeasurementScale {
    linear: f64,
}

impl ImageMeasurementScale {
    pub(crate) fn new(linear: f64) -> Self {
        Self { linear }
    }

    pub(crate) fn length(self, raw_mm: f32) -> f64 {
        f64::from(raw_mm) * self.linear
    }

    pub(crate) fn area(self, raw_mm2: f32) -> f64 {
        f64::from(raw_mm2) * self.linear * self.linear
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
