use crate::tools::interaction::ImageMeasurementScale;

impl ImageMeasurementScale {
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
