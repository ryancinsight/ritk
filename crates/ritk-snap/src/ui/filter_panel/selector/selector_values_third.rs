use crate::FilterKind;
use ritk_filter::ForegroundValue;

/// Third portion of the ComboBox selectable_value entries (MirrorPad
/// through CurvatureFlow).
pub fn show_third_half(ui: &mut egui::Ui, active_filter: &mut FilterKind) {
    if ui
        .selectable_value(
            &mut *active_filter,
            FilterKind::MirrorPad {
                pad_lower_z: 1,
                pad_lower_y: 1,
                pad_lower_x: 1,
                pad_upper_z: 1,
                pad_upper_y: 1,
                pad_upper_x: 1 },
            "Mirror Pad",
        )
        .clicked()
    {
        *active_filter = FilterKind::MirrorPad {
            pad_lower_z: 1,
            pad_lower_y: 1,
            pad_lower_x: 1,
            pad_upper_z: 1,
            pad_upper_y: 1,
            pad_upper_x: 1 };
    }
    if ui
        .selectable_value(
            &mut *active_filter,
            FilterKind::WrapPad {
                pad_lower_z: 1,
                pad_lower_y: 1,
                pad_lower_x: 1,
                pad_upper_z: 1,
                pad_upper_y: 1,
                pad_upper_x: 1 },
            "Wrap Pad",
        )
        .clicked()
    {
        *active_filter = FilterKind::WrapPad {
            pad_lower_z: 1,
            pad_lower_y: 1,
            pad_lower_x: 1,
            pad_upper_z: 1,
            pad_upper_y: 1,
            pad_upper_x: 1 };
    }
    if ui
        .selectable_value(
            &mut *active_filter,
            FilterKind::GrayscaleErode { radius: 1 },
            "Grayscale Erode",
        )
        .clicked()
    {
        *active_filter = FilterKind::GrayscaleErode { radius: 1 };
    }
    if ui
        .selectable_value(
            &mut *active_filter,
            FilterKind::GrayscaleDilate { radius: 1 },
            "Grayscale Dilate",
        )
        .clicked()
    {
        *active_filter = FilterKind::GrayscaleDilate { radius: 1 };
    }
    if ui
        .selectable_value(
            &mut *active_filter,
            FilterKind::BinaryThreshold {
                lower: 100.0,
                upper: 500.0,
                foreground: ForegroundValue::ONE,
                background: 0.0 },
            "Binary Threshold",
        )
        .clicked()
    {
        *active_filter = FilterKind::BinaryThreshold {
            lower: 100.0,
            upper: 500.0,
            foreground: ForegroundValue::ONE,
            background: 0.0 };
    }
    if ui
        .selectable_value(
            &mut *active_filter,
            FilterKind::RescaleIntensity {
                out_min: 0.0,
                out_max: 1.0 },
            "Rescale Intensity",
        )
        .clicked()
    {
        *active_filter = FilterKind::RescaleIntensity {
            out_min: 0.0,
            out_max: 1.0 };
    }
    if ui
        .selectable_value(
            &mut *active_filter,
            FilterKind::Clamp {
                lower: 0.0,
                upper: 255.0 },
            "Clamp",
        )
        .clicked()
    {
        *active_filter = FilterKind::Clamp {
            lower: 0.0,
            upper: 255.0 };
    }
    if ui
        .selectable_value(
            &mut *active_filter,
            FilterKind::ConnectedThreshold {
                seed_z: 0,
                seed_y: 0,
                seed_x: 0,
                lower: 100.0,
                upper: 500.0 },
            "Connected Threshold",
        )
        .clicked()
    {
        *active_filter = FilterKind::ConnectedThreshold {
            seed_z: 0,
            seed_y: 0,
            seed_x: 0,
            lower: 100.0,
            upper: 500.0 };
    }
    if ui
        .selectable_value(
            &mut *active_filter,
            FilterKind::ConfidenceConnected {
                seed_z: 0,
                seed_y: 0,
                seed_x: 0,
                initial_lower: 0.0,
                initial_upper: 100.0,
                multiplier: 2.5,
                max_iterations: 15 },
            "Confidence Connected",
        )
        .clicked()
    {
        *active_filter = FilterKind::ConfidenceConnected {
            seed_z: 0,
            seed_y: 0,
            seed_x: 0,
            initial_lower: 0.0,
            initial_upper: 100.0,
            multiplier: 2.5,
            max_iterations: 15 };
    }
    if ui
        .selectable_value(
            &mut *active_filter,
            FilterKind::NeighborhoodConnected {
                seed_z: 0,
                seed_y: 0,
                seed_x: 0,
                lower: 100.0,
                upper: 500.0,
                radius_z: 1,
                radius_y: 1,
                radius_x: 1 },
            "Neighborhood Connected",
        )
        .clicked()
    {
        *active_filter = FilterKind::NeighborhoodConnected {
            seed_z: 0,
            seed_y: 0,
            seed_x: 0,
            lower: 100.0,
            upper: 500.0,
            radius_z: 1,
            radius_y: 1,
            radius_x: 1 };
    }
    if ui
        .selectable_value(&mut *active_filter, FilterKind::Atan, "Atan")
        .clicked()
    {
        *active_filter = FilterKind::Atan;
    }
    if ui
        .selectable_value(&mut *active_filter, FilterKind::Sin, "Sin")
        .clicked()
    {
        *active_filter = FilterKind::Sin;
    }
    if ui
        .selectable_value(&mut *active_filter, FilterKind::Cos, "Cos")
        .clicked()
    {
        *active_filter = FilterKind::Cos;
    }
    if ui
        .selectable_value(&mut *active_filter, FilterKind::Tan, "Tan")
        .clicked()
    {
        *active_filter = FilterKind::Tan;
    }
    if ui
        .selectable_value(&mut *active_filter, FilterKind::Asin, "Asin")
        .clicked()
    {
        *active_filter = FilterKind::Asin;
    }
    if ui
        .selectable_value(&mut *active_filter, FilterKind::Acos, "Acos")
        .clicked()
    {
        *active_filter = FilterKind::Acos;
    }
    if ui
        .selectable_value(
            &mut *active_filter,
            FilterKind::BoundedReciprocal,
            "Bounded Reciprocal",
        )
        .clicked()
    {
        *active_filter = FilterKind::BoundedReciprocal;
    }
    if ui
        .selectable_value(
            &mut *active_filter,
            FilterKind::CurvatureFlow {
                iterations: 5,
                time_step: 0.0625 },
            "Curvature Flow",
        )
        .clicked()
    {
        *active_filter = FilterKind::CurvatureFlow {
            iterations: 5,
            time_step: 0.0625 };
    }
    if ui
        .selectable_value(
            &mut *active_filter,
            FilterKind::Cpr {
                control_points: vec![[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
                num_path_samples: 256,
                cross_section_half_width: 10.0,
                num_cross_samples: 64 },
            "CPR",
        )
        .clicked()
    {
        *active_filter = FilterKind::Cpr {
            control_points: vec![[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            num_path_samples: 256,
            cross_section_half_width: 10.0,
            num_cross_samples: 64 };
    }
}
