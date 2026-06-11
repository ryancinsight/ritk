use crate::FilterKind;
use ritk_core::filter::Connectivity;

/// Second portion of the ComboBox selectable_value entries (FlipZ
/// through ConstantPad).
pub fn show_second_half(ui: &mut egui::Ui, active_filter: &mut FilterKind) {
    if ui
        .selectable_value(&mut *active_filter, FilterKind::FlipZ, "Flip Z")
        .clicked()
    {
        *active_filter = FilterKind::FlipZ;
    }
    if ui
        .selectable_value(&mut *active_filter, FilterKind::FlipY, "Flip Y")
        .clicked()
    {
        *active_filter = FilterKind::FlipY;
    }
    if ui
        .selectable_value(&mut *active_filter, FilterKind::FlipX, "Flip X")
        .clicked()
    {
        *active_filter = FilterKind::FlipX;
    }
    if ui
        .selectable_value(
            &mut *active_filter,
            FilterKind::MaskThreshold {
                threshold: 0.5.into(),
            },
            "Mask Threshold",
        )
        .clicked()
    {
        *active_filter = FilterKind::MaskThreshold {
            threshold: 0.5.into(),
        };
    }
    if ui
        .selectable_value(
            &mut *active_filter,
            FilterKind::GeodesicDilationSelf,
            "Geodesic Dilation (self)",
        )
        .clicked()
    {
        *active_filter = FilterKind::GeodesicDilationSelf;
    }
    if ui
        .selectable_value(
            &mut *active_filter,
            FilterKind::GeodesicErosionSelf,
            "Geodesic Erosion (self)",
        )
        .clicked()
    {
        *active_filter = FilterKind::GeodesicErosionSelf;
    }
    if ui
        .selectable_value(
            &mut *active_filter,
            FilterKind::ShiftScale {
                shift: 0.0,
                scale: 1.0,
            },
            "Shift Scale",
        )
        .clicked()
    {
        *active_filter = FilterKind::ShiftScale {
            shift: 0.0,
            scale: 1.0,
        };
    }
    if ui
        .selectable_value(
            &mut *active_filter,
            FilterKind::ZeroCrossing {
                foreground_value: ritk_core::filter::ForegroundValue::ONE,
                background_value: 0.0,
            },
            "Zero Crossing",
        )
        .clicked()
    {
        *active_filter = FilterKind::ZeroCrossing {
            foreground_value: ritk_core::filter::ForegroundValue::ONE,
            background_value: 0.0,
        };
    }
    if ui
        .selectable_value(
            &mut *active_filter,
            FilterKind::RegionOfInterest {
                start_z: 0,
                start_y: 0,
                start_x: 0,
                size_z: 1,
                size_y: 1,
                size_x: 1,
            },
            "Region Of Interest",
        )
        .clicked()
    {
        *active_filter = FilterKind::RegionOfInterest {
            start_z: 0,
            start_y: 0,
            start_x: 0,
            size_z: 1,
            size_y: 1,
            size_x: 1,
        };
    }
    if ui
        .selectable_value(
            &mut *active_filter,
            FilterKind::PermuteAxes {
                order_0: 0,
                order_1: 1,
                order_2: 2,
            },
            "Permute Axes",
        )
        .clicked()
    {
        *active_filter = FilterKind::PermuteAxes {
            order_0: 0,
            order_1: 1,
            order_2: 2,
        };
    }
    if ui
        .selectable_value(&mut *active_filter, FilterKind::Mean { radius: 1 }, "Mean")
        .clicked()
    {
        *active_filter = FilterKind::Mean { radius: 1 };
    }
    if ui
        .selectable_value(
            &mut *active_filter,
            FilterKind::BinaryContour {
                connectivity: Connectivity::Face6,
                foreground_value: ritk_core::filter::ForegroundValue::ONE,
            },
            "Binary Contour",
        )
        .clicked()
    {
        *active_filter = FilterKind::BinaryContour {
            connectivity: Connectivity::Face6,
            foreground_value: ritk_core::filter::ForegroundValue::ONE,
        };
    }
    if ui
        .selectable_value(
            &mut *active_filter,
            FilterKind::LabelContour {
                connectivity: Connectivity::Face6,
                background_value: 0.0,
            },
            "Label Contour",
        )
        .clicked()
    {
        *active_filter = FilterKind::LabelContour {
            connectivity: Connectivity::Face6,
            background_value: 0.0,
        };
    }
    if ui
        .selectable_value(
            &mut *active_filter,
            FilterKind::VotingBinary {
                radius: 1,
                birth_threshold: 1,
                survival_threshold: 1,
                foreground_value: ritk_core::filter::ForegroundValue::ONE,
                background_value: 0.0,
            },
            "Voting Binary",
        )
        .clicked()
    {
        *active_filter = FilterKind::VotingBinary {
            radius: 1,
            birth_threshold: 1,
            survival_threshold: 1,
            foreground_value: ritk_core::filter::ForegroundValue::ONE,
            background_value: 0.0,
        };
    }
    if ui
        .selectable_value(
            &mut *active_filter,
            FilterKind::Shrink {
                factor_z: 2,
                factor_y: 2,
                factor_x: 2,
            },
            "Shrink",
        )
        .clicked()
    {
        *active_filter = FilterKind::Shrink {
            factor_z: 2,
            factor_y: 2,
            factor_x: 2,
        };
    }
    if ui
        .selectable_value(
            &mut *active_filter,
            FilterKind::ConstantPad {
                pad_lower_z: 1,
                pad_lower_y: 1,
                pad_lower_x: 1,
                pad_upper_z: 1,
                pad_upper_y: 1,
                pad_upper_x: 1,
                constant: 0.0,
            },
            "Constant Pad",
        )
        .clicked()
    {
        *active_filter = FilterKind::ConstantPad {
            pad_lower_z: 1,
            pad_lower_y: 1,
            pad_lower_x: 1,
            pad_upper_z: 1,
            pad_upper_y: 1,
            pad_upper_x: 1,
            constant: 0.0,
        };
    }

    super::selector_values_third::show_third_half(ui, active_filter);
}
