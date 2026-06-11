use crate::FilterKind;
use ritk_filter::BinarizationThreshold;

/// First half of the ComboBox selectable_value entries (Gaussian through
/// MorphologicalGradient).
pub fn show_first_half(ui: &mut egui::Ui, active_filter: &mut FilterKind) {
    if ui
        .selectable_value(
            &mut *active_filter,
            FilterKind::Gaussian { sigma: 1.0 },
            "Gaussian",
        )
        .clicked()
    {
        *active_filter = FilterKind::Gaussian { sigma: 1.0 };
    }
    if ui
        .selectable_value(
            &mut *active_filter,
            FilterKind::Median { radius: 1 },
            "Median",
        )
        .clicked()
    {
        *active_filter = FilterKind::Median { radius: 1 };
    }
    if ui
        .selectable_value(
            &mut *active_filter,
            FilterKind::Clahe {
                tile_grid_size: [8, 8],
                clip_limit: 40.0,
            },
            "CLAHE",
        )
        .clicked()
    {
        *active_filter = FilterKind::Clahe {
            tile_grid_size: [8, 8],
            clip_limit: 40.0,
        };
    }
    if ui
        .selectable_value(
            &mut *active_filter,
            FilterKind::HistEq { bins: 256 },
            "Hist Equalize",
        )
        .clicked()
    {
        *active_filter = FilterKind::HistEq { bins: 256 };
    }
    if ui
        .selectable_value(
            &mut *active_filter,
            FilterKind::UnsharpMask {
                sigma: 1.0,
                amount: 0.5,
                threshold: 0.0,
                clamp: ritk_filter::ClampPolicy::ClampToInputRange,
            },
            "Unsharp Mask",
        )
        .clicked()
    {
        *active_filter = FilterKind::UnsharpMask {
            sigma: 1.0,
            amount: 0.5,
            threshold: 0.0,
            clamp: ritk_filter::ClampPolicy::ClampToInputRange,
        };
    }
    if ui
        .selectable_value(
            &mut *active_filter,
            FilterKind::GradientAnisotropicDiffusion {
                iterations: 5,
                time_step: 0.125,
                conductance: 1.0,
            },
            "Gradient Aniso. Diffusion",
        )
        .clicked()
    {
        *active_filter = FilterKind::GradientAnisotropicDiffusion {
            iterations: 5,
            time_step: 0.125,
            conductance: 1.0,
        };
    }
    if ui
        .selectable_value(
            &mut *active_filter,
            FilterKind::ConnectedComponents {
                connectivity: ritk_filter::Connectivity::Face6,
                background_value: 0.0,
            },
            "Connected Components",
        )
        .clicked()
    {
        *active_filter = FilterKind::ConnectedComponents {
            connectivity: ritk_filter::Connectivity::Face6,
            background_value: 0.0,
        };
    }
    if ui
        .selectable_value(
            &mut *active_filter,
            FilterKind::RelabelComponents {
                minimum_object_size: 0,
            },
            "Relabel Components",
        )
        .clicked()
    {
        *active_filter = FilterKind::RelabelComponents {
            minimum_object_size: 0,
        };
    }
    if ui
        .selectable_value(
            &mut *active_filter,
            FilterKind::MultiOtsuThreshold { num_classes: 3 },
            "Multi-Otsu Threshold",
        )
        .clicked()
    {
        *active_filter = FilterKind::MultiOtsuThreshold { num_classes: 3 };
    }
    if ui
        .selectable_value(
            &mut *active_filter,
            FilterKind::BinaryErode {
                radius: 1,
                foreground_value: ritk_filter::ForegroundValue::ONE,
            },
            "Binary Erode",
        )
        .clicked()
    {
        *active_filter = FilterKind::BinaryErode {
            radius: 1,
            foreground_value: ritk_filter::ForegroundValue::ONE,
        };
    }
    if ui
        .selectable_value(
            &mut *active_filter,
            FilterKind::BinaryDilate {
                radius: 1,
                foreground_value: ritk_filter::ForegroundValue::ONE,
            },
            "Binary Dilate",
        )
        .clicked()
    {
        *active_filter = FilterKind::BinaryDilate {
            radius: 1,
            foreground_value: ritk_filter::ForegroundValue::ONE,
        };
    }
    if ui
        .selectable_value(
            &mut *active_filter,
            FilterKind::BinaryClosing {
                radius: 1,
                foreground_value: ritk_filter::ForegroundValue::ONE,
            },
            "Binary Closing",
        )
        .clicked()
    {
        *active_filter = FilterKind::BinaryClosing {
            radius: 1,
            foreground_value: ritk_filter::ForegroundValue::ONE,
        };
    }
    if ui
        .selectable_value(
            &mut *active_filter,
            FilterKind::BinaryOpening {
                radius: 1,
                foreground_value: ritk_filter::ForegroundValue::ONE,
            },
            "Binary Opening",
        )
        .clicked()
    {
        *active_filter = FilterKind::BinaryOpening {
            radius: 1,
            foreground_value: ritk_filter::ForegroundValue::ONE,
        };
    }
    if ui
        .selectable_value(
            &mut *active_filter,
            FilterKind::BinaryFillhole {
                foreground_value: ritk_filter::ForegroundValue::ONE,
            },
            "Binary Fill Holes",
        )
        .clicked()
    {
        *active_filter = FilterKind::BinaryFillhole {
            foreground_value: ritk_filter::ForegroundValue::ONE,
        };
    }
    if ui
        .selectable_value(
            &mut *active_filter,
            FilterKind::GrayscaleClosing { radius: 1 },
            "Grayscale Closing",
        )
        .clicked()
    {
        *active_filter = FilterKind::GrayscaleClosing { radius: 1 };
    }
    if ui
        .selectable_value(
            &mut *active_filter,
            FilterKind::GrayscaleOpening { radius: 1 },
            "Grayscale Opening",
        )
        .clicked()
    {
        *active_filter = FilterKind::GrayscaleOpening { radius: 1 };
    }
    if ui
        .selectable_value(
            &mut *active_filter,
            FilterKind::GrayscaleFillhole,
            "Grayscale Fill Holes",
        )
        .clicked()
    {
        *active_filter = FilterKind::GrayscaleFillhole;
    }
    if ui
        .selectable_value(&mut *active_filter, FilterKind::Abs, "Abs")
        .clicked()
    {
        *active_filter = FilterKind::Abs;
    }
    if ui
        .selectable_value(
            &mut *active_filter,
            FilterKind::InvertIntensity { maximum: None },
            "Invert Intensity",
        )
        .clicked()
    {
        *active_filter = FilterKind::InvertIntensity { maximum: None };
    }
    if ui
        .selectable_value(
            &mut *active_filter,
            FilterKind::NormalizeIntensity,
            "Normalize",
        )
        .clicked()
    {
        *active_filter = FilterKind::NormalizeIntensity;
    }
    if ui
        .selectable_value(&mut *active_filter, FilterKind::Square, "Square")
        .clicked()
    {
        *active_filter = FilterKind::Square;
    }
    if ui
        .selectable_value(&mut *active_filter, FilterKind::Sqrt, "Sqrt")
        .clicked()
    {
        *active_filter = FilterKind::Sqrt;
    }
    if ui
        .selectable_value(&mut *active_filter, FilterKind::Log, "Log")
        .clicked()
    {
        *active_filter = FilterKind::Log;
    }
    if ui
        .selectable_value(&mut *active_filter, FilterKind::Exp, "Exp")
        .clicked()
    {
        *active_filter = FilterKind::Exp;
    }
    if ui
        .selectable_value(
            &mut *active_filter,
            FilterKind::MorphologicalGradient { radius: 1 },
            "Morphological Gradient",
        )
        .clicked()
    {
        *active_filter = FilterKind::MorphologicalGradient { radius: 1 };
    }
    if ui
        .selectable_value(
            &mut *active_filter,
            FilterKind::DistanceTransform {
                threshold: BinarizationThreshold::DEFAULT,
            },
            "Distance Transform",
        )
        .clicked()
    {
        *active_filter = FilterKind::DistanceTransform {
            threshold: BinarizationThreshold::DEFAULT,
        };
    }
    if ui
        .selectable_value(
            &mut *active_filter,
            FilterKind::SignedDistanceTransform {
                threshold: BinarizationThreshold::DEFAULT,
            },
            "Signed Distance Transform",
        )
        .clicked()
    {
        *active_filter = FilterKind::SignedDistanceTransform {
            threshold: BinarizationThreshold::DEFAULT,
        };
    }
}
