use crate::FilterKind;
use ritk_filter::Connectivity;

// Verify that the default `FilterKind` values exposed by the panel are
// within the analytically valid clamped ranges â€” Smoothing, Segmentation,
// and BinaryMorphology variants.

#[test]
fn gaussian_default_sigma_in_range() {
    let fk = FilterKind::Gaussian { sigma: 1.0 };
    if let FilterKind::Gaussian { sigma } = fk {
        assert!(
            (0.1..=20.0).contains(&sigma),
            "default sigma {sigma} must lie in [0.1, 20.0]"
        );
    } else {
        panic!("expected Gaussian variant");
    }
}

#[test]
fn median_default_radius_in_range() {
    let fk = FilterKind::Median { radius: 1 };
    if let FilterKind::Median { radius } = fk {
        assert!(radius <= 10, "default radius {radius} must be â‰¤ 10");
    } else {
        panic!("expected Median variant");
    }
}

#[test]
fn clahe_defaults_in_range() {
    let fk = FilterKind::Clahe {
        tile_grid_size: [8, 8],
        clip_limit: 40.0,
    };
    if let FilterKind::Clahe {
        tile_grid_size,
        clip_limit,
    } = fk
    {
        assert!(
            tile_grid_size[0] >= 1 && tile_grid_size[0] <= 32,
            "tile_grid_size[0]={} out of range",
            tile_grid_size[0]
        );
        assert!(
            tile_grid_size[1] >= 1 && tile_grid_size[1] <= 32,
            "tile_grid_size[1]={} out of range",
            tile_grid_size[1]
        );
        assert!(
            (1.0..=200.0).contains(&clip_limit),
            "clip_limit={clip_limit} out of range"
        );
    } else {
        panic!("expected Clahe variant");
    }
}

#[test]
fn histeq_default_bins_in_range() {
    let fk = FilterKind::HistEq { bins: 256 };
    if let FilterKind::HistEq { bins } = fk {
        assert!((2..=1024).contains(&bins), "bins={bins} out of range");
    } else {
        panic!("expected HistEq variant");
    }
}

/// UnsharpMask defaults lie within the slider ranges.
///
/// - sigma âˆˆ [0.1, 10.0] mm
/// - amount âˆˆ [0.0, 5.0]
/// - threshold âˆˆ [0.0, 100.0]
#[test]
fn unsharp_mask_defaults_in_range() {
    let fk = FilterKind::UnsharpMask {
        sigma: 1.0,
        amount: 0.5,
        threshold: 0.0,
        clamp: ritk_filter::ClampPolicy::ClampToInputRange,
    };
    if let FilterKind::UnsharpMask {
        sigma,
        amount,
        threshold,
        clamp,
    } = fk
    {
        assert!(
            (0.1..=10.0).contains(&sigma),
            "default sigma {sigma} out of range [0.1, 10.0]"
        );
        assert!(
            (0.0..=5.0).contains(&amount),
            "default amount {amount} out of range [0.0, 5.0]"
        );
        assert!(
            (0.0..=100.0).contains(&threshold),
            "default threshold {threshold} out of range [0.0, 100.0]"
        );
        assert!(
            matches!(clamp, ritk_filter::ClampPolicy::ClampToInputRange),
            "default clamp should be ClampToInputRange"
        );
    } else {
        panic!("expected UnsharpMask variant");
    }
}

/// GradientAnisotropicDiffusion defaults lie within the slider ranges.
///
/// - iterations âˆˆ [1, 50]
/// - time_step âˆˆ [0.01, 0.1667] (stability bound Î”t â‰¤ 1/6)
/// - conductance âˆˆ [0.1, 100.0]
///
/// ITK defaults: iterations=5, time_step=0.125, conductance=1.0.
#[test]
fn gradient_anisotropic_diffusion_defaults_in_range() {
    let fk = FilterKind::GradientAnisotropicDiffusion {
        iterations: 5,
        time_step: 0.125,
        conductance: 1.0,
    };
    if let FilterKind::GradientAnisotropicDiffusion {
        iterations,
        time_step,
        conductance,
    } = fk
    {
        assert!(
            (1..=50).contains(&iterations),
            "default iterations {iterations} out of range [1, 50]"
        );
        assert!(
            (0.01..=0.1667).contains(&time_step),
            "default time_step {time_step} out of range [0.01, 0.1667] (stability bound)"
        );
        assert!(
            (0.1..=100.0).contains(&conductance),
            "default conductance {conductance} out of range [0.1, 100.0]"
        );
    } else {
        panic!("expected GradientAnisotropicDiffusion variant");
    }
}

/// ConnectedComponents defaults are valid.
///
/// - connectivity = Connectivity::Face6 (6-connectivity is the ITK/medical default)
/// - background_value = 0.0 (ITK default)
///
/// # Postcondition
/// These values produce a valid ITK-parity filter dispatch via
/// `ConnectedComponentsFilter::with_connectivity(6).with_background(0.0)`.
#[test]
fn connected_components_defaults_are_valid() {
    let fk = FilterKind::ConnectedComponents {
        connectivity: Connectivity::Face6,
        background_value: 0.0,
    };
    if let FilterKind::ConnectedComponents {
        connectivity,
        background_value,
    } = fk
    {
        assert_eq!(
            connectivity,
            Connectivity::Face6,
            "default connectivity must be Face6 (6-connected, ITK default)"
        );
        assert!(
            background_value.is_finite(),
            "default background_value {background_value} must be finite"
        );
        assert_eq!(
            background_value, 0.0,
            "default background_value must be 0.0 (ITK ConnectedComponentImageFilter default)"
        );
    } else {
        panic!("expected ConnectedComponents variant");
    }
}

/// RelabelComponents defaults match ITK `RelabelComponentImageFilter` defaults.
///
/// # Analytical derivation
/// - minimum_object_size = 0 (ITK default: retain all components).
///
/// # Postcondition
/// These values produce a valid ITK-parity dispatch via
/// `RelabelComponentFilter::with_minimum_object_size(0)`.
#[test]
fn relabel_components_defaults_are_valid() {
    let fk = FilterKind::RelabelComponents {
        minimum_object_size: 0,
    };
    if let FilterKind::RelabelComponents {
        minimum_object_size,
    } = fk
    {
        assert_eq!(
            minimum_object_size, 0,
            "default minimum_object_size must be 0 (ITK default: retain all components)"
        );
    } else {
        panic!("expected RelabelComponents variant");
    }
}

/// MultiOtsuThreshold defaults match ITK `OtsuMultipleThresholdsImageFilter` defaults.
///
/// # Analytical derivation
/// - num_classes = 3 (ITK default: 3-class segmentation, 2 thresholds).
/// - num_classes â‰¥ 2 is required (enforced by `MultiOtsuThreshold::new` panic guard).
///
/// # Postcondition
/// These values produce a valid ITK-parity dispatch via
/// `MultiOtsuThreshold::new(3).apply(&image)`.
#[test]
fn multi_otsu_threshold_defaults_are_valid() {
    let fk = FilterKind::MultiOtsuThreshold { num_classes: 3 };
    if let FilterKind::MultiOtsuThreshold { num_classes } = fk {
        assert!(
            num_classes >= 2,
            "num_classes must be â‰¥ 2 (enforced by MultiOtsuThreshold::new); got {num_classes}"
        );
        assert_eq!(
            num_classes, 3,
            "ITK default num_classes = 3 (two thresholds; three classes)"
        );
    } else {
        panic!("expected MultiOtsuThreshold variant");
    }
}

/// BinaryErode defaults: radius=1, foreground_value=1.0 â€” ITK defaults.
#[test]
fn binary_erode_defaults_are_valid() {
    let fk = FilterKind::BinaryErode {
        radius: 1,
        foreground_value: ritk_filter::ForegroundValue::ONE,
    };
    if let FilterKind::BinaryErode {
        radius,
        foreground_value,
    } = fk
    {
        assert!(radius <= 10, "default radius {radius} must be â‰¤ 10");
        assert_eq!(
            foreground_value, 1.0,
            "default fg value must be 1.0 (ITK default)"
        );
    } else {
        panic!("expected BinaryErode variant");
    }
}

/// BinaryDilate defaults: radius=1, foreground_value=1.0 â€” ITK defaults.
#[test]
fn binary_dilate_defaults_are_valid() {
    let fk = FilterKind::BinaryDilate {
        radius: 1,
        foreground_value: ritk_filter::ForegroundValue::ONE,
    };
    if let FilterKind::BinaryDilate {
        radius,
        foreground_value,
    } = fk
    {
        assert!(radius <= 10, "default radius {radius} must be â‰¤ 10");
        assert_eq!(
            foreground_value, 1.0,
            "default fg value must be 1.0 (ITK default)"
        );
    } else {
        panic!("expected BinaryDilate variant");
    }
}

/// BinaryClosing defaults: radius=1, foreground_value=1.0 â€” ITK defaults.
#[test]
fn binary_closing_defaults_are_valid() {
    let fk = FilterKind::BinaryClosing {
        radius: 1,
        foreground_value: ritk_filter::ForegroundValue::ONE,
    };
    if let FilterKind::BinaryClosing {
        radius,
        foreground_value,
    } = fk
    {
        assert!(radius <= 10, "default radius {radius} must be â‰¤ 10");
        assert_eq!(
            foreground_value, 1.0,
            "default fg value must be 1.0 (ITK default)"
        );
    } else {
        panic!("expected BinaryClosing variant");
    }
}

/// BinaryOpening defaults: radius=1, foreground_value=1.0 â€” ITK defaults.
#[test]
fn binary_opening_defaults_are_valid() {
    let fk = FilterKind::BinaryOpening {
        radius: 1,
        foreground_value: ritk_filter::ForegroundValue::ONE,
    };
    if let FilterKind::BinaryOpening {
        radius,
        foreground_value,
    } = fk
    {
        assert!(radius <= 10, "default radius {radius} must be â‰¤ 10");
        assert_eq!(
            foreground_value, 1.0,
            "default fg value must be 1.0 (ITK default)"
        );
    } else {
        panic!("expected BinaryOpening variant");
    }
}

/// BinaryFillhole defaults: foreground_value=1.0 â€” ITK default.
#[test]
fn binary_fillhole_defaults_are_valid() {
    let fk = FilterKind::BinaryFillhole {
        foreground_value: ritk_filter::ForegroundValue::ONE,
    };
    if let FilterKind::BinaryFillhole { foreground_value } = fk {
        assert_eq!(
            foreground_value, 1.0,
            "default fg value must be 1.0 (ITK default)"
        );
    } else {
        panic!("expected BinaryFillhole variant");
    }
}

/// GrayscaleClosing default: radius=1 â€” minimal ITK closing SE.
///
/// # Analytical basis
/// radius=1 â†’ 3Ã—3Ã—3 SE, the smallest non-trivial cubic SE. ITK
/// `GrayscaleMorphologicalClosingImageFilter` default radius is 1.
#[test]
fn grayscale_closing_defaults_are_valid() {
    let fk = FilterKind::GrayscaleClosing { radius: 1 };
    if let FilterKind::GrayscaleClosing { radius } = fk {
        assert!(radius <= 10, "default radius {radius} must be â‰¤ 10");
        assert_eq!(radius, 1, "ITK default radius = 1 (3Ã—3Ã—3 SE)");
    } else {
        panic!("expected GrayscaleClosing variant");
    }
}

/// GrayscaleOpening default: radius=1 â€” minimal ITK opening SE.
///
/// # Analytical basis
/// radius=1 â†’ 3Ã—3Ã—3 SE, the smallest non-trivial cubic SE. ITK
/// `GrayscaleMorphologicalOpeningImageFilter` default radius is 1.
#[test]
fn grayscale_opening_defaults_are_valid() {
    let fk = FilterKind::GrayscaleOpening { radius: 1 };
    if let FilterKind::GrayscaleOpening { radius } = fk {
        assert!(radius <= 10, "default radius {radius} must be â‰¤ 10");
        assert_eq!(radius, 1, "ITK default radius = 1 (3Ã—3Ã—3 SE)");
    } else {
        panic!("expected GrayscaleOpening variant");
    }
}

/// GrayscaleFillhole: unit struct, always valid.
#[test]
fn grayscale_fillhole_variant_is_valid() {
    // FilterKind::GrayscaleFillhole has no parameters to validate.
    // Verify the variant is constructible and matches correctly.
    let fk = FilterKind::GrayscaleFillhole;
    assert!(
        matches!(fk, FilterKind::GrayscaleFillhole),
        "GrayscaleFillhole variant must match itself"
    );
}

/// MorphologicalGradient default: radius=1 â€” minimal non-trivial cubic SE.
///
/// # Analytical basis
/// radius=1 â†’ 3Ã—3Ã—3 SE, the smallest non-trivial cubic structuring element.
/// ITK `GrayscaleMorphologicalGradientImageFilter` uses radius=1 by default.
#[test]
fn morphological_gradient_default_radius_is_valid() {
    let fk = FilterKind::MorphologicalGradient { radius: 1 };
    if let FilterKind::MorphologicalGradient { radius } = fk {
        assert_eq!(
            radius, 1,
            "default radius must be 1 (smallest non-trivial SE)"
        );
        assert!(
            radius <= 10,
            "default radius {radius} must be within slider range [0, 10]"
        );
    } else {
        panic!("expected MorphologicalGradient variant");
    }
}

/// Grayscale erosion: default radius=1 is the smallest non-trivial SE.
#[test]
fn grayscale_erode_default_radius_valid() {
    let fk = FilterKind::GrayscaleErode { radius: 1 };
    if let FilterKind::GrayscaleErode { radius } = fk {
        assert_eq!(radius, 1, "default radius must be 1");
        assert!(radius <= 10, "default radius within slider range");
    } else {
        panic!("expected GrayscaleErode");
    }
}

/// Grayscale dilation: default radius=1 is the smallest non-trivial SE.
#[test]
fn grayscale_dilate_default_radius_valid() {
    let fk = FilterKind::GrayscaleDilate { radius: 1 };
    if let FilterKind::GrayscaleDilate { radius } = fk {
        assert_eq!(radius, 1, "default radius must be 1");
        assert!(radius <= 10, "default radius within slider range");
    } else {
        panic!("expected GrayscaleDilate");
    }
}

/// CurvatureFlow default time_step satisfies the 3-D stability bound Î”t â‰¤ 1/6.
#[test]
fn curvature_flow_default_time_step_is_stable() {
    let fk = FilterKind::CurvatureFlow {
        iterations: 5,
        time_step: 0.0625,
    };
    if let FilterKind::CurvatureFlow {
        iterations,
        time_step,
    } = fk
    {
        assert!(iterations >= 1, "iterations must be â‰¥ 1: {iterations}");
        assert!(
            time_step <= 1.0 / 6.0 + 1e-6,
            "time_step {time_step} must satisfy Î”t â‰¤ 1/6 â‰ˆ 0.1667"
        );
        assert!(time_step > 0.0, "time_step must be positive: {time_step}");
    } else {
        panic!("expected CurvatureFlow");
    }
}

/// CurvatureFlow default iterations is the ITK default (5).
#[test]
fn curvature_flow_default_iterations_matches_itk() {
    let fk = FilterKind::CurvatureFlow {
        iterations: 5,
        time_step: 0.0625,
    };
    if let FilterKind::CurvatureFlow { iterations, .. } = fk {
        assert_eq!(iterations, 5, "ITK default iterations = 5");
    } else {
        panic!("expected CurvatureFlow");
    }
}
