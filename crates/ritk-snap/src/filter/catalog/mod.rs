//! Selectable image filters with their display labels and starting parameters.
//!
//! Starting parameters follow the ITK defaults where one exists. Valid ranges
//! for edits return with the native filter panel.

use super::FilterKind;
use ritk_filter::ForegroundValue;
use ritk_filter::{BinarizationThreshold, Connectivity};

/// Every selectable filter as `(label, starting parameters)`, in display order.
pub(crate) fn filter_catalog() -> Vec<(&'static str, FilterKind)> {
    vec![
        ("Gaussian", FilterKind::Gaussian { sigma: 1.0 }),
        ("Median", FilterKind::Median { radius: 1 }),
        (
            "CLAHE",
            FilterKind::Clahe {
                tile_grid_size: [8, 8],
                clip_limit: 40.0,
            },
        ),
        ("Hist Equalize", FilterKind::HistEq { bins: 256 }),
        (
            "Unsharp Mask",
            FilterKind::UnsharpMask {
                sigma: 1.0,
                amount: 0.5,
                threshold: 0.0,
                clamp: ritk_filter::ClampPolicy::ClampToInputRange,
            },
        ),
        (
            "Gradient Aniso. Diffusion",
            FilterKind::GradientAnisotropicDiffusion {
                iterations: 5,
                time_step: 0.125,
                conductance: 1.0,
            },
        ),
        (
            "Connected Components",
            FilterKind::ConnectedComponents {
                connectivity: ritk_filter::Connectivity::Face6,
                background_value: 0.0,
            },
        ),
        (
            "Relabel Components",
            FilterKind::RelabelComponents {
                minimum_object_size: 0,
            },
        ),
        (
            "Multi-Otsu Threshold",
            FilterKind::MultiOtsuThreshold { num_classes: 3 },
        ),
        (
            "Binary Erode",
            FilterKind::BinaryErode {
                radius: 1,
                foreground_value: ritk_filter::ForegroundValue::ONE,
            },
        ),
        (
            "Binary Dilate",
            FilterKind::BinaryDilate {
                radius: 1,
                foreground_value: ritk_filter::ForegroundValue::ONE,
            },
        ),
        (
            "Binary Closing",
            FilterKind::BinaryClosing {
                radius: 1,
                foreground_value: ritk_filter::ForegroundValue::ONE,
            },
        ),
        (
            "Binary Opening",
            FilterKind::BinaryOpening {
                radius: 1,
                foreground_value: ritk_filter::ForegroundValue::ONE,
            },
        ),
        (
            "Binary Fill Holes",
            FilterKind::BinaryFillhole {
                foreground_value: ritk_filter::ForegroundValue::ONE,
            },
        ),
        (
            "Grayscale Closing",
            FilterKind::GrayscaleClosing { radius: 1 },
        ),
        (
            "Grayscale Opening",
            FilterKind::GrayscaleOpening { radius: 1 },
        ),
        ("Grayscale Fill Holes", FilterKind::GrayscaleFillhole),
        ("Abs", FilterKind::Abs),
        (
            "Invert Intensity",
            FilterKind::InvertIntensity { maximum: None },
        ),
        ("Normalize", FilterKind::NormalizeIntensity),
        ("Square", FilterKind::Square),
        ("Sqrt", FilterKind::Sqrt),
        ("Log", FilterKind::Log),
        ("Exp", FilterKind::Exp),
        (
            "Morphological Gradient",
            FilterKind::MorphologicalGradient { radius: 1 },
        ),
        (
            "Distance Transform",
            FilterKind::DistanceTransform {
                threshold: BinarizationThreshold::DEFAULT,
            },
        ),
        (
            "Signed Distance Transform",
            FilterKind::SignedDistanceTransform {
                threshold: BinarizationThreshold::DEFAULT,
            },
        ),
        ("Flip Z", FilterKind::FlipZ),
        ("Flip Y", FilterKind::FlipY),
        ("Flip X", FilterKind::FlipX),
        (
            "Mask Threshold",
            FilterKind::MaskThreshold {
                threshold: BinarizationThreshold::DEFAULT,
            },
        ),
        ("Geodesic Dilation (self)", FilterKind::GeodesicDilationSelf),
        ("Geodesic Erosion (self)", FilterKind::GeodesicErosionSelf),
        (
            "Shift Scale",
            FilterKind::ShiftScale {
                shift: 0.0,
                scale: 1.0,
            },
        ),
        (
            "Zero Crossing",
            FilterKind::ZeroCrossing {
                foreground_value: ritk_filter::ForegroundValue::ONE,
                background_value: 0.0,
            },
        ),
        (
            "Region Of Interest",
            FilterKind::RegionOfInterest {
                start_z: 0,
                start_y: 0,
                start_x: 0,
                size_z: 1,
                size_y: 1,
                size_x: 1,
            },
        ),
        (
            "Permute Axes",
            FilterKind::PermuteAxes {
                order_0: 0,
                order_1: 1,
                order_2: 2,
            },
        ),
        ("Mean", FilterKind::Mean { radius: 1 }),
        (
            "Binary Contour",
            FilterKind::BinaryContour {
                connectivity: Connectivity::Face6,
                foreground_value: ritk_filter::ForegroundValue::ONE,
            },
        ),
        (
            "Label Contour",
            FilterKind::LabelContour {
                connectivity: Connectivity::Face6,
                background_value: 0.0,
            },
        ),
        (
            "Voting Binary",
            FilterKind::VotingBinary {
                radius: 1,
                birth_threshold: 1,
                survival_threshold: 1,
                foreground_value: ritk_filter::ForegroundValue::ONE,
                background_value: 0.0,
            },
        ),
        (
            "Shrink",
            FilterKind::Shrink {
                factor_z: 2,
                factor_y: 2,
                factor_x: 2,
            },
        ),
        (
            "Constant Pad",
            FilterKind::ConstantPad {
                pad_lower_z: 1,
                pad_lower_y: 1,
                pad_lower_x: 1,
                pad_upper_z: 1,
                pad_upper_y: 1,
                pad_upper_x: 1,
                constant: 0.0,
            },
        ),
        (
            "Mirror Pad",
            FilterKind::MirrorPad {
                pad_lower_z: 1,
                pad_lower_y: 1,
                pad_lower_x: 1,
                pad_upper_z: 1,
                pad_upper_y: 1,
                pad_upper_x: 1,
            },
        ),
        (
            "Wrap Pad",
            FilterKind::WrapPad {
                pad_lower_z: 1,
                pad_lower_y: 1,
                pad_lower_x: 1,
                pad_upper_z: 1,
                pad_upper_y: 1,
                pad_upper_x: 1,
            },
        ),
        ("Grayscale Erode", FilterKind::GrayscaleErode { radius: 1 }),
        (
            "Grayscale Dilate",
            FilterKind::GrayscaleDilate { radius: 1 },
        ),
        (
            "Binary Threshold",
            FilterKind::BinaryThreshold {
                lower: 100.0,
                upper: 500.0,
                foreground: ForegroundValue::ONE,
                background: 0.0,
            },
        ),
        (
            "Rescale Intensity",
            FilterKind::RescaleIntensity {
                out_min: 0.0,
                out_max: 1.0,
            },
        ),
        (
            "Clamp",
            FilterKind::Clamp {
                lower: 0.0,
                upper: 255.0,
            },
        ),
        (
            "Connected Threshold",
            FilterKind::ConnectedThreshold {
                seed_z: 0,
                seed_y: 0,
                seed_x: 0,
                lower: 100.0,
                upper: 500.0,
            },
        ),
        (
            "Confidence Connected",
            FilterKind::ConfidenceConnected {
                seed_z: 0,
                seed_y: 0,
                seed_x: 0,
                initial_lower: 0.0,
                initial_upper: 100.0,
                multiplier: 2.5,
                max_iterations: 15,
            },
        ),
        (
            "Neighborhood Connected",
            FilterKind::NeighborhoodConnected {
                seed_z: 0,
                seed_y: 0,
                seed_x: 0,
                lower: 100.0,
                upper: 500.0,
                radius_z: 1,
                radius_y: 1,
                radius_x: 1,
            },
        ),
        ("Atan", FilterKind::Atan),
        ("Sin", FilterKind::Sin),
        ("Cos", FilterKind::Cos),
        ("Tan", FilterKind::Tan),
        ("Asin", FilterKind::Asin),
        ("Acos", FilterKind::Acos),
        ("Bounded Reciprocal", FilterKind::BoundedReciprocal),
        (
            "Curvature Flow",
            FilterKind::CurvatureFlow {
                iterations: 5,
                time_step: 0.0625,
            },
        ),
        (
            "CPR",
            FilterKind::Cpr {
                control_points: vec![[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
                num_path_samples: 256,
                cross_section_half_width: 10.0,
                num_cross_samples: 64,
            },
        ),
    ]
}

#[cfg(test)]
mod tests_integrity;
#[cfg(test)]
mod tests_smoothing;

#[cfg(test)]
fn catalog_default(label: &str) -> FilterKind {
    filter_catalog()
        .into_iter()
        .find_map(|(entry, kind)| (entry == label).then_some(kind))
        .expect("label is in the filter catalog")
}
