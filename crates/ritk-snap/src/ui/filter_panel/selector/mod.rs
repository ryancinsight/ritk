use crate::FilterKind;

mod selector_values;
mod selector_values_ext;
mod selector_values_third;

/// Render the filter-kind selector (label match + ComboBox with all
/// `selectable_value` entries).
pub fn show_selector(ui: &mut egui::Ui, active_filter: &mut FilterKind) {
    let kind_label = match active_filter {
        FilterKind::Gaussian { .. } => "Gaussian",
        FilterKind::Median { .. } => "Median",
        FilterKind::BedSeparation(_) => "Bed Separation",
        FilterKind::Clahe { .. } => "CLAHE",
        FilterKind::HistEq { .. } => "Hist Equalize",
        FilterKind::UnsharpMask { .. } => "Unsharp Mask",
        FilterKind::GradientAnisotropicDiffusion { .. } => "Gradient Aniso. Diffusion",
        FilterKind::ConnectedComponents { .. } => "Connected Components",
        FilterKind::RelabelComponents { .. } => "Relabel Components",
        FilterKind::MultiOtsuThreshold { .. } => "Multi-Otsu Threshold",
        FilterKind::BinaryErode { .. } => "Binary Erode",
        FilterKind::BinaryDilate { .. } => "Binary Dilate",
        FilterKind::BinaryClosing { .. } => "Binary Closing",
        FilterKind::BinaryOpening { .. } => "Binary Opening",
        FilterKind::BinaryFillhole { .. } => "Binary Fill Holes",
        FilterKind::GrayscaleClosing { .. } => "Grayscale Closing",
        FilterKind::GrayscaleOpening { .. } => "Grayscale Opening",
        FilterKind::GrayscaleFillhole => "Grayscale Fill Holes",
        FilterKind::Abs => "Abs",
        FilterKind::InvertIntensity { .. } => "Invert Intensity",
        FilterKind::NormalizeIntensity => "Normalize",
        FilterKind::Square => "Square",
        FilterKind::Sqrt => "Sqrt",
        FilterKind::Log => "Log",
        FilterKind::Exp => "Exp",
        FilterKind::MorphologicalGradient { .. } => "Morphological Gradient",
        FilterKind::DistanceTransform { .. } => "Distance Transform",
        FilterKind::SignedDistanceTransform { .. } => "Signed Distance Transform",
        FilterKind::FlipZ => "Flip Z",
        FilterKind::FlipY => "Flip Y",
        FilterKind::FlipX => "Flip X",
        FilterKind::MaskThreshold { .. } => "Mask Threshold",
        FilterKind::GeodesicDilationSelf => "Geodesic Dilation (self)",
        FilterKind::GeodesicErosionSelf => "Geodesic Erosion (self)",
        FilterKind::ShiftScale { .. } => "Shift Scale",
        FilterKind::ZeroCrossing { .. } => "Zero Crossing",
        FilterKind::RegionOfInterest { .. } => "Region Of Interest",
        FilterKind::PermuteAxes { .. } => "Permute Axes",
        FilterKind::Mean { .. } => "Mean",
        FilterKind::BinaryContour { .. } => "Binary Contour",
        FilterKind::LabelContour { .. } => "Label Contour",
        FilterKind::VotingBinary { .. } => "Voting Binary",
        FilterKind::Shrink { .. } => "Shrink",
        FilterKind::ConstantPad { .. } => "Constant Pad",
        FilterKind::MirrorPad { .. } => "Mirror Pad",
        FilterKind::WrapPad { .. } => "Wrap Pad",
        FilterKind::GrayscaleErode { .. } => "Grayscale Erode",
        FilterKind::GrayscaleDilate { .. } => "Grayscale Dilate",
        FilterKind::BinaryThreshold { .. } => "Binary Threshold",
        FilterKind::RescaleIntensity { .. } => "Rescale Intensity",
        FilterKind::Clamp { .. } => "Clamp",
        FilterKind::ConnectedThreshold { .. } => "Connected Threshold",
        FilterKind::ConfidenceConnected { .. } => "Confidence Connected",
        FilterKind::NeighborhoodConnected { .. } => "Neighborhood Connected",
        FilterKind::Atan => "Atan",
        FilterKind::Sin => "Sin",
        FilterKind::Cos => "Cos",
        FilterKind::Tan => "Tan",
        FilterKind::Asin => "Asin",
        FilterKind::Acos => "Acos",
        FilterKind::BoundedReciprocal => "Bounded Reciprocal",
        FilterKind::CurvatureFlow { .. } => "Curvature Flow",
        FilterKind::Cpr { .. } => "CPR" };

    egui::ComboBox::from_label("Filter")
        .selected_text(kind_label)
        .show_ui(ui, |ui| {
            selector_values::show_first_half(ui, active_filter);
            selector_values_ext::show_second_half(ui, active_filter);
        });
}
