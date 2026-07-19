use crate::FilterKind;
use ritk_filter::BinarizationThreshold;

/// Render parameter controls for Pointwise + Geometry + Threshold filter variants.
///
/// Returns `true` if the active variant was handled (i.e. belonged to this
/// group), `false` otherwise.
pub fn show_controls(ui: &mut egui::Ui, active_filter: &mut FilterKind) -> bool {
    match active_filter {
        FilterKind::Abs => {
            ui.label(
                egui::RichText::new(
                    "ITK AbsImageFilter / ImageJ Abs. out(x) = |in(x)|. No adjustable parameters.",
                )
                .small(),
            );
            true
        }
        FilterKind::InvertIntensity { maximum } => {
            ui.label(
                egui::RichText::new(
                    "ITK InvertIntensityImageFilter. out(x) = maximum - in(x). maximum=None â†’ computed from image.",
                )
                .small(),
            );
            let mut use_fixed = maximum.is_some();
            if ui.checkbox(&mut use_fixed, "Use fixed maximum").changed() {
                *maximum = if use_fixed { Some(255.0) } else { None };
            }
            if let Some(ref mut m) = maximum {
                ui.horizontal(|ui| {
                    ui.label("Maximum:");
                    ui.add(egui::DragValue::new(m).speed(1.0).range(0.0..=f32::MAX));
                });
            }
            true
        }
        FilterKind::NormalizeIntensity => {
            ui.label(
                egui::RichText::new(
                    "ITK NormalizeImageFilter. out(x) = (in(x) - mean) / std. Constant image â†’ all zero. No adjustable parameters.",
                )
                .small(),
            );
            true
        }
        FilterKind::Square => {
            ui.label(
                egui::RichText::new(
                    "ITK SquareImageFilter / ImageJ Square. out(x) = in(x)Â². No adjustable parameters.",
                )
                .small(),
            );
            true
        }
        FilterKind::Sqrt => {
            ui.label(
                egui::RichText::new(
                    "ITK SqrtImageFilter / ImageJ Square Root. out(x) = âˆšin(x). Negative â†’ NaN. No adjustable parameters.",
                )
                .small(),
            );
            true
        }
        FilterKind::Log => {
            ui.label(
                egui::RichText::new(
                    "ITK LogImageFilter / ImageJ Log. out(x) = ln(in(x)). Non-positive â†’ -inf/NaN. No adjustable parameters.",
                )
                .small(),
            );
            true
        }
        FilterKind::Exp => {
            ui.label(
                egui::RichText::new(
                    "ITK ExpImageFilter / ImageJ Exp. out(x) = e^in(x). No adjustable parameters.",
                )
                .small(),
            );
            true
        }
        FilterKind::DistanceTransform { threshold }
        | FilterKind::SignedDistanceTransform { threshold } => {
            ui.label(
                egui::RichText::new(
                    "Euclidean distance transform. Each voxel receives distance (mm) to nearest foreground voxel.",
                )
                .small(),
            );
            ui.horizontal(|ui| {
                ui.label("Foreground threshold:");
                let mut t = f32::from(*threshold);
                if ui
                    .add(egui::Slider::new(&mut t, 0.0_f32..=1000.0).step_by(0.1))
                    .changed()
                {
                    *threshold = BinarizationThreshold::new(t)
                        .expect("invariant: threshold slider is finite and non-negative");
                }
            });
            true
        }
        FilterKind::FlipZ | FilterKind::FlipY | FilterKind::FlipX => {
            ui.label(
                egui::RichText::new(
                    "Reverses voxel ordering along the selected axis. No adjustable parameters.",
                )
                .small(),
            );
            true
        }
        FilterKind::MaskThreshold { threshold } => {
            ui.label(
                egui::RichText::new(
                    "Zero-out voxels at or below the threshold (binary self-mask).",
                )
                .small(),
            );
            ui.horizontal(|ui| {
                ui.label("Threshold:");
                let mut t = f32::from(*threshold);
                if ui
                    .add(egui::Slider::new(&mut t, 0.0_f32..=1000.0).step_by(0.1))
                    .changed()
                {
                    *threshold = BinarizationThreshold::new(t)
                        .expect("invariant: threshold slider is finite and non-negative");
                }
            });
            true
        }
        FilterKind::ShiftScale { shift, scale } => {
            ui.label(
                egui::RichText::new(
                    "ITK ShiftScaleImageFilter: out(x) = (in(x) + shift) Ã— scale. Applied in f64 precision.",
                )
                .small(),
            );
            ui.horizontal(|ui| {
                ui.label("Shift:");
                ui.add(egui::Slider::new(shift, -10000.0_f32..=10000.0).step_by(1.0));
            });
            ui.horizontal(|ui| {
                ui.label("Scale:");
                ui.add(egui::Slider::new(scale, -100.0_f32..=100.0).step_by(0.001));
            });
            true
        }
        FilterKind::ZeroCrossing {
            foreground_value,
            background_value,
        } => {
            ui.label(
                egui::RichText::new(
                    "ITK ZeroCrossingImageFilter: emits foreground_value where a sign change (or exact zero) exists in the 6-connected neighbourhood.",
                )
                .small(),
            );
            ui.horizontal(|ui| {
                ui.label("Foreground:");
                ui.add(egui::Slider::new(&mut foreground_value.0, 0.0_f32..=1000.0).step_by(1.0));
            });
            ui.horizontal(|ui| {
                ui.label("Background:");
                ui.add(egui::Slider::new(background_value, -1000.0_f32..=1000.0).step_by(1.0));
            });
            true
        }
        FilterKind::RegionOfInterest {
            start_z,
            start_y,
            start_x,
            size_z,
            size_y,
            size_x,
        } => {
            ui.label(
                egui::RichText::new(
                    "ITK RegionOfInterestImageFilter: extract a rectangular sub-volume. Origin is updated to the physical start voxel.",
                )
                .small(),
            );
            let mut sz_ = *start_z as i32;
            let mut sy_ = *start_y as i32;
            let mut sx_ = *start_x as i32;
            let mut esz = *size_z as i32;
            let mut esy = *size_y as i32;
            let mut esx = *size_x as i32;
            ui.horizontal(|ui| {
                ui.label("Start Z:");
                if ui
                    .add(egui::Slider::new(&mut sz_, 0..=4095).step_by(1.0))
                    .changed()
                {
                    *start_z = sz_.max(0) as usize;
                }
            });
            ui.horizontal(|ui| {
                ui.label("Start Y:");
                if ui
                    .add(egui::Slider::new(&mut sy_, 0..=4095).step_by(1.0))
                    .changed()
                {
                    *start_y = sy_.max(0) as usize;
                }
            });
            ui.horizontal(|ui| {
                ui.label("Start X:");
                if ui
                    .add(egui::Slider::new(&mut sx_, 0..=4095).step_by(1.0))
                    .changed()
                {
                    *start_x = sx_.max(0) as usize;
                }
            });
            ui.horizontal(|ui| {
                ui.label("Size Z:");
                if ui
                    .add(egui::Slider::new(&mut esz, 1..=4096).step_by(1.0))
                    .changed()
                {
                    *size_z = esz.max(1) as usize;
                }
            });
            ui.horizontal(|ui| {
                ui.label("Size Y:");
                if ui
                    .add(egui::Slider::new(&mut esy, 1..=4096).step_by(1.0))
                    .changed()
                {
                    *size_y = esy.max(1) as usize;
                }
            });
            ui.horizontal(|ui| {
                ui.label("Size X:");
                if ui
                    .add(egui::Slider::new(&mut esx, 1..=4096).step_by(1.0))
                    .changed()
                {
                    *size_x = esx.max(1) as usize;
                }
            });
            true
        }
        FilterKind::PermuteAxes {
            order_0,
            order_1,
            order_2,
        } => {
            ui.label(
                egui::RichText::new(
                    "ITK PermuteAxesImageFilter: rearrange axes. order[i] = source axis for output axis i. Must be a permutation of {0, 1, 2}.",
                )
                .small(),
            );
            let axes = [0i32, 1, 2];
            let mut o0 = *order_0 as i32;
            let mut o1 = *order_1 as i32;
            let mut o2 = *order_2 as i32;
            ui.horizontal(|ui| {
                ui.label("Output axis 0 â† input axis:");
                egui::ComboBox::from_id_source("perm_ax0")
                    .selected_text(format!("{o0}"))
                    .show_ui(ui, |ui| {
                        for &ax in &axes {
                            ui.selectable_value(&mut o0, ax, format!("{ax}"));
                        }
                    });
                *order_0 = o0.max(0) as usize;
            });
            ui.horizontal(|ui| {
                ui.label("Output axis 1 â† input axis:");
                egui::ComboBox::from_id_source("perm_ax1")
                    .selected_text(format!("{o1}"))
                    .show_ui(ui, |ui| {
                        for &ax in &axes {
                            ui.selectable_value(&mut o1, ax, format!("{ax}"));
                        }
                    });
                *order_1 = o1.max(0) as usize;
            });
            ui.horizontal(|ui| {
                ui.label("Output axis 2 â† input axis:");
                egui::ComboBox::from_id_source("perm_ax2")
                    .selected_text(format!("{o2}"))
                    .show_ui(ui, |ui| {
                        for &ax in &axes {
                            ui.selectable_value(&mut o2, ax, format!("{ax}"));
                        }
                    });
                *order_2 = o2.max(0) as usize;
            });
            true
        }
        FilterKind::Mean { radius } => {
            let mut r = *radius as i32;
            ui.horizontal(|ui| {
                ui.label("Radius (voxels):");
                if ui
                    .add(egui::Slider::new(&mut r, 0..=10).step_by(1.0))
                    .changed()
                {
                    *radius = r.max(0) as usize;
                }
            });
            ui.label(
                egui::RichText::new(
                    "ITK MeanImageFilter: arithmetic mean of (2r+1)Â³ neighbourhood.",
                )
                .small(),
            );
            true
        }
        FilterKind::BinaryThreshold {
            lower,
            upper,
            foreground,
            background,
        } => {
            ui.horizontal(|ui| {
                ui.label("Lower:");
                ui.add(egui::DragValue::new(lower).speed(1.0));
            });
            ui.horizontal(|ui| {
                ui.label("Upper:");
                ui.add(egui::DragValue::new(upper).speed(1.0));
            });
            ui.horizontal(|ui| {
                ui.label("Foreground:");
                ui.add(egui::DragValue::new(&mut foreground.0).speed(0.1));
            });
            ui.horizontal(|ui| {
                ui.label("Background:");
                ui.add(egui::DragValue::new(background).speed(0.1));
            });
            ui.label(
                egui::RichText::new(
                    "ITK BinaryThresholdImageFilter. out = fg if lowerâ‰¤Iâ‰¤upper, else bg.",
                )
                .small(),
            );
            true
        }
        FilterKind::RescaleIntensity { out_min, out_max } => {
            ui.horizontal(|ui| {
                ui.label("Out min:");
                ui.add(egui::DragValue::new(out_min).speed(1.0));
            });
            ui.horizontal(|ui| {
                ui.label("Out max:");
                ui.add(egui::DragValue::new(out_max).speed(1.0));
            });
            ui.label(
                egui::RichText::new(
                    "ITK RescaleIntensityImageFilter. Maps [I_min, I_max] linearly to [out_min, out_max].",
                )
                .small(),
            );
            true
        }
        FilterKind::Clamp { lower, upper } => {
            ui.horizontal(|ui| {
                ui.label("Lower:");
                ui.add(egui::DragValue::new(lower).speed(1.0));
            });
            ui.horizontal(|ui| {
                ui.label("Upper:");
                ui.add(egui::DragValue::new(upper).speed(1.0));
            });
            ui.label(
                egui::RichText::new("ITK ClampImageFilter. out = clamp(I, lower, upper).").small(),
            );
            true
        }
        FilterKind::Atan => {
            ui.label(
                egui::RichText::new(
                    "ITK AtanImageFilter. out(x) = atan(in(x)). Range (âˆ’Ï€/2, Ï€/2). No parameters.",
                )
                .small(),
            );
            true
        }
        FilterKind::Sin => {
            ui.label(
                egui::RichText::new(
                    "ITK SinImageFilter. out(x) = sin(in(x)), input in radians. Range [âˆ’1, 1]. No parameters.",
                )
                .small(),
            );
            true
        }
        FilterKind::Cos => {
            ui.label(
                egui::RichText::new(
                    "ITK CosImageFilter. out(x) = cos(in(x)), input in radians. Range [âˆ’1, 1]. No parameters.",
                )
                .small(),
            );
            true
        }
        FilterKind::Tan => {
            ui.label(
                egui::RichText::new(
                    "ITK TanImageFilter. out(x) = tan(in(x)), input in radians. No parameters.",
                )
                .small(),
            );
            true
        }
        FilterKind::Asin => {
            ui.label(
                egui::RichText::new(
                    "ITK AsinImageFilter. out(x) = asin(in(x)). Domain [âˆ’1,1], range [âˆ’Ï€/2, Ï€/2]. No parameters.",
                )
                .small(),
            );
            true
        }
        FilterKind::Acos => {
            ui.label(
                egui::RichText::new(
                    "ITK AcosImageFilter. out(x) = acos(in(x)). Domain [âˆ’1,1], range [0, Ï€]. No parameters.",
                )
                .small(),
            );
            true
        }
        FilterKind::BoundedReciprocal => {
            ui.label(
                egui::RichText::new(
                    "ITK BoundedReciprocalImageFilter. out(x) = 1/(1+|x|). Range (0,1]. No parameters.",
                )
                .small(),
            );
            true
        }
        _ => false,
    }
}
