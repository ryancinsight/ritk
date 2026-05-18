use crate::FilterKind;

/// Render parameter controls for Binary + Grayscale Morphology + Pad/Geometry
/// filter variants.
///
/// Returns `true` if the active variant was handled (i.e. belonged to this
/// group), `false` otherwise.
pub fn show_controls(ui: &mut egui::Ui, active_filter: &mut FilterKind) -> bool {
    match active_filter {
        FilterKind::BinaryErode {
            radius,
            foreground_value,
        } => {
            let mut r = *radius as i32;
            ui.horizontal(|ui| {
                ui.label("Radius (voxels):");
                if ui
                    .add(egui::DragValue::new(&mut r).speed(1.0).range(0..=10))
                    .changed()
                {
                    *radius = r.clamp(0, 10) as usize;
                }
            });
            ui.horizontal(|ui| {
                ui.label("Foreground value:");
                ui.add(egui::DragValue::new(foreground_value).speed(1.0));
            });
            ui.label(egui::RichText::new("ITK BinaryErodeImageFilter parity.").small());
            true
        }
        FilterKind::BinaryDilate {
            radius,
            foreground_value,
        } => {
            let mut r = *radius as i32;
            ui.horizontal(|ui| {
                ui.label("Radius (voxels):");
                if ui
                    .add(egui::DragValue::new(&mut r).speed(1.0).range(0..=10))
                    .changed()
                {
                    *radius = r.clamp(0, 10) as usize;
                }
            });
            ui.horizontal(|ui| {
                ui.label("Foreground value:");
                ui.add(egui::DragValue::new(foreground_value).speed(1.0));
            });
            ui.label(egui::RichText::new("ITK BinaryDilateImageFilter parity.").small());
            true
        }
        FilterKind::BinaryClosing {
            radius,
            foreground_value,
        } => {
            let mut r = *radius as i32;
            ui.horizontal(|ui| {
                ui.label("Radius (voxels):");
                if ui
                    .add(egui::DragValue::new(&mut r).speed(1.0).range(0..=10))
                    .changed()
                {
                    *radius = r.clamp(0, 10) as usize;
                }
            });
            ui.horizontal(|ui| {
                ui.label("Foreground value:");
                ui.add(egui::DragValue::new(foreground_value).speed(1.0));
            });
            ui.label(
                egui::RichText::new(
                    "ITK BinaryMorphologicalClosingImageFilter parity. Fills dark holes.",
                )
                .small(),
            );
            true
        }
        FilterKind::BinaryOpening {
            radius,
            foreground_value,
        } => {
            let mut r = *radius as i32;
            ui.horizontal(|ui| {
                ui.label("Radius (voxels):");
                if ui
                    .add(egui::DragValue::new(&mut r).speed(1.0).range(0..=10))
                    .changed()
                {
                    *radius = r.clamp(0, 10) as usize;
                }
            });
            ui.horizontal(|ui| {
                ui.label("Foreground value:");
                ui.add(egui::DragValue::new(foreground_value).speed(1.0));
            });
            ui.label(
                egui::RichText::new(
                    "ITK BinaryMorphologicalOpeningImageFilter parity. Removes small bright blobs.",
                )
                .small(),
            );
            true
        }
        FilterKind::BinaryFillhole { foreground_value } => {
            ui.horizontal(|ui| {
                ui.label("Foreground value:");
                ui.add(egui::DragValue::new(foreground_value).speed(1.0));
            });
            ui.label(
                egui::RichText::new(
                    "ITK BinaryFillholeImageFilter parity. Fills enclosed background cavities.",
                )
                .small(),
            );
            true
        }
        FilterKind::BinaryContour {
            fully_connected,
            foreground_value,
        } => {
            ui.horizontal(|ui| {
                ui.label("26-connected:");
                ui.checkbox(fully_connected, "");
            });
            ui.horizontal(|ui| {
                ui.label("Foreground value:");
                ui.add(egui::DragValue::new(foreground_value).speed(1.0));
            });
            ui.label(
                egui::RichText::new(
                    "ITK BinaryContourImageFilter: border voxels of binary objects.",
                )
                .small(),
            );
            true
        }
        FilterKind::LabelContour {
            fully_connected,
            background_value,
        } => {
            ui.horizontal(|ui| {
                ui.label("26-connected:");
                ui.checkbox(fully_connected, "");
            });
            ui.horizontal(|ui| {
                ui.label("Background value:");
                ui.add(egui::DragValue::new(background_value).speed(1.0));
            });
            ui.label(
                egui::RichText::new(
                    "ITK LabelContourImageFilter: boundaries between label regions.",
                )
                .small(),
            );
            true
        }
        FilterKind::VotingBinary {
            radius,
            birth_threshold,
            survival_threshold,
            foreground_value,
            background_value,
        } => {
            let mut r = *radius as i32;
            let mut bt = *birth_threshold as i32;
            let mut st = *survival_threshold as i32;
            ui.horizontal(|ui| {
                ui.label("Radius:");
                if ui
                    .add(egui::Slider::new(&mut r, 0..=5).step_by(1.0))
                    .changed()
                {
                    *radius = r.max(0) as usize;
                }
            });
            ui.horizontal(|ui| {
                ui.label("Birth threshold:");
                if ui
                    .add(egui::Slider::new(&mut bt, 0..=26).step_by(1.0))
                    .changed()
                {
                    *birth_threshold = bt.max(0) as usize;
                }
            });
            ui.horizontal(|ui| {
                ui.label("Survival threshold:");
                if ui
                    .add(egui::Slider::new(&mut st, 0..=26).step_by(1.0))
                    .changed()
                {
                    *survival_threshold = st.max(0) as usize;
                }
            });
            ui.horizontal(|ui| {
                ui.label("Foreground value:");
                ui.add(egui::DragValue::new(foreground_value).speed(1.0));
            });
            ui.horizontal(|ui| {
                ui.label("Background value:");
                ui.add(egui::DragValue::new(background_value).speed(1.0));
            });
            ui.label(
                egui::RichText::new("ITK VotingBinaryImageFilter: cellular automata voting step.")
                    .small(),
            );
            true
        }
        FilterKind::GrayscaleClosing { radius } => {
            let mut r = *radius as i32;
            ui.horizontal(|ui| {
                ui.label("Radius (voxels):");
                if ui
                    .add(egui::DragValue::new(&mut r).speed(1.0).range(0..=10))
                    .changed()
                {
                    *radius = r.clamp(0, 10) as usize;
                }
            });
            ui.label(
                egui::RichText::new(
                    "ITK GrayscaleMorphologicalClosingImageFilter parity. C_B(f)=E_B(D_B(f)). Fills dark voids.",
                )
                .small(),
            );
            true
        }
        FilterKind::GrayscaleOpening { radius } => {
            let mut r = *radius as i32;
            ui.horizontal(|ui| {
                ui.label("Radius (voxels):");
                if ui
                    .add(egui::DragValue::new(&mut r).speed(1.0).range(0..=10))
                    .changed()
                {
                    *radius = r.clamp(0, 10) as usize;
                }
            });
            ui.label(
                egui::RichText::new(
                    "ITK GrayscaleMorphologicalOpeningImageFilter parity. O_B(f)=D_B(E_B(f)). Removes bright protrusions.",
                )
                .small(),
            );
            true
        }
        FilterKind::GrayscaleFillhole => {
            ui.label(
                egui::RichText::new(
                    "ITK GrayscaleFillholeImageFilter parity. Fills dark regional minima not connected to the image border.",
                )
                .small(),
            );
            true
        }
        FilterKind::MorphologicalGradient { radius } => {
            ui.label(
                egui::RichText::new(
                    "ITK GrayscaleMorphologicalGradientImageFilter (Beucher gradient). out(x) = D_B(f)(x) - E_B(f)(x). Non-negative.",
                )
                .small(),
            );
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
            true
        }
        FilterKind::GrayscaleErode { radius } => {
            let mut r = *radius as i32;
            ui.horizontal(|ui| {
                ui.label("Radius (voxels):");
                if ui
                    .add(egui::DragValue::new(&mut r).speed(1.0).range(0..=10))
                    .changed()
                {
                    *radius = r.clamp(0, 10) as usize;
                }
            });
            ui.label(
                egui::RichText::new(
                    "ITK GrayscaleErodeImageFilter (flat SE). E_B(f)(x) = min_{b∈B} f(x+b). Anti-extensive.",
                )
                .small(),
            );
            true
        }
        FilterKind::GrayscaleDilate { radius } => {
            let mut r = *radius as i32;
            ui.horizontal(|ui| {
                ui.label("Radius (voxels):");
                if ui
                    .add(egui::DragValue::new(&mut r).speed(1.0).range(0..=10))
                    .changed()
                {
                    *radius = r.clamp(0, 10) as usize;
                }
            });
            ui.label(
                egui::RichText::new(
                    "ITK GrayscaleDilateImageFilter (flat SE). D_B(f)(x) = max_{b∈B} f(x+b). Extensive.",
                )
                .small(),
            );
            true
        }
        FilterKind::GeodesicDilationSelf | FilterKind::GeodesicErosionSelf => {
            ui.label(
                egui::RichText::new(
                    "Geodesic morphological reconstruction (marker = mask = current image). Identity on self; for two-image reconstruction use the ritk_core API.",
                )
                .small(),
            );
            true
        }
        FilterKind::Shrink {
            factor_z,
            factor_y,
            factor_x,
        } => {
            let mut fz = *factor_z as i32;
            let mut fy = *factor_y as i32;
            let mut fx = *factor_x as i32;
            ui.horizontal(|ui| {
                ui.label("Factor Z:");
                if ui
                    .add(egui::Slider::new(&mut fz, 1..=16).step_by(1.0))
                    .changed()
                {
                    *factor_z = fz.max(1) as usize;
                }
            });
            ui.horizontal(|ui| {
                ui.label("Factor Y:");
                if ui
                    .add(egui::Slider::new(&mut fy, 1..=16).step_by(1.0))
                    .changed()
                {
                    *factor_y = fy.max(1) as usize;
                }
            });
            ui.horizontal(|ui| {
                ui.label("Factor X:");
                if ui
                    .add(egui::Slider::new(&mut fx, 1..=16).step_by(1.0))
                    .changed()
                {
                    *factor_x = fx.max(1) as usize;
                }
            });
            ui.label(
                egui::RichText::new(
                    "ITK ShrinkImageFilter: integer downsampling by tile averaging. Spacing is updated.",
                )
                .small(),
            );
            true
        }
        FilterKind::ConstantPad {
            pad_lower_z,
            pad_lower_y,
            pad_lower_x,
            pad_upper_z,
            pad_upper_y,
            pad_upper_x,
            constant,
        } => {
            for (label, val) in [
                ("↓Z", pad_lower_z),
                ("↓Y", pad_lower_y),
                ("↓X", pad_lower_x),
                ("↑Z", pad_upper_z),
                ("↑Y", pad_upper_y),
                ("↑X", pad_upper_x),
            ] {
                let mut v = *val as i32;
                ui.horizontal(|ui| {
                    ui.label(format!("Pad {label}:"));
                    if ui
                        .add(egui::Slider::new(&mut v, 0..=128).step_by(1.0))
                        .changed()
                    {
                        *val = v.max(0) as usize;
                    }
                });
            }
            ui.horizontal(|ui| {
                ui.label("Constant:");
                ui.add(egui::DragValue::new(constant).speed(1.0));
            });
            ui.label(egui::RichText::new("ITK ConstantPadImageFilter.").small());
            true
        }
        FilterKind::MirrorPad {
            pad_lower_z,
            pad_lower_y,
            pad_lower_x,
            pad_upper_z,
            pad_upper_y,
            pad_upper_x,
        } => {
            for (label, val) in [
                ("↓Z", pad_lower_z),
                ("↓Y", pad_lower_y),
                ("↓X", pad_lower_x),
                ("↑Z", pad_upper_z),
                ("↑Y", pad_upper_y),
                ("↑X", pad_upper_x),
            ] {
                let mut v = *val as i32;
                ui.horizontal(|ui| {
                    ui.label(format!("Pad {label}:"));
                    if ui
                        .add(egui::Slider::new(&mut v, 0..=128).step_by(1.0))
                        .changed()
                    {
                        *val = v.max(0) as usize;
                    }
                });
            }
            ui.label(
                egui::RichText::new("ITK MirrorPadImageFilter: symmetric reflection.").small(),
            );
            true
        }
        FilterKind::WrapPad {
            pad_lower_z,
            pad_lower_y,
            pad_lower_x,
            pad_upper_z,
            pad_upper_y,
            pad_upper_x,
        } => {
            for (label, val) in [
                ("↓Z", pad_lower_z),
                ("↓Y", pad_lower_y),
                ("↓X", pad_lower_x),
                ("↑Z", pad_upper_z),
                ("↑Y", pad_upper_y),
                ("↑X", pad_upper_x),
            ] {
                let mut v = *val as i32;
                ui.horizontal(|ui| {
                    ui.label(format!("Pad {label}:"));
                    if ui
                        .add(egui::Slider::new(&mut v, 0..=128).step_by(1.0))
                        .changed()
                    {
                        *val = v.max(0) as usize;
                    }
                });
            }
            ui.label(egui::RichText::new("ITK WrapPadImageFilter: periodic extension.").small());
            true
        }
        _ => false,
    }
}
