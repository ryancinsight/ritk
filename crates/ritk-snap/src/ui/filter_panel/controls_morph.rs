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
        _ => false,
    }
}
