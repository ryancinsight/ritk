use crate::FilterKind;

/// Render parameter controls for CPR (Curved Planar Reformation) filter.
///
/// Returns `true` if the active variant was handled, `false` otherwise.
pub fn show_controls(ui: &mut egui::Ui, active_filter: &mut FilterKind) -> bool {
    match active_filter {
        FilterKind::Cpr {
            control_points,
            num_path_samples,
            cross_section_half_width,
            num_cross_samples,
        } => {
            ui.label(egui::RichText::new("CPR generates a 2-D straightened view along a curved path defined by 3-D control points.").small());

            let mut nps = *num_path_samples as i32;
            ui.horizontal(|ui| {
                ui.label("Path samples:");
                if ui
                    .add(egui::Slider::new(&mut nps, 2..=1024).step_by(1.0))
                    .changed()
                {
                    *num_path_samples = nps.max(2) as u32;
                }
            });

            ui.horizontal(|ui| {
                ui.label("Cross-section half-width (mm):");
                ui.add(egui::Slider::new(cross_section_half_width, 0.1_f32..=100.0).step_by(0.5));
            });

            let mut ncs = *num_cross_samples as i32;
            ui.horizontal(|ui| {
                ui.label("Cross-section samples:");
                if ui
                    .add(egui::Slider::new(&mut ncs, 2..=512).step_by(1.0))
                    .changed()
                {
                    *num_cross_samples = ncs.max(2) as u32;
                }
            });

            ui.label("Control points ([z, y, x]; semicolon-separated):");
            let pts_str = control_points
                .iter()
                .map(|p| format!("[{}, {}, {}]", p[0], p[1], p[2]))
                .collect::<Vec<_>>()
                .join("; ");
            let mut buf = pts_str.clone();
            if ui
                .add(egui::TextEdit::singleline(&mut buf).hint_text("[0, 0, 0]; [10, 0, 0]"))
                .changed()
            {
                *control_points = buf
                    .split(';')
                    .filter_map(|s| {
                        let s = s.trim().trim_matches(|c| c == '[' || c == ']');
                        let parts: Vec<f64> =
                            s.split(',').filter_map(|v| v.trim().parse().ok()).collect();
                        if parts.len() == 3 {
                            Some([parts[0], parts[1], parts[2]])
                        } else {
                            None
                        }
                    })
                    .collect();
            }

            if control_points.len() < 2 {
                ui.label(egui::RichText::new("Requires at least 2 control points.").strong());
            }

            if ui.button("Reset to defaults").clicked() {
                *control_points = vec![[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]];
                *num_path_samples = 256;
                *cross_section_half_width = 10.0;
                *num_cross_samples = 64;
            }

            true
        }
        _ => false,
    }
}
