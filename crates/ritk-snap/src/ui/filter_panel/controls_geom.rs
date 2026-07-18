use crate::FilterKind;

/// Render parameter controls for geometry and padding filter variants
/// (`Shrink`, `ConstantPad`, `MirrorPad`, `WrapPad`).
///
/// Returns `true` if the active variant was handled, `false` otherwise.
pub fn show_controls(ui: &mut egui::Ui, active_filter: &mut FilterKind) -> bool {
    match active_filter {
        FilterKind::Shrink {
            factor_z,
            factor_y,
            factor_x } => {
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
                    "Tile-averaging downsample (anti-aliased). Integer factors per axis; spacing is updated.",
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
            constant } => {
            for (label, val) in [
                ("â†“Z", pad_lower_z),
                ("â†“Y", pad_lower_y),
                ("â†“X", pad_lower_x),
                ("â†‘Z", pad_upper_z),
                ("â†‘Y", pad_upper_y),
                ("â†‘X", pad_upper_x),
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
            pad_upper_x } => {
            for (label, val) in [
                ("â†“Z", pad_lower_z),
                ("â†“Y", pad_lower_y),
                ("â†“X", pad_lower_x),
                ("â†‘Z", pad_upper_z),
                ("â†‘Y", pad_upper_y),
                ("â†‘X", pad_upper_x),
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
            pad_upper_x } => {
            for (label, val) in [
                ("â†“Z", pad_lower_z),
                ("â†“Y", pad_lower_y),
                ("â†“X", pad_lower_x),
                ("â†‘Z", pad_upper_z),
                ("â†‘Y", pad_upper_y),
                ("â†‘X", pad_upper_x),
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
        _ => false }
}
