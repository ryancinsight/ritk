use crate::FilterKind;

/// Render parameter controls for Smoothing + Segmentation filter variants
/// (including ConnectedThreshold, ConfidenceConnected, NeighborhoodConnected).
///
/// Returns `true` if the active variant was handled (i.e. belonged to this
/// group), `false` otherwise.
pub fn show_controls(ui: &mut egui::Ui, active_filter: &mut FilterKind) -> bool {
    match active_filter {
        FilterKind::Gaussian { sigma } => {
            ui.horizontal(|ui| {
                ui.label("σ (mm):");
                ui.add(
                    egui::Slider::new(sigma, 0.1_f32..=20.0)
                        .step_by(0.1)
                        .suffix(" mm"),
                );
            });
            true
        }
        FilterKind::Median { radius } => {
            // usize slider: convert via i32 proxy.
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
        FilterKind::BedSeparation(_) => {
            ui.label("No adjustable parameters.");
            true
        }
        FilterKind::Clahe {
            tile_grid_size,
            clip_limit,
        } => {
            let mut ty = tile_grid_size[0] as i32;
            let mut tx = tile_grid_size[1] as i32;
            ui.horizontal(|ui| {
                ui.label("Tiles Y:");
                if ui
                    .add(egui::Slider::new(&mut ty, 1..=32).step_by(1.0))
                    .changed()
                {
                    tile_grid_size[0] = ty.max(1) as usize;
                }
            });
            ui.horizontal(|ui| {
                ui.label("Tiles X:");
                if ui
                    .add(egui::Slider::new(&mut tx, 1..=32).step_by(1.0))
                    .changed()
                {
                    tile_grid_size[1] = tx.max(1) as usize;
                }
            });
            ui.horizontal(|ui| {
                ui.label("Clip limit:");
                ui.add(egui::Slider::new(clip_limit, 1.0_f32..=200.0).step_by(1.0));
            });
            true
        }
        FilterKind::HistEq { bins } => {
            let mut b = *bins as i32;
            ui.horizontal(|ui| {
                ui.label("Bins:");
                if ui
                    .add(egui::Slider::new(&mut b, 2..=1024).step_by(1.0))
                    .changed()
                {
                    *bins = b.max(2) as usize;
                }
            });
            true
        }
        FilterKind::UnsharpMask {
            sigma,
            amount,
            threshold,
            clamp,
        } => {
            ui.horizontal(|ui| {
                ui.label("σ (mm):");
                ui.add(
                    egui::Slider::new(sigma, 0.1_f32..=10.0)
                        .step_by(0.1)
                        .suffix(" mm"),
                );
            });
            ui.horizontal(|ui| {
                ui.label("Amount:");
                ui.add(egui::Slider::new(amount, 0.0_f32..=5.0).step_by(0.05));
            });
            ui.horizontal(|ui| {
                ui.label("Threshold:");
                ui.add(egui::Slider::new(threshold, 0.0_f32..=100.0).step_by(0.5));
            });
            ui.horizontal(|ui| {
                ui.label("Clamp:");
                let mut clamp_bool =
                    matches!(*clamp, ritk_core::filter::ClampPolicy::ClampToInputRange);
                if ui.checkbox(&mut clamp_bool, "").changed() {
                    *clamp = if clamp_bool {
                        ritk_core::filter::ClampPolicy::ClampToInputRange
                    } else {
                        ritk_core::filter::ClampPolicy::NoClamp
                    };
                }
            });
            true
        }
        FilterKind::GradientAnisotropicDiffusion {
            iterations,
            time_step,
            conductance,
        } => {
            let mut it = *iterations as i32;
            ui.horizontal(|ui| {
                ui.label("Iterations:");
                if ui
                    .add(egui::Slider::new(&mut it, 1..=50).step_by(1.0))
                    .changed()
                {
                    *iterations = it.max(1) as u32;
                }
            });
            ui.horizontal(|ui| {
                ui.label("Time step Δt:");
                // Stability bound: Δt ≤ 1/6 ≈ 0.1667 in 3-D.
                ui.add(egui::Slider::new(time_step, 0.01_f32..=0.1667).step_by(0.005));
            });
            ui.horizontal(|ui| {
                ui.label("Conductance K:");
                ui.add(
                    egui::Slider::new(conductance, 0.1_f32..=100.0)
                        .step_by(0.1)
                        .logarithmic(true),
                );
            });
            true
        }
        FilterKind::ConnectedComponents {
            connectivity,
            background_value,
        } => {
            ui.horizontal(|ui| {
                ui.label("Connectivity:");
                egui::ComboBox::from_id_source("connected_components_connectivity")
                    .selected_text(match connectivity {
                        ritk_core::filter::Connectivity::Face6 => "6-connected (default)",
                        ritk_core::filter::Connectivity::Vertex26 => "26-connected",
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            connectivity,
                            ritk_core::filter::Connectivity::Face6,
                            "6-connected (default)",
                        );
                        ui.selectable_value(
                            connectivity,
                            ritk_core::filter::Connectivity::Vertex26,
                            "26-connected",
                        );
                    });
            });
            ui.horizontal(|ui| {
                ui.label("Background value:");
                ui.add(egui::DragValue::new(background_value).speed(1.0).prefix(""));
            });
            ui.label(
                egui::RichText::new("Output: integer label image (0=background, 1…N=components)")
                    .small(),
            );
            true
        }
        FilterKind::RelabelComponents {
            minimum_object_size,
        } => {
            // minimum_object_size is u32; use i32 proxy for DragValue.
            let mut mos = *minimum_object_size as i32;
            ui.horizontal(|ui| {
                ui.label("Min object size (voxels):");
                if ui
                    .add(
                        egui::DragValue::new(&mut mos)
                            .speed(1.0)
                            .range(0..=i32::MAX),
                    )
                    .changed()
                {
                    *minimum_object_size = mos.max(0) as u32;
                }
            });
            ui.label(
                egui::RichText::new(
                    "Input: label image. Output: relabeled image (label 1 = largest component).",
                )
                .small(),
            );
            true
        }
        FilterKind::MultiOtsuThreshold { num_classes } => {
            // num_classes is u32; use i32 proxy for Slider.
            let mut nc = *num_classes as i32;
            ui.horizontal(|ui| {
                ui.label("Classes K:");
                if ui
                    .add(egui::Slider::new(&mut nc, 2..=8).step_by(1.0))
                    .changed()
                {
                    *num_classes = nc.max(2) as u32;
                }
            });
            ui.label(
                egui::RichText::new(
                    "Output: class label image with values 0…K−1. ITK default K=3.",
                )
                .small(),
            );
            true
        }
        FilterKind::CurvatureFlow {
            iterations,
            time_step,
        } => {
            let mut iters_i = *iterations as i32;
            ui.horizontal(|ui| {
                ui.label("Iterations:");
                if ui
                    .add(egui::Slider::new(&mut iters_i, 1..=50).step_by(1.0))
                    .changed()
                {
                    *iterations = iters_i.max(1) as u32;
                }
            });
            ui.horizontal(|ui| {
                ui.label("Time Step Δt:");
                ui.add(egui::Slider::new(time_step, 0.001_f32..=0.166_f32).step_by(0.001));
            });
            ui.label(
                egui::RichText::new(
                    "ITK CurvatureFlowImageFilter. ∂I/∂t = κ (mean curvature, no gradient weighting). Stability: Δt ≤ 1/6.",
                )
                .small(),
            );
            true
        }
        FilterKind::ConnectedThreshold {
            seed_z,
            seed_y,
            seed_x,
            lower,
            upper,
        } => {
            for (label, val) in [("Seed Z", seed_z), ("Seed Y", seed_y), ("Seed X", seed_x)] {
                let mut v = *val as i32;
                ui.horizontal(|ui| {
                    ui.label(format!("{label}:"));
                    if ui
                        .add(egui::DragValue::new(&mut v).speed(1.0).range(0..=i32::MAX))
                        .changed()
                    {
                        *val = v.max(0) as usize;
                    }
                });
            }
            ui.horizontal(|ui| {
                ui.label("Lower:");
                ui.add(egui::DragValue::new(lower).speed(1.0));
            });
            ui.horizontal(|ui| {
                ui.label("Upper:");
                ui.add(egui::DragValue::new(upper).speed(1.0));
            });
            ui.label(
                egui::RichText::new(
                    "ITK ConnectedThresholdImageFilter. BFS flood-fill where I(v) ∈ [lower, upper]. Output: binary mask.",
                )
                .small(),
            );
            true
        }
        FilterKind::ConfidenceConnected {
            seed_z,
            seed_y,
            seed_x,
            initial_lower,
            initial_upper,
            multiplier,
            max_iterations,
        } => {
            for (label, val) in [("Seed Z", seed_z), ("Seed Y", seed_y), ("Seed X", seed_x)] {
                let mut v = *val as i32;
                ui.horizontal(|ui| {
                    ui.label(format!("{label}:"));
                    if ui
                        .add(egui::DragValue::new(&mut v).speed(1.0).range(0..=i32::MAX))
                        .changed()
                    {
                        *val = v.max(0) as usize;
                    }
                });
            }
            ui.horizontal(|ui| {
                ui.label("Initial lower:");
                ui.add(egui::DragValue::new(initial_lower).speed(1.0));
            });
            ui.horizontal(|ui| {
                ui.label("Initial upper:");
                ui.add(egui::DragValue::new(initial_upper).speed(1.0));
            });
            ui.horizontal(|ui| {
                ui.label("Multiplier k:");
                ui.add(egui::Slider::new(multiplier, 0.5_f32..=10.0).step_by(0.1));
            });
            let mut mi = *max_iterations as i32;
            ui.horizontal(|ui| {
                ui.label("Max iterations:");
                if ui
                    .add(egui::Slider::new(&mut mi, 1..=100).step_by(1.0))
                    .changed()
                {
                    *max_iterations = mi.max(1) as u32;
                }
            });
            ui.label(
                egui::RichText::new(
                    "ITK ConfidenceConnectedImageFilter. Adaptive BFS: expands region using mean±k·σ statistics. Output: binary mask.",
                )
                .small(),
            );
            true
        }
        FilterKind::NeighborhoodConnected {
            seed_z,
            seed_y,
            seed_x,
            lower,
            upper,
            radius_z,
            radius_y,
            radius_x,
        } => {
            for (label, val) in [("Seed Z", seed_z), ("Seed Y", seed_y), ("Seed X", seed_x)] {
                let mut v = *val as i32;
                ui.horizontal(|ui| {
                    ui.label(format!("{label}:"));
                    if ui
                        .add(egui::DragValue::new(&mut v).speed(1.0).range(0..=i32::MAX))
                        .changed()
                    {
                        *val = v.max(0) as usize;
                    }
                });
            }
            ui.horizontal(|ui| {
                ui.label("Lower:");
                ui.add(egui::DragValue::new(lower).speed(1.0));
            });
            ui.horizontal(|ui| {
                ui.label("Upper:");
                ui.add(egui::DragValue::new(upper).speed(1.0));
            });
            for (label, val) in [
                ("Radius Z", radius_z),
                ("Radius Y", radius_y),
                ("Radius X", radius_x),
            ] {
                let mut v = *val as i32;
                ui.horizontal(|ui| {
                    ui.label(format!("{label}:"));
                    if ui
                        .add(egui::DragValue::new(&mut v).speed(1.0).range(0..=5))
                        .changed()
                    {
                        *val = v.clamp(0, 5) as usize;
                    }
                });
            }
            ui.label(
                egui::RichText::new(
                    "ITK NeighborhoodConnectedImageFilter. BFS where all voxels in candidate neighborhood satisfy [lower,upper]. Output: binary mask.",
                )
                .small(),
            );
            true
        }
        _ => false,
    }
}
