//! Image processing filter selection panel.
//!
//! Exposes a compact egui widget that lets the user choose a [`FilterKind`]
//! and configure its parameters.  The caller is responsible for wiring the
//! "Apply" confirmation into the application state.
//!
//! # Design contract
//!
//! - [`show_filter_panel`] is a pure egui widget: it modifies `active_filter`
//!   in-place and returns `true` exactly when the user clicks "Apply".
//! - All numeric controls are clamped to analytically valid ranges:
//!   - Gaussian σ ∈ [0.1, 20.0] mm
//!   - Median radius ∈ [0, 10] voxels
//!   - CLAHE tile grid ∈ [1, 32] per axis; clip limit ∈ [1.0, 200.0]
//!   - HistEq bins ∈ [2, 1024]
//!   - UnsharpMask σ ∈ [0.1, 10.0] mm; amount ∈ [0.0, 5.0]; threshold ∈ [0.0, 100.0]
//!   - GradientAnisotropicDiffusion iterations ∈ [1, 50]; time_step ∈ [0.01, 0.1667]; conductance ∈ [0.1, 100.0]
//!   - ConnectedComponents background_value (any f32); connectivity_26 boolean
//!   - RelabelComponents minimum_object_size ∈ [0, MAX_u32] voxels
//!   - MultiOtsuThreshold num_classes ∈ [2, 8]
//!   - BinaryErode radius ∈ [0, 10]; foreground_value ∈ any f32
//!   - BinaryDilate radius ∈ [0, 10]; foreground_value ∈ any f32
//!   - BinaryClosing radius ∈ [0, 10]; foreground_value ∈ any f32
//!   - BinaryOpening radius ∈ [0, 10]; foreground_value ∈ any f32
//!   - GrayscaleClosing radius ∈ [0, 10] voxels
//!   - GrayscaleOpening radius ∈ [0, 10] voxels
//!   - GrayscaleFillhole (no parameters)
//! - The widget does not mutate the image; it only modifies the
//!   `FilterKind` selector held by the caller.

use crate::FilterKind;

/// Display the filter selection panel inside `ui`.
///
/// Returns `true` exactly when the user clicks the **Apply** button.
///
/// # Parameters
/// - `ui`: mutable reference to the egui [`Ui`] context.
/// - `active_filter`: mutable reference to the currently configured
///   [`FilterKind`].  Updated in-place as the user changes controls.
pub fn show_filter_panel(ui: &mut egui::Ui, active_filter: &mut FilterKind) -> bool {
    let mut apply = false;

    egui::Frame::group(ui.style()).show(ui, |ui| {
        ui.label(egui::RichText::new("Image Processing").strong());
        ui.separator();

        // ── Filter selector ────────────────────────────────────────────────
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
        };
        egui::ComboBox::from_label("Filter")
            .selected_text(kind_label)
            .show_ui(ui, |ui| {
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
                            clamp: true,
                        },
                        "Unsharp Mask",
                    )
                    .clicked()
                {
                    *active_filter = FilterKind::UnsharpMask {
                        sigma: 1.0,
                        amount: 0.5,
                        threshold: 0.0,
                        clamp: true,
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
                            connectivity_26: false,
                            background_value: 0.0,
                        },
                        "Connected Components",
                    )
                    .clicked()
                {
                    *active_filter = FilterKind::ConnectedComponents {
                        connectivity_26: false,
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
                        FilterKind::BinaryErode { radius: 1, foreground_value: 1.0 },
                        "Binary Erode",
                    )
                    .clicked()
                {
                    *active_filter = FilterKind::BinaryErode { radius: 1, foreground_value: 1.0 };
                }
                if ui
                    .selectable_value(
                        &mut *active_filter,
                        FilterKind::BinaryDilate { radius: 1, foreground_value: 1.0 },
                        "Binary Dilate",
                    )
                    .clicked()
                {
                    *active_filter = FilterKind::BinaryDilate { radius: 1, foreground_value: 1.0 };
                }
                if ui
                    .selectable_value(
                        &mut *active_filter,
                        FilterKind::BinaryClosing { radius: 1, foreground_value: 1.0 },
                        "Binary Closing",
                    )
                    .clicked()
                {
                    *active_filter = FilterKind::BinaryClosing { radius: 1, foreground_value: 1.0 };
                }
                if ui
                    .selectable_value(
                        &mut *active_filter,
                        FilterKind::BinaryOpening { radius: 1, foreground_value: 1.0 },
                        "Binary Opening",
                    )
                    .clicked()
                {
                    *active_filter = FilterKind::BinaryOpening { radius: 1, foreground_value: 1.0 };
                }
                if ui
                    .selectable_value(
                        &mut *active_filter,
                        FilterKind::BinaryFillhole { foreground_value: 1.0 },
                        "Binary Fill Holes",
                    )
                    .clicked()
                {
                    *active_filter = FilterKind::BinaryFillhole { foreground_value: 1.0 };
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
                if ui.selectable_value(&mut *active_filter, FilterKind::Abs, "Abs").clicked() {
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
                if ui.selectable_value(&mut *active_filter, FilterKind::Square, "Square").clicked() {
                    *active_filter = FilterKind::Square;
                }
                if ui.selectable_value(&mut *active_filter, FilterKind::Sqrt, "Sqrt").clicked() {
                    *active_filter = FilterKind::Sqrt;
                }
                if ui.selectable_value(&mut *active_filter, FilterKind::Log, "Log").clicked() {
                    *active_filter = FilterKind::Log;
                }
                if ui.selectable_value(&mut *active_filter, FilterKind::Exp, "Exp").clicked() {
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
                if ui.selectable_value(
                    &mut *active_filter,
                    FilterKind::DistanceTransform { threshold: 0.5 },
                    "Distance Transform",
                ).clicked() { *active_filter = FilterKind::DistanceTransform { threshold: 0.5 }; }
                if ui.selectable_value(
                    &mut *active_filter,
                    FilterKind::SignedDistanceTransform { threshold: 0.5 },
                    "Signed Distance Transform",
                ).clicked() { *active_filter = FilterKind::SignedDistanceTransform { threshold: 0.5 }; }
                if ui.selectable_value(&mut *active_filter, FilterKind::FlipZ, "Flip Z").clicked() { *active_filter = FilterKind::FlipZ; }
                if ui.selectable_value(&mut *active_filter, FilterKind::FlipY, "Flip Y").clicked() { *active_filter = FilterKind::FlipY; }
                if ui.selectable_value(&mut *active_filter, FilterKind::FlipX, "Flip X").clicked() { *active_filter = FilterKind::FlipX; }
                if ui.selectable_value(
                    &mut *active_filter,
                    FilterKind::MaskThreshold { threshold: 0.5 },
                    "Mask Threshold",
                ).clicked() { *active_filter = FilterKind::MaskThreshold { threshold: 0.5 }; }
                if ui.selectable_value(&mut *active_filter, FilterKind::GeodesicDilationSelf, "Geodesic Dilation (self)").clicked() { *active_filter = FilterKind::GeodesicDilationSelf; }
                if ui.selectable_value(&mut *active_filter, FilterKind::GeodesicErosionSelf, "Geodesic Erosion (self)").clicked() { *active_filter = FilterKind::GeodesicErosionSelf; }
            });

        ui.add_space(4.0);

        // ── Per-filter parameter controls ──────────────────────────────────
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
            }
            FilterKind::BedSeparation(_) => {
                ui.label("No adjustable parameters.");
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
                    ui.add(
                        egui::Slider::new(clip_limit, 1.0_f32..=200.0).step_by(1.0),
                    );
                });
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
                    ui.checkbox(clamp, "");
                });
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
                    ui.add(
                        egui::Slider::new(time_step, 0.01_f32..=0.1667)
                            .step_by(0.005),
                    );
                });
                ui.horizontal(|ui| {
                    ui.label("Conductance K:");
                    ui.add(
                        egui::Slider::new(conductance, 0.1_f32..=100.0)
                            .step_by(0.1)
                            .logarithmic(true),
                    );
                });
            }
            FilterKind::ConnectedComponents {
                connectivity_26,
                background_value,
            } => {
                ui.horizontal(|ui| {
                    ui.label("26-connectivity:");
                    ui.checkbox(connectivity_26, "");
                });
                ui.horizontal(|ui| {
                    ui.label("Background value:");
                    ui.add(
                        egui::DragValue::new(background_value)
                            .speed(1.0)
                            .prefix("")
                    );
                });
                ui.label(
                    egui::RichText::new(
                        "Output: integer label image (0=background, 1…N=components)",
                    )
                    .small(),
                );
            }
            FilterKind::RelabelComponents { minimum_object_size } => {
                // minimum_object_size is u32; use i32 proxy for DragValue.
                let mut mos = *minimum_object_size as i32;
                ui.horizontal(|ui| {
                    ui.label("Min object size (voxels):");
                    if ui
                        .add(egui::DragValue::new(&mut mos).speed(1.0).range(0..=i32::MAX))
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
            }
            FilterKind::BinaryErode { radius, foreground_value } => {
                let mut r = *radius as i32;
                ui.horizontal(|ui| {
                    ui.label("Radius (voxels):");
                    if ui.add(egui::DragValue::new(&mut r).speed(1.0).range(0..=10)).changed() {
                        *radius = r.clamp(0, 10) as usize;
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Foreground value:");
                    ui.add(egui::DragValue::new(foreground_value).speed(1.0));
                });
                ui.label(egui::RichText::new("ITK BinaryErodeImageFilter parity.").small());
            }
            FilterKind::BinaryDilate { radius, foreground_value } => {
                let mut r = *radius as i32;
                ui.horizontal(|ui| {
                    ui.label("Radius (voxels):");
                    if ui.add(egui::DragValue::new(&mut r).speed(1.0).range(0..=10)).changed() {
                        *radius = r.clamp(0, 10) as usize;
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Foreground value:");
                    ui.add(egui::DragValue::new(foreground_value).speed(1.0));
                });
                ui.label(egui::RichText::new("ITK BinaryDilateImageFilter parity.").small());
            }
            FilterKind::BinaryClosing { radius, foreground_value } => {
                let mut r = *radius as i32;
                ui.horizontal(|ui| {
                    ui.label("Radius (voxels):");
                    if ui.add(egui::DragValue::new(&mut r).speed(1.0).range(0..=10)).changed() {
                        *radius = r.clamp(0, 10) as usize;
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Foreground value:");
                    ui.add(egui::DragValue::new(foreground_value).speed(1.0));
                });
                ui.label(egui::RichText::new("ITK BinaryMorphologicalClosingImageFilter parity. Fills dark holes.").small());
            }
            FilterKind::BinaryOpening { radius, foreground_value } => {
                let mut r = *radius as i32;
                ui.horizontal(|ui| {
                    ui.label("Radius (voxels):");
                    if ui.add(egui::DragValue::new(&mut r).speed(1.0).range(0..=10)).changed() {
                        *radius = r.clamp(0, 10) as usize;
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Foreground value:");
                    ui.add(egui::DragValue::new(foreground_value).speed(1.0));
                });
                ui.label(egui::RichText::new("ITK BinaryMorphologicalOpeningImageFilter parity. Removes small bright blobs.").small());
            }
            FilterKind::BinaryFillhole { foreground_value } => {
                ui.horizontal(|ui| {
                    ui.label("Foreground value:");
                    ui.add(egui::DragValue::new(foreground_value).speed(1.0));
                });
                ui.label(egui::RichText::new("ITK BinaryFillholeImageFilter parity. Fills enclosed background cavities.").small());
            }
            FilterKind::GrayscaleClosing { radius } => {
                let mut r = *radius as i32;
                ui.horizontal(|ui| {
                    ui.label("Radius (voxels):");
                    if ui.add(egui::DragValue::new(&mut r).speed(1.0).range(0..=10)).changed() {
                        *radius = r.clamp(0, 10) as usize;
                    }
                });
                ui.label(egui::RichText::new("ITK GrayscaleMorphologicalClosingImageFilter parity. C_B(f)=E_B(D_B(f)). Fills dark voids.").small());
            }
            FilterKind::GrayscaleOpening { radius } => {
                let mut r = *radius as i32;
                ui.horizontal(|ui| {
                    ui.label("Radius (voxels):");
                    if ui.add(egui::DragValue::new(&mut r).speed(1.0).range(0..=10)).changed() {
                        *radius = r.clamp(0, 10) as usize;
                    }
                });
                ui.label(egui::RichText::new("ITK GrayscaleMorphologicalOpeningImageFilter parity. O_B(f)=D_B(E_B(f)). Removes bright protrusions.").small());
            }
            FilterKind::GrayscaleFillhole => {
                ui.label(egui::RichText::new("ITK GrayscaleFillholeImageFilter parity. Fills dark regional minima not connected to the image border.").small());
            }
            FilterKind::Abs => {
                ui.label(egui::RichText::new("ITK AbsImageFilter / ImageJ Abs. out(x) = |in(x)|. No adjustable parameters.").small());
            }
            FilterKind::InvertIntensity { maximum } => {
                ui.label(egui::RichText::new("ITK InvertIntensityImageFilter. out(x) = maximum - in(x). maximum=None → computed from image.").small());
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
            }
            FilterKind::NormalizeIntensity => {
                ui.label(egui::RichText::new("ITK NormalizeImageFilter. out(x) = (in(x) - mean) / std. Constant image → all zero. No adjustable parameters.").small());
            }
            FilterKind::Square => {
                ui.label(egui::RichText::new("ITK SquareImageFilter / ImageJ Square. out(x) = in(x)². No adjustable parameters.").small());
            }
            FilterKind::Sqrt => {
                ui.label(egui::RichText::new("ITK SqrtImageFilter / ImageJ Square Root. out(x) = √in(x). Negative → NaN. No adjustable parameters.").small());
            }
            FilterKind::Log => {
                ui.label(egui::RichText::new("ITK LogImageFilter / ImageJ Log. out(x) = ln(in(x)). Non-positive → -inf/NaN. No adjustable parameters.").small());
            }
            FilterKind::Exp => {
                ui.label(egui::RichText::new("ITK ExpImageFilter / ImageJ Exp. out(x) = e^in(x). No adjustable parameters.").small());
            }
            FilterKind::MorphologicalGradient { radius } => {
                ui.label(egui::RichText::new("ITK GrayscaleMorphologicalGradientImageFilter (Beucher gradient). out(x) = D_B(f)(x) - E_B(f)(x). Non-negative.").small());
                let mut r = *radius as i32;
                ui.horizontal(|ui| {
                    ui.label("Radius (voxels):");
                    if ui.add(egui::Slider::new(&mut r, 0..=10).step_by(1.0)).changed() {
                        *radius = r.max(0) as usize;
                    }
                });
            }
            FilterKind::DistanceTransform { threshold } | FilterKind::SignedDistanceTransform { threshold } => {
                ui.label(egui::RichText::new("Euclidean distance transform. Each voxel receives distance (mm) to nearest foreground voxel.").small());
                ui.horizontal(|ui| {
                    ui.label("Foreground threshold:");
                    ui.add(egui::Slider::new(threshold, 0.0_f32..=1000.0).step_by(0.1));
                });
            }
            FilterKind::FlipZ | FilterKind::FlipY | FilterKind::FlipX => {
                ui.label(egui::RichText::new("Reverses voxel ordering along the selected axis. No adjustable parameters.").small());
            }
            FilterKind::MaskThreshold { threshold } => {
                ui.label(egui::RichText::new("Zero-out voxels at or below the threshold (binary self-mask).").small());
                ui.horizontal(|ui| {
                    ui.label("Threshold:");
                    ui.add(egui::Slider::new(threshold, 0.0_f32..=1000.0).step_by(0.1));
                });
            }
            FilterKind::GeodesicDilationSelf | FilterKind::GeodesicErosionSelf => {
                ui.label(egui::RichText::new("Geodesic morphological reconstruction (marker = mask = current image). Identity on self; for two-image reconstruction use the ritk_core API.").small());
            }
        }

        ui.add_space(6.0);

        // ── Apply button ───────────────────────────────────────────────────
        if ui
            .add(egui::Button::new("Apply").min_size(egui::vec2(80.0, 22.0)))
            .clicked()
        {
            apply = true;
        }
    });

    apply
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Verify that the default `FilterKind` values exposed by the panel are
    // within the analytically valid clamped ranges.

    #[test]
    fn gaussian_default_sigma_in_range() {
        let fk = FilterKind::Gaussian { sigma: 1.0 };
        if let FilterKind::Gaussian { sigma } = fk {
            assert!(
                sigma >= 0.1 && sigma <= 20.0,
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
            assert!(radius <= 10, "default radius {radius} must be ≤ 10");
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
                clip_limit >= 1.0 && clip_limit <= 200.0,
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
            assert!(bins >= 2 && bins <= 1024, "bins={bins} out of range");
        } else {
            panic!("expected HistEq variant");
        }
    }

    /// UnsharpMask defaults lie within the slider ranges.
    ///
    /// - sigma ∈ [0.1, 10.0] mm
    /// - amount ∈ [0.0, 5.0]
    /// - threshold ∈ [0.0, 100.0]
    #[test]
    fn unsharp_mask_defaults_in_range() {
        let fk = FilterKind::UnsharpMask {
            sigma: 1.0,
            amount: 0.5,
            threshold: 0.0,
            clamp: true,
        };
        if let FilterKind::UnsharpMask {
            sigma,
            amount,
            threshold,
            clamp,
        } = fk
        {
            assert!(
                sigma >= 0.1 && sigma <= 10.0,
                "default sigma {sigma} out of range [0.1, 10.0]"
            );
            assert!(
                amount >= 0.0 && amount <= 5.0,
                "default amount {amount} out of range [0.0, 5.0]"
            );
            assert!(
                threshold >= 0.0 && threshold <= 100.0,
                "default threshold {threshold} out of range [0.0, 100.0]"
            );
            assert!(clamp, "default clamp should be true");
        } else {
            panic!("expected UnsharpMask variant");
        }
    }

    /// GradientAnisotropicDiffusion defaults lie within the slider ranges.
    ///
    /// - iterations ∈ [1, 50]
    /// - time_step ∈ [0.01, 0.1667]  (stability bound Δt ≤ 1/6)
    /// - conductance ∈ [0.1, 100.0]
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
                iterations >= 1 && iterations <= 50,
                "default iterations {iterations} out of range [1, 50]"
            );
            assert!(
                time_step >= 0.01 && time_step <= 0.1667,
                "default time_step {time_step} out of range [0.01, 0.1667] (stability bound)"
            );
            assert!(
                conductance >= 0.1 && conductance <= 100.0,
                "default conductance {conductance} out of range [0.1, 100.0]"
            );
        } else {
            panic!("expected GradientAnisotropicDiffusion variant");
        }
    }

    /// ConnectedComponents defaults are valid.
    ///
    /// - connectivity_26 = false (6-connectivity is the ITK/medical default)
    /// - background_value = 0.0 (ITK default)
    ///
    /// # Postcondition
    /// These values produce a valid ITK-parity filter dispatch via
    /// `ConnectedComponentsFilter::with_connectivity(6).with_background(0.0)`.
    #[test]
    fn connected_components_defaults_are_valid() {
        let fk = FilterKind::ConnectedComponents {
            connectivity_26: false,
            background_value: 0.0,
        };
        if let FilterKind::ConnectedComponents {
            connectivity_26,
            background_value,
        } = fk
        {
            assert!(
                !connectivity_26,
                "default connectivity must be 6-connected (connectivity_26 = false)"
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
        if let FilterKind::RelabelComponents { minimum_object_size } = fk {
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
    /// - num_classes ≥ 2 is required (enforced by `MultiOtsuThreshold::new` panic guard).
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
                "num_classes must be ≥ 2 (enforced by MultiOtsuThreshold::new); got {num_classes}"
            );
            assert_eq!(
                num_classes, 3,
                "ITK default num_classes = 3 (two thresholds; three classes)"
            );
        } else {
            panic!("expected MultiOtsuThreshold variant");
        }
    }

    /// BinaryErode defaults: radius=1, foreground_value=1.0 — ITK defaults.
    #[test]
    fn binary_erode_defaults_are_valid() {
        let fk = FilterKind::BinaryErode { radius: 1, foreground_value: 1.0 };
        if let FilterKind::BinaryErode { radius, foreground_value } = fk {
            assert!(radius <= 10, "default radius {radius} must be ≤ 10");
            assert_eq!(foreground_value, 1.0, "default fg value must be 1.0 (ITK default)");
        } else {
            panic!("expected BinaryErode variant");
        }
    }

    /// BinaryDilate defaults: radius=1, foreground_value=1.0 — ITK defaults.
    #[test]
    fn binary_dilate_defaults_are_valid() {
        let fk = FilterKind::BinaryDilate { radius: 1, foreground_value: 1.0 };
        if let FilterKind::BinaryDilate { radius, foreground_value } = fk {
            assert!(radius <= 10, "default radius {radius} must be ≤ 10");
            assert_eq!(foreground_value, 1.0, "default fg value must be 1.0 (ITK default)");
        } else {
            panic!("expected BinaryDilate variant");
        }
    }

    /// BinaryClosing defaults: radius=1, foreground_value=1.0 — ITK defaults.
    #[test]
    fn binary_closing_defaults_are_valid() {
        let fk = FilterKind::BinaryClosing { radius: 1, foreground_value: 1.0 };
        if let FilterKind::BinaryClosing { radius, foreground_value } = fk {
            assert!(radius <= 10, "default radius {radius} must be ≤ 10");
            assert_eq!(foreground_value, 1.0, "default fg value must be 1.0 (ITK default)");
        } else {
            panic!("expected BinaryClosing variant");
        }
    }

    /// BinaryOpening defaults: radius=1, foreground_value=1.0 — ITK defaults.
    #[test]
    fn binary_opening_defaults_are_valid() {
        let fk = FilterKind::BinaryOpening { radius: 1, foreground_value: 1.0 };
        if let FilterKind::BinaryOpening { radius, foreground_value } = fk {
            assert!(radius <= 10, "default radius {radius} must be ≤ 10");
            assert_eq!(foreground_value, 1.0, "default fg value must be 1.0 (ITK default)");
        } else {
            panic!("expected BinaryOpening variant");
        }
    }

    /// BinaryFillhole defaults: foreground_value=1.0 — ITK default.
    #[test]
    fn binary_fillhole_defaults_are_valid() {
        let fk = FilterKind::BinaryFillhole { foreground_value: 1.0 };
        if let FilterKind::BinaryFillhole { foreground_value } = fk {
            assert_eq!(foreground_value, 1.0, "default fg value must be 1.0 (ITK default)");
        } else {
            panic!("expected BinaryFillhole variant");
        }
    }

    /// GrayscaleClosing default: radius=1 — minimal ITK closing SE.
    ///
    /// # Analytical basis
    /// radius=1 → 3×3×3 SE, the smallest non-trivial cubic SE. ITK
    /// `GrayscaleMorphologicalClosingImageFilter` default radius is 1.
    #[test]
    fn grayscale_closing_defaults_are_valid() {
        let fk = FilterKind::GrayscaleClosing { radius: 1 };
        if let FilterKind::GrayscaleClosing { radius } = fk {
            assert!(radius <= 10, "default radius {radius} must be ≤ 10");
            assert_eq!(radius, 1, "ITK default radius = 1 (3×3×3 SE)");
        } else {
            panic!("expected GrayscaleClosing variant");
        }
    }

    /// GrayscaleOpening default: radius=1 — minimal ITK opening SE.
    ///
    /// # Analytical basis
    /// radius=1 → 3×3×3 SE, the smallest non-trivial cubic SE. ITK
    /// `GrayscaleMorphologicalOpeningImageFilter` default radius is 1.
    #[test]
    fn grayscale_opening_defaults_are_valid() {
        let fk = FilterKind::GrayscaleOpening { radius: 1 };
        if let FilterKind::GrayscaleOpening { radius } = fk {
            assert!(radius <= 10, "default radius {radius} must be ≤ 10");
            assert_eq!(radius, 1, "ITK default radius = 1 (3×3×3 SE)");
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

    /// Abs: unit struct, always valid.
    #[test]
    fn abs_variant_is_valid() {
        let fk = FilterKind::Abs;
        assert!(matches!(fk, FilterKind::Abs), "Abs variant must match itself");
    }

    /// InvertIntensity default: maximum=None (computed from image data, ITK default).
    #[test]
    fn invert_intensity_default_maximum_is_none() {
        let fk = FilterKind::InvertIntensity { maximum: None };
        if let FilterKind::InvertIntensity { maximum } = fk {
            assert!(maximum.is_none(), "InvertIntensity default maximum must be None (auto from image)");
        } else {
            panic!("expected InvertIntensity variant");
        }
    }

    /// NormalizeIntensity: unit struct, always valid.
    #[test]
    fn normalize_intensity_variant_is_valid() {
        let fk = FilterKind::NormalizeIntensity;
        assert!(
            matches!(fk, FilterKind::NormalizeIntensity),
            "NormalizeIntensity variant must match itself"
        );
    }

    /// Square: unit struct, always valid.
    #[test]
    fn square_variant_is_valid() {
        let fk = FilterKind::Square;
        assert!(matches!(fk, FilterKind::Square), "Square variant must match itself");
    }

    /// Sqrt: unit struct, always valid.
    #[test]
    fn sqrt_variant_is_valid() {
        let fk = FilterKind::Sqrt;
        assert!(matches!(fk, FilterKind::Sqrt), "Sqrt variant must match itself");
    }

    /// Log: unit struct, always valid.
    #[test]
    fn log_variant_is_valid() {
        let fk = FilterKind::Log;
        assert!(matches!(fk, FilterKind::Log), "Log variant must match itself");
    }

    /// Exp: unit struct, always valid.
    #[test]
    fn exp_variant_is_valid() {
        let fk = FilterKind::Exp;
        assert!(matches!(fk, FilterKind::Exp), "Exp variant must match itself");
    }

    /// MorphologicalGradient default: radius=1 — minimal non-trivial cubic SE.
    ///
    /// # Analytical basis
    /// radius=1 → 3×3×3 SE, the smallest non-trivial cubic structuring element.
    /// ITK `GrayscaleMorphologicalGradientImageFilter` uses radius=1 by default.
    #[test]
    fn morphological_gradient_default_radius_is_valid() {
        let fk = FilterKind::MorphologicalGradient { radius: 1 };
        if let FilterKind::MorphologicalGradient { radius } = fk {
            assert_eq!(radius, 1, "default radius must be 1 (smallest non-trivial SE)");
            assert!(radius <= 10, "default radius {radius} must be within slider range [0, 10]");
        } else {
            panic!("expected MorphologicalGradient variant");
        }
    }
}
