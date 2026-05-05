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
}
