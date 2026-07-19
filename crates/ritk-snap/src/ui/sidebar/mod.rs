//! Series browser and DICOM metadata panel.
//!
//! # Overview
//!
//! Three-tab panel:
//! - **"Series" tab**: collapsible Patient â†’ Study â†’ Series tree; clicking a
//!   series entry requests it be loaded.
//! - **"Tags" tab**: scrollable table of DICOM metadata for the currently
//!   loaded volume.
//! - **"PET SUV" tab**: PET quantification panel with SUVbw readouts and
//!   acquisition parameters.
//!
//! # Borrow strategy
//!
//! [`SidebarPanel`] holds short-lived references into the caller's state.
//! `show_series_tab` copies the `&'a SeriesTree` (a `Copy` reference), clones
//! the current selection once, and updates `selected_path` only after the
//! `ScrollArea` closure has released all other borrows. This avoids
//! simultaneous mutable and immutable borrows of the same field inside a
//! closure.

use egui::{CollapsingHeader, ScrollArea, Ui};

use crate::dicom::metadata_table::build_metadata_rows;
use crate::dicom::series_tree::SeriesTree;
use crate::LoadedVolume;

// â”€â”€ SidebarTab â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Active tab in the sidebar panel.
///
/// Governs which of the three content areas is displayed when
/// [`SidebarPanel::show`] is called.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum SidebarTab {
    /// Shows the collapsible Patient â†’ Study â†’ Series browser tree.
    #[default]
    Series,
    /// Shows the scrollable DICOM metadata table for the loaded volume.
    Metadata,
    /// Shows the PET SUV quantification panel (only meaningful for PT modality).
    PetSuv,
}

// â”€â”€ SidebarPanel â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Three-tab sidebar panel combining a DICOM series browser, a metadata table,
/// and a PET SUV quantification panel.
///
/// Constructed via [`SidebarPanel::new`] and consumed by a single call to
/// [`SidebarPanel::show`], which returns the folder path of the series the
/// user selected (if any).
///
/// # Lifetime
///
/// The lifetime `'a` covers all references:
/// - `tree` â€” the series hierarchy model (immutable).
/// - `selected_path` â€” the currently highlighted series folder (mutable).
/// - `active_tab` â€” which tab is displayed (mutable).
/// - `metadata_volume` â€” the currently loaded volume for the Tags tab
///   (shared, optional).
pub struct SidebarPanel<'a> {
    /// Hierarchical series tree to render in the Series tab.
    pub tree: &'a SeriesTree<'static>,
    /// The series folder path that is currently selected/highlighted.
    pub selected_path: &'a mut Option<std::path::PathBuf>,
    /// Which tab is currently active.
    pub active_tab: &'a mut SidebarTab,
    /// Volume whose metadata populates the Tags tab; `None` when no volume is
    /// loaded.
    pub metadata_volume: Option<&'a LoadedVolume>,
    /// SUVbw at the pointer position (for PET SUV tab).
    pub pointer_suv: Option<f32>,
    /// SUVbw at the linked-cursor position (for PET SUV tab).
    pub cursor_suv: Option<f32>,
}

impl<'a> SidebarPanel<'a> {
    /// Construct the panel from borrowed application state.
    ///
    /// All parameters map directly to the corresponding public fields.
    pub fn new(
        tree: &'a SeriesTree<'static>,
        selected_path: &'a mut Option<std::path::PathBuf>,
        active_tab: &'a mut SidebarTab,
        metadata_volume: Option<&'a LoadedVolume>,
        pointer_suv: Option<f32>,
        cursor_suv: Option<f32>,
    ) -> Self {
        Self {
            tree,
            selected_path,
            active_tab,
            metadata_volume,
            pointer_suv,
            cursor_suv,
        }
    }

    /// Construct the panel with an explicit tag search string (backwards-compat alias).
    pub fn with_tag_search(
        tree: &'a SeriesTree<'static>,
        selected_path: &'a mut Option<std::path::PathBuf>,
        active_tab: &'a mut SidebarTab,
        metadata_volume: Option<&'a LoadedVolume>,
        _tag_search: Option<&'a mut String>,
        pointer_suv: Option<f32>,
        cursor_suv: Option<f32>,
    ) -> Self {
        Self::new(
            tree,
            selected_path,
            active_tab,
            metadata_volume,
            pointer_suv,
            cursor_suv,
        )
    }

    /// Render the panel into `ui`.
    ///
    /// Draws the tab bar at the top, then delegates to the active tab handler.
    ///
    /// # Returns
    ///
    /// `Some(folder_path)` when the user clicks a series entry in the Series
    /// tab, requesting that series to be loaded. `None` on all other frames.
    pub fn show(&mut self, ui: &mut Ui) -> Option<std::path::PathBuf> {
        // â”€â”€ Tab selector row â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        ui.horizontal(|ui| {
            if ui
                .selectable_label(*self.active_tab == SidebarTab::Series, "Series")
                .clicked()
            {
                *self.active_tab = SidebarTab::Series;
            }
            if ui
                .selectable_label(*self.active_tab == SidebarTab::Metadata, "Tags")
                .clicked()
            {
                *self.active_tab = SidebarTab::Metadata;
            }
            if ui
                .selectable_label(*self.active_tab == SidebarTab::PetSuv, "PET SUV")
                .clicked()
            {
                *self.active_tab = SidebarTab::PetSuv;
            }
        });
        ui.separator();
        match *self.active_tab {
            SidebarTab::Series => self.show_series_tab(ui),
            SidebarTab::Metadata => {
                self.show_metadata_tab(ui);
                None
            }
            SidebarTab::PetSuv => {
                self.show_pet_suv_tab(ui);
                None
            }
        }
    }

    // â”€â”€ Series tab â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Render the Patient â†’ Study â†’ Series collapsible hierarchy.
    ///
    /// # Borrow invariant
    ///
    /// `tree` is copied out of `self` (it is a `Copy` reference) before the
    /// `ScrollArea` closure begins; `selected_path` is not accessed inside the
    /// closure body. After the closure exits, a single write to
    /// `*self.selected_path` records the newly selected path when the user
    /// clicked an entry.
    fn show_series_tab(&mut self, ui: &mut Ui) -> Option<std::path::PathBuf> {
        // Copy the &'a SeriesTree reference â€” &T: Copy, releases borrow on self.
        let tree: &'a SeriesTree<'static> = self.tree;
        // Clone current selection for inside-closure comparison without holding
        // a borrow on self.selected_path during the closure.
        let current_path: Option<std::path::PathBuf> = self.selected_path.as_ref().cloned();
        let mut new_selection: Option<std::path::PathBuf> = None;

        ScrollArea::vertical()
            .id_source("series_scroll")
            .show(ui, |ui| {
                if tree.patients.is_empty() {
                    ui.label("No series found.");
                    ui.label("Use File â†’ Open DICOM folder to scan a directory.");
                    return;
                }
                for (patient_idx, patient) in tree.patients.iter().enumerate() {
                    // â”€â”€ Patient node â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                    let patient_label = if patient.patient_name.is_empty() {
                        format!("ðŸ‘¤ (anonymous) [{}]", patient.patient_id)
                    } else if patient.patient_id.is_empty() {
                        format!("ðŸ‘¤ {}", patient.patient_name)
                    } else {
                        format!("ðŸ‘¤ {} [{}]", patient.patient_name, patient.patient_id)
                    };

                    CollapsingHeader::new(patient_label)
                        .id_source(("patient", patient_idx, patient.patient_id.as_ref()))
                        .default_open(true)
                        .show(ui, |ui| {
                            for (study_idx, study) in patient.studies.iter().enumerate() {
                                // â”€â”€ Study node â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                                let study_label = match (&study.study_date, &study.study_uid) {
                                    (Some(date), _) => format!("ðŸ“… {date}"),
                                    (None, Some(uid)) => format!("ðŸ“… UID:{uid}"),
                                    (None, None) => "ðŸ“… (unknown date)".to_string(),
                                };

                                CollapsingHeader::new(study_label)
                                    .id_source((
                                        "study",
                                        patient_idx,
                                        study_idx,
                                        study.study_uid.as_deref().unwrap_or(""),
                                        study.study_date.as_deref().unwrap_or(""),
                                    ))
                                    .default_open(true)
                                    .show(ui, |ui| {
                                        for (series_idx, series) in study.series.iter().enumerate()
                                        {
                                            // â”€â”€ Series entry â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                                            let is_selected = current_path
                                                .as_ref()
                                                .map(|p| p == series.folder.as_ref())
                                                .unwrap_or(false);
                                            let hover_text = format!(
                                                "Folder: {}\nUID: {}\nSlices: {}",
                                                series.folder.as_ref().display(),
                                                series.series_uid,
                                                series.num_slices,
                                            );
                                            let series_label = series.display_label();
                                            let resp = ui
                                                .push_id(
                                                    (
                                                        "series",
                                                        patient_idx,
                                                        study_idx,
                                                        series_idx,
                                                        series.series_uid.as_ref(),
                                                        series.folder.as_ref().to_string_lossy(),
                                                    ),
                                                    |ui| {
                                                        ui.selectable_label(
                                                            is_selected,
                                                            series_label,
                                                        )
                                                    },
                                                )
                                                .inner
                                                .on_hover_text(hover_text);
                                            if resp.clicked() {
                                                new_selection =
                                                    Some(series.folder.as_ref().to_path_buf());
                                            }
                                        }
                                    });
                            }
                        });
                }
            });

        // Update selected_path after the closure has released all other borrows.
        if let Some(ref path) = new_selection {
            *self.selected_path = Some(path.clone());
        }
        new_selection
    }

    // â”€â”€ Metadata (Tags) tab â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Render a scrollable table of DICOM metadata for the loaded volume.
    fn show_metadata_tab(&self, ui: &mut Ui) {
        // â”€â”€ Tag search input â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        let search_id = egui::Id::new("dicom_tag_search_filter");
        let mut search: String = ui.data(|d| d.get_temp(search_id).unwrap_or_default());
        ui.horizontal(|ui| {
            ui.label("ðŸ”");
            ui.text_edit_singleline(&mut search)
                .on_hover_text("Filter tags by keyword or tag hex code (case-insensitive)");
            if ui.small_button("âœ–").clicked() {
                search.clear();
            }
        });
        ui.data_mut(|d| d.insert_temp(search_id, search.clone()));
        let needle = search.to_lowercase();

        ScrollArea::vertical()
            .id_source("metadata_scroll")
            .show(ui, |ui| {
                let Some(vol) = self.metadata_volume else {
                    ui.label("No volume loaded.");
                    return;
                };

                let [depth, rows, cols] = vol.shape;
                let [dz, dy, dx] = vol.spacing;
                let [ox, oy, oz] = vol.origin;

                egui::Grid::new("meta_grid")
                    .num_columns(2)
                    .spacing([8.0, 4.0])
                    .striped(true)
                    .show(ui, |ui| {
                        let row = |ui: &mut Ui, key: &str, val: &str| {
                            ui.label(key);
                            ui.label(val);
                            ui.end_row();
                        };
                        row(ui, "Patient:", vol.patient_name.as_deref().unwrap_or("â€”"));
                        row(ui, "ID:", vol.patient_id.as_deref().unwrap_or("â€”"));
                        row(ui, "Date:", vol.study_date.as_deref().unwrap_or("â€”"));
                        row(ui, "Modality:", vol.modality.as_deref().unwrap_or("â€”"));
                        row(
                            ui,
                            "Series:",
                            vol.series_description.as_deref().unwrap_or("â€”"),
                        );
                        row(ui, "Dimensions:", &format!("{depth} Ã— {rows} Ã— {cols}"));
                        row(ui, "Spacing:", &format!("{dz:.3} Ã— {dy:.3} Ã— {dx:.3} mm"));
                        row(ui, "Origin:", &format!("{ox:.2}, {oy:.2}, {oz:.2}"));
                        if let Some(src) = &vol.source {
                            row(ui, "Source:", &src.to_string_lossy());
                        }
                    });

                let Some(meta) = &vol.metadata else {
                    ui.separator();
                    ui.label("No DICOM metadata attached.");
                    return;
                };

                ui.separator();
                ui.label("DICOM Tags");

                let all_rows = build_metadata_rows(meta);
                let total = all_rows.len();
                let filtered: Vec<_> = if needle.is_empty() {
                    all_rows
                } else {
                    all_rows
                        .into_iter()
                        .filter(|r| {
                            r.keyword.to_lowercase().contains(&needle)
                                || r.tag.to_lowercase().contains(&needle)
                                || r.value.to_lowercase().contains(&needle)
                        })
                        .collect()
                };

                let match_label = if needle.is_empty() {
                    format!("{total} tags")
                } else {
                    format!("{} of {total} match", filtered.len())
                };
                ui.label(egui::RichText::new(match_label).small().weak());

                egui::Grid::new("dicom_tag_grid")
                    .num_columns(5)
                    .spacing([8.0, 4.0])
                    .striped(true)
                    .show(ui, |ui| {
                        ui.label("Scope");
                        ui.label("Tag");
                        ui.label("Keyword");
                        ui.label("VR");
                        ui.label("Value");
                        ui.end_row();
                        for row in filtered {
                            ui.label(row.scope.label());
                            ui.label(row.tag);
                            ui.label(row.keyword);
                            ui.label(row.vr);
                            ui.label(row.value);
                            ui.end_row();
                        }
                    });
            });
    }

    // â”€â”€ PET SUV tab â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Render the PET SUV quantification panel.
    ///
    /// Delegates to [`crate::ui::pet_suv_panel::draw_pet_suv_panel`] with
    /// parameters extracted from the loaded volume.
    fn show_pet_suv_tab(&self, ui: &mut Ui) {
        let (modality, patient_weight_kg, injected_dose_bq, half_life_s, decay_correction) =
            match self.metadata_volume {
                Some(vol) => (
                    vol.modality.as_deref(),
                    vol.patient_weight_kg,
                    vol.injected_dose_bq,
                    vol.radionuclide_half_life_s,
                    vol.decay_correction.as_deref(),
                ),
                None => (None, None, None, None, None),
            };
        crate::ui::pet_suv_panel::draw_pet_suv_panel(
            ui,
            modality,
            self.pointer_suv,
            self.cursor_suv,
            patient_weight_kg,
            injected_dose_bq,
            half_life_s,
            decay_correction,
        );
    }
}

#[cfg(test)]
mod tests;
