//! Series browser and DICOM metadata panel.
//!
//! # Overview
//!
//! Two-tab panel:
//! - **"Series" tab**: collapsible Patient → Study → Series tree; clicking a
//!   series entry requests it be loaded.
//! - **"Tags" tab**: scrollable table of DICOM metadata for the currently
//!   loaded volume.
//!
//! # Borrow strategy
//!
//! [`SidebarPanel`] holds four short-lived references into the caller's state.
//! `show_series_tab` copies the `&'a SeriesTree` (a `Copy` reference), clones
//! the current selection once, and updates `selected_path` only after the
//! `ScrollArea` closure has released all other borrows.  This avoids
//! simultaneous mutable and immutable borrows of the same field inside a
//! closure.

use egui::{CollapsingHeader, ScrollArea, Ui};

use crate::dicom::metadata_table::build_metadata_rows;
use crate::dicom::series_tree::SeriesTree;
use crate::LoadedVolume;

// ── SidebarTab ────────────────────────────────────────────────────────────────

/// Active tab in the sidebar panel.
///
/// Governs which of the two content areas is displayed when
/// [`SidebarPanel::show`] is called.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum SidebarTab {
    /// Shows the collapsible Patient → Study → Series browser tree.
    #[default]
    Series,
    /// Shows the scrollable DICOM metadata table for the loaded volume.
    Metadata,
}

// ── SidebarPanel ──────────────────────────────────────────────────────────────

/// Two-tab sidebar panel combining a DICOM series browser and a metadata table.
///
/// Constructed via [`SidebarPanel::new`] and consumed by a single call to
/// [`SidebarPanel::show`], which returns the folder path of the series the
/// user selected (if any).
///
/// # Lifetime
///
/// The lifetime `'a` covers all four references:
/// - `tree` — the series hierarchy model (immutable).
/// - `selected_path` — the currently highlighted series folder (mutable).
/// - `active_tab` — which tab is displayed (mutable).
/// - `metadata_volume` — the currently loaded volume for the Tags tab
///   (shared, optional).
pub struct SidebarPanel<'a> {
    /// Hierarchical series tree to render in the Series tab.
    pub tree: &'a SeriesTree,
    /// The series folder path that is currently selected/highlighted.
    pub selected_path: &'a mut Option<std::path::PathBuf>,
    /// Which tab is currently active.
    pub active_tab: &'a mut SidebarTab,
    /// Volume whose metadata populates the Tags tab; `None` when no volume is
    /// loaded.
    pub metadata_volume: Option<&'a LoadedVolume>,
}

impl<'a> SidebarPanel<'a> {
    /// Construct the panel from borrowed application state.
    ///
    /// All parameters map directly to the corresponding public fields.
    pub fn new(
        tree: &'a SeriesTree,
        selected_path: &'a mut Option<std::path::PathBuf>,
        active_tab: &'a mut SidebarTab,
        metadata_volume: Option<&'a LoadedVolume>,
    ) -> Self {
        Self {
            tree,
            selected_path,
            active_tab,
            metadata_volume,
        }
    }

    /// Construct the panel with an explicit tag search string (backwards-compat alias).
    pub fn with_tag_search(
        tree: &'a SeriesTree,
        selected_path: &'a mut Option<std::path::PathBuf>,
        active_tab: &'a mut SidebarTab,
        metadata_volume: Option<&'a LoadedVolume>,
        _tag_search: Option<&'a mut String>,
    ) -> Self {
        Self::new(tree, selected_path, active_tab, metadata_volume)
    }

    /// Render the panel into `ui`.
    ///
    /// Draws the tab bar at the top, then delegates to [`show_series_tab`] or
    /// [`show_metadata_tab`] depending on [`active_tab`].
    ///
    /// # Returns
    ///
    /// `Some(folder_path)` when the user clicks a series entry in the Series
    /// tab, requesting that series to be loaded.  `None` on all other frames.
    ///
    /// [`show_series_tab`]: SidebarPanel::show_series_tab
    /// [`show_metadata_tab`]: SidebarPanel::show_metadata_tab
    /// [`active_tab`]: SidebarPanel::active_tab
    pub fn show(&mut self, ui: &mut Ui) -> Option<std::path::PathBuf> {
        // ── Tab selector row ──────────────────────────────────────────────────
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
        });
        ui.separator();

        match *self.active_tab {
            SidebarTab::Series => self.show_series_tab(ui),
            SidebarTab::Metadata => {
                self.show_metadata_tab(ui);
                None
            }
        }
    }

    // ── Series tab ────────────────────────────────────────────────────────────

    /// Render the Patient → Study → Series collapsible hierarchy.
    ///
    /// # Borrow invariant
    ///
    /// `tree` is copied out of `self` (it is a `Copy` reference) before the
    /// `ScrollArea` closure begins; `selected_path` is not accessed inside the
    /// closure body.  After the closure exits, a single write to
    /// `*self.selected_path` records the newly selected path when the user
    /// clicked an entry.
    fn show_series_tab(&mut self, ui: &mut Ui) -> Option<std::path::PathBuf> {
        // Copy the &'a SeriesTree reference — &T: Copy, releases borrow on self.
        let tree: &'a SeriesTree = self.tree;

        // Clone current selection for inside-closure comparison without holding
        // a borrow on self.selected_path during the closure.
        let current_path: Option<std::path::PathBuf> = self.selected_path.as_ref().cloned();

        let mut new_selection: Option<std::path::PathBuf> = None;

        ScrollArea::vertical()
            .id_source("series_scroll")
            .show(ui, |ui| {
                if tree.patients.is_empty() {
                    ui.label("No series found.");
                    ui.label("Use File → Open DICOM folder to scan a directory.");
                    return;
                }

                for patient in &tree.patients {
                    // ── Patient node ──────────────────────────────────────────
                    let patient_label = if patient.patient_name.is_empty() {
                        format!("👤 (anonymous) [{}]", patient.patient_id)
                    } else if patient.patient_id.is_empty() {
                        format!("👤 {}", patient.patient_name)
                    } else {
                        format!("👤 {} [{}]", patient.patient_name, patient.patient_id)
                    };

                    CollapsingHeader::new(patient_label)
                        .default_open(true)
                        .show(ui, |ui| {
                            for study in &patient.studies {
                                // ── Study node ────────────────────────────────
                                let study_label = match (&study.study_date, &study.study_uid) {
                                    (Some(date), _) => format!("📅 {date}"),
                                    (None, Some(uid)) => format!("📅 UID:{uid}"),
                                    (None, None) => "📅 (unknown date)".to_string(),
                                };

                                CollapsingHeader::new(study_label).default_open(true).show(
                                    ui,
                                    |ui| {
                                        for series in &study.series {
                                            // ── Series entry ──────────────────
                                            let is_selected = current_path
                                                .as_ref()
                                                .map(|p| p == &series.folder)
                                                .unwrap_or(false);

                                            let hover_text = format!(
                                                "Folder: {}\nUID: {}\nSlices: {}",
                                                series.folder.display(),
                                                series.series_uid,
                                                series.num_slices,
                                            );

                                            let resp = ui
                                                .selectable_label(
                                                    is_selected,
                                                    series.display_label(),
                                                )
                                                .on_hover_text(hover_text);

                                            if resp.clicked() {
                                                new_selection = Some(series.folder.clone());
                                            }
                                        }
                                    },
                                );
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

    // ── Metadata (Tags) tab ───────────────────────────────────────────────────

    /// Render a scrollable table of DICOM metadata for the loaded volume.
    fn show_metadata_tab(&self, ui: &mut Ui) {
        // ── Tag search input ──────────────────────────────────────────────────
        // Uses egui's per-Id persistent storage to maintain the search string
        // across frames without requiring a field in SidebarPanel or SnapApp.
        // The Id is stable and unique to this widget.
        let search_id = egui::Id::new("dicom_tag_search_filter");
        let mut search: String = ui.data(|d| d.get_temp(search_id).unwrap_or_default());
        ui.horizontal(|ui| {
            ui.label("🔍");
            ui.text_edit_singleline(&mut search).on_hover_text(
                "Filter tags by keyword or tag hex code (case-insensitive)",
            );
            if ui.small_button("✖").clicked() {
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
                        // Row helper: label | value, then end_row.
                        let row = |ui: &mut Ui, key: &str, val: &str| {
                            ui.label(key);
                            ui.label(val);
                            ui.end_row();
                        };

                        row(ui, "Patient:", vol.patient_name.as_deref().unwrap_or("—"));
                        row(ui, "ID:", vol.patient_id.as_deref().unwrap_or("—"));
                        row(ui, "Date:", vol.study_date.as_deref().unwrap_or("—"));
                        row(ui, "Modality:", vol.modality.as_deref().unwrap_or("—"));
                        row(
                            ui,
                            "Series:",
                            vol.series_description.as_deref().unwrap_or("—"),
                        );
                        row(ui, "Dimensions:", &format!("{depth} × {rows} × {cols}"));
                        row(ui, "Spacing:", &format!("{dz:.3} × {dy:.3} × {dx:.3} mm"));
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
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// `SidebarTab::default()` must be `SidebarTab::Series`.
    ///
    /// The `#[default]` attribute on `Series` drives the derived `Default`
    /// impl; this test verifies the attribute is in place.
    #[test]
    fn test_sidebar_tab_default_is_series() {
        let tab = SidebarTab::default();
        assert_eq!(
            tab,
            SidebarTab::Series,
            "default SidebarTab must be Series, got {tab:?}"
        );
    }

    /// `SidebarTab::Series` and `SidebarTab::Metadata` must be distinct values.
    ///
    /// The two tab variants differ in their discriminant; the derived `PartialEq`
    /// and `Eq` impls must reflect that inequality.
    #[test]
    fn test_sidebar_tab_variants_are_distinct() {
        assert_ne!(
            SidebarTab::Series,
            SidebarTab::Metadata,
            "SidebarTab::Series and SidebarTab::Metadata must be distinct variants"
        );
    }

    // ── Tag filter logic ──────────────────────────────────────────────────────

    fn make_row(tag: &str, keyword: &str, value: &str) -> crate::dicom::metadata_table::MetadataRow {
        crate::dicom::metadata_table::MetadataRow {
            scope: crate::dicom::metadata_table::MetadataScope::Series,
            tag: tag.to_owned(),
            keyword: keyword.to_owned(),
            vr: "LO".to_owned(),
            value: value.to_owned(),
        }
    }

    /// Mirrors the exact filter predicate in `show_metadata_tab`.
    fn filter_rows(
        rows: &[crate::dicom::metadata_table::MetadataRow],
        needle: &str,
    ) -> Vec<crate::dicom::metadata_table::MetadataRow> {
        let needle_lc = needle.to_lowercase();
        rows.iter()
            .filter(|r| {
                r.keyword.to_lowercase().contains(&needle_lc)
                    || r.tag.to_lowercase().contains(&needle_lc)
                    || r.value.to_lowercase().contains(&needle_lc)
            })
            .cloned()
            .collect()
    }

    /// Needle "patient" (case-insensitive) must match PatientName and PatientID.
    #[test]
    fn test_tag_filter_keyword_case_insensitive() {
        let rows = vec![
            make_row("(0010,0010)", "PatientName", "Doe^John"),
            make_row("(0010,0020)", "PatientID", "MR12345"),
            make_row("(0008,0060)", "Modality", "CT"),
        ];
        let filtered = filter_rows(&rows, "patient");
        assert_eq!(filtered.len(), 2, "needle 'patient' must match PatientName and PatientID");
        assert_eq!(filtered[0].keyword, "PatientName");
        assert_eq!(filtered[1].keyword, "PatientID");
    }

    /// Needle matching a tag hex code must return the matching row.
    #[test]
    fn test_tag_filter_by_hex_tag() {
        let rows = vec![
            make_row("(0010,0010)", "PatientName", "Doe^John"),
            make_row("(0008,0060)", "Modality", "CT"),
        ];
        let filtered = filter_rows(&rows, "0008,0060");
        assert_eq!(filtered.len(), 1, "tag hex search must match exactly one row");
        assert_eq!(filtered[0].keyword, "Modality");
    }

    /// Needle matching a value must return the correct row.
    #[test]
    fn test_tag_filter_by_value() {
        let rows = vec![
            make_row("(0010,0010)", "PatientName", "Doe^John"),
            make_row("(0008,0060)", "Modality", "CT"),
            make_row("(0020,0013)", "InstanceNumber", "42"),
        ];
        let filtered = filter_rows(&rows, "doe^john");
        assert_eq!(filtered.len(), 1, "value search must match exactly one row");
        assert_eq!(filtered[0].keyword, "PatientName");
    }

    /// Needle that matches no field must return empty.
    #[test]
    fn test_tag_filter_no_match_returns_empty() {
        let rows = vec![
            make_row("(0010,0010)", "PatientName", "Doe^John"),
            make_row("(0008,0060)", "Modality", "CT"),
        ];
        let filtered = filter_rows(&rows, "xxxxnonexistent");
        assert_eq!(filtered.len(), 0, "unmatched needle must return empty");
    }

    /// Empty string needle: all rows pass through ("" is contained in every string).
    #[test]
    fn test_tag_filter_empty_needle_passes_all() {
        let rows = vec![
            make_row("(0010,0010)", "PatientName", "Doe^John"),
            make_row("(0008,0060)", "Modality", "CT"),
        ];
        let filtered = filter_rows(&rows, "");
        assert_eq!(filtered.len(), rows.len(), "empty needle: all rows pass");
    }
}
