//! PACS network panel — egui UI for PACS server configuration, C-ECHO, C-FIND,
//! and C-MOVE triggers.
//!
//! # Overview
//!
//! [`show_pacs_panel`] renders the full panel into a supplied `&mut egui::Ui`
//! and returns a [`PacsPanelAction`] describing any action the user triggered
//! on this frame.  All state transitions and PACS network calls are handled by
//! the caller (`SnapApp`) after the function returns.
//!
//! # Layout
//!
//! ```text
//! ┌─ Connection ──────────────────────────────────────────────────────┐
//! │  Calling AE │ [RITKSNAP     ]   Called AE │ [ORTHANC       ]     │
//! │  Host       │ [localhost    ]   Port       │ [4242]               │
//! │  Move dest  │ [RITKSNAP     ]   Timeout(s) │ [30  ]               │
//! │  [Test Connection (C-ECHO)]   ● Connected (0x0000)               │
//! ├─ Query ────────────────────────────────────────────────────────────┤
//! │  Patient Name: [*         ]   Modality: [    ]                   │
//! │  [Search (C-FIND)]  [Clear]                                        │
//! ├─ Results ──────────────────────────────────────────────────────────┤
//! │  3 result(s)                                                       │
//! │  PatientName  │ Date     │ Modality │ #S │ Description            │
//! │  DOE^JOHN     │ 20240115 │ CT       │  3 │ CHEST CT               │
//! │  [Retrieve (C-MOVE)]  → destination AE: RITKSNAP                 │
//! └───────────────────────────────────────────────────────────────────┘
//! ```

use crate::pacs::{PacsConfig, QueryState};

// ── PacsPanelAction ───────────────────────────────────────────────────────────

/// Action triggered by the PACS panel on a given egui frame.
///
/// The panel is purely declarative — it returns an action and the caller
/// (`SnapApp`) performs the resulting state transition or network call.
#[derive(Debug, Default)]
pub enum PacsPanelAction {
    /// No action was triggered this frame.
    #[default]
    None,
    /// User pressed "Test Connection (C-ECHO)".
    SubmitEcho,
    /// User pressed "Search (C-FIND)".
    SubmitFind { patient_name: String, modality: String },
    /// User pressed "Retrieve (C-MOVE)" for a specific study.
    SubmitRetrieve { study_uid: String },
    /// User pressed "Clear" to reset the results table.
    ClearResults,
}

// ── show_pacs_panel ───────────────────────────────────────────────────────────

/// Render the PACS panel into `ui`.
///
/// # Parameters
///
/// - `config` — mutable reference to the current PACS server configuration.
/// - `query_state` — mutable reference to the current query state machine.
/// - `echo_display` — human-readable result of the last C-ECHO (empty if none).
/// - `patient_name` — patient name filter string (DICOM wildcard format).
/// - `modality` — modality filter (empty = all).
/// - `selected_row` — index of the currently selected result row (for C-MOVE).
///
/// # Returns
///
/// A [`PacsPanelAction`] describing the action triggered this frame, if any.
pub fn show_pacs_panel(
    ui: &mut egui::Ui,
    config: &mut PacsConfig,
    query_state: &mut QueryState,
    echo_display: &mut String,
    patient_name: &mut String,
    modality: &mut String,
    selected_row: &mut Option<usize>,
) -> PacsPanelAction {
    let mut action = PacsPanelAction::None;

    // ── Connection configuration ──────────────────────────────────────────────
    ui.heading("Connection");
    egui::Grid::new("pacs_config_grid")
        .num_columns(4)
        .spacing([6.0, 4.0])
        .show(ui, |ui| {
            ui.label("Calling AE:");
            ui.add(egui::TextEdit::singleline(&mut config.calling_ae_title).desired_width(120.0));
            ui.label("Called AE:");
            ui.add(egui::TextEdit::singleline(&mut config.called_ae_title).desired_width(120.0));
            ui.end_row();

            ui.label("Host:");
            ui.add(egui::TextEdit::singleline(&mut config.host).desired_width(120.0));
            ui.label("Port:");
            let mut port = config.port as i32;
            if ui
                .add(egui::DragValue::new(&mut port).speed(1.0).range(1..=65535))
                .changed()
            {
                config.port = port as u16;
            }
            ui.end_row();

            ui.label("Move dest AE:");
            ui.add(egui::TextEdit::singleline(&mut config.move_destination).desired_width(120.0));
            ui.label("Timeout (s):");
            let mut timeout = config.timeout_secs as i64;
            if ui
                .add(egui::DragValue::new(&mut timeout).speed(1.0).range(1..=300))
                .changed()
            {
                config.timeout_secs = timeout as u64;
            }
            ui.end_row();
        });

    ui.horizontal(|ui| {
        if ui.button("Test Connection (C-ECHO)").clicked() {
            action = PacsPanelAction::SubmitEcho;
        }
        if !echo_display.is_empty() {
            let color = if echo_display.starts_with('\u{2713}') {
                egui::Color32::GREEN
            } else {
                egui::Color32::from_rgb(220, 80, 80)
            };
            ui.colored_label(color, echo_display.as_str());
        }
    });

    ui.separator();

    // ── Query form ────────────────────────────────────────────────────────────
    ui.heading("Query");
    egui::Grid::new("pacs_query_grid")
        .num_columns(4)
        .spacing([6.0, 4.0])
        .show(ui, |ui| {
            ui.label("Patient Name:");
            ui.add(
                egui::TextEdit::singleline(patient_name)
                    .desired_width(150.0)
                    .hint_text("* (wildcard)"),
            );
            ui.label("Modality:");
            ui.add(
                egui::TextEdit::singleline(modality)
                    .desired_width(60.0)
                    .hint_text("CT, MR … (empty=all)"),
            );
            ui.end_row();
        });

    ui.horizontal(|ui| {
        let is_pending = matches!(query_state, QueryState::Pending { .. });
        if ui
            .add_enabled(!is_pending, egui::Button::new("Search (C-FIND)"))
            .clicked()
        {
            action = PacsPanelAction::SubmitFind {
                patient_name: patient_name.clone(),
                modality: modality.clone(),
            };
        }
        if matches!(query_state, QueryState::Results(_) | QueryState::Error(_)) {
            if ui.button("Clear").clicked() {
                action = PacsPanelAction::ClearResults;
            }
        }
    });

    ui.separator();

    // ── Results ───────────────────────────────────────────────────────────────
    ui.heading("Results");
    show_results_section(ui, config, query_state, selected_row, &mut action);

    action
}

// ── Results section ───────────────────────────────────────────────────────────

fn show_results_section(
    ui: &mut egui::Ui,
    config: &PacsConfig,
    query_state: &mut QueryState,
    selected_row: &mut Option<usize>,
    action: &mut PacsPanelAction,
) {
    match query_state {
        QueryState::Idle => {
            ui.weak("Enter search criteria above and press Search.");
        }
        QueryState::Pending { label } => {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(label.as_str());
            });
        }
        QueryState::Error(msg) => {
            ui.colored_label(egui::Color32::from_rgb(220, 80, 80), msg.as_str());
        }
        QueryState::Results(rows) => {
            let n = rows.len();
            ui.label(format!("{n} result(s)"));

            if n == 0 {
                ui.weak("No matching studies found.");
                return;
            }

            egui::ScrollArea::vertical()
                .id_source("pacs_results_scroll")
                .max_height(220.0)
                .show(ui, |ui| {
                    egui::Grid::new("pacs_results_grid")
                        .num_columns(6)
                        .striped(true)
                        .spacing([8.0, 3.0])
                        .show(ui, |ui| {
                            // Header row.
                            ui.strong("Patient");
                            ui.strong("Date");
                            ui.strong("Modality");
                            ui.strong("#S");
                            ui.strong("Description");
                            ui.strong("StudyUID (tail)");
                            ui.end_row();

                            for (idx, row) in rows.iter().enumerate() {
                                let is_sel = *selected_row == Some(idx);
                                let name = if row.patient_name.is_empty() {
                                    "(unknown)"
                                } else {
                                    &row.patient_name
                                };
                                if ui.selectable_label(is_sel, name).clicked() {
                                    *selected_row = Some(idx);
                                }
                                ui.label(&row.study_date);
                                ui.label(&row.modality);
                                ui.label(&row.num_series);
                                // Truncate long study descriptions with a trailing ellipsis.
                                let char_count = row.study_description.chars().count();
                                let desc: String = if char_count > 28 {
                                    format!(
                                        "{}\u{2026}",
                                        row.study_description.chars().take(27).collect::<String>()
                                    )
                                } else {
                                    row.study_description.clone()
                                };
                                ui.label(desc);
                                // Show trailing 12 chars of UID for readability.
                                let uid_tail = if row.study_instance_uid.len() > 12 {
                                    let s = &row.study_instance_uid;
                                    format!("\u{2026}{}", &s[s.len() - 12..])
                                } else {
                                    row.study_instance_uid.clone()
                                };
                                ui.label(uid_tail)
                                    .on_hover_text(&row.study_instance_uid);
                                ui.end_row();
                            }
                        });
                });

            // Retrieve button for selected row.
            if let Some(idx) = *selected_row {
                if idx < rows.len() {
                    ui.separator();
                    let study_uid = rows[idx].study_instance_uid.clone();
                    ui.horizontal(|ui| {
                        if ui.button("\u{25b6} Retrieve (C-MOVE)").clicked() {
                            *action = PacsPanelAction::SubmitRetrieve {
                                study_uid,
                            };
                        }
                        ui.weak(format!("\u{2192} destination AE: {}", config.move_destination));
                    });
                }
            }
        }
    }
}
