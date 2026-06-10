//! PACS network panel — egui UI for PACS server configuration, C-ECHO, C-FIND,
//! and C-MOVE triggers.
//!
//! # Overview
//!
//! [`show_pacs_panel`] renders the full panel into a supplied `&mut egui::Ui`
//! and returns a [`PacsPanelAction`] describing any action the user triggered
//! on this frame. All state transitions and PACS network calls are handled by
//! the caller (`SnapApp`) after the function returns.
//!
//! # Layout
//!
//! ```text
//! ┌─ Connection ──────────────────────────────────────────────────────┐
//! │ Calling AE │ [RITKSNAP ]  Called AE │ [ORTHANC   ]              │
//! │ Host       │ [localhost ]  Port     │ [4242      ]              │
//! │ Move dest  │ [RITKSNAP ]  Timeout(s)│ [30        ]              │
//! │ [Test Connection (C-ECHO)]  ● Connected (0x0000)                │
//! ├─ Query ────────────────────────────────────────────────────────────┤
//! │ Patient Name: [*        ]  Modality: [    ]                     │
//! │ [Search (C-FIND)]  [Clear]                                       │
//! ├─ Results ──────────────────────────────────────────────────────────┤
//! │ 3 result(s)                                                      │
//! │ PatientName │ Date │ Modality │ #S │ Description │               │
//! │ DOE^JOHN    │ 20240115 │ CT   │ 3  │ CHEST CT    │               │
//! │ [Retrieve (C-MOVE)]  → destination AE: RITKSNAP                 │
//! └───────────────────────────────────────────────────────────────────┘
//! ```

use crate::pacs::{AutoLoadPolicy, PacsConfig, QueryState};

mod results;

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
    SubmitFind {
        patient_name: String,
        modality: String,
        study_date: String,
        accession_number: String,
    },
    /// User pressed "Retrieve (C-MOVE)" for a specific study.
    SubmitRetrieve { study_uid: String },
    /// User pressed "Show Series" to drill into a study's series list.
    SubmitFindSeries { study_instance_uid: String },
    /// User pressed "Retrieve Series (C-MOVE)" for a specific series.
    SubmitRetrieveSeries {
        series_uid: String,
        study_uid: String,
    },
    /// User pressed "Back to studies" to return from series drill-down.
    BackToStudies,
    /// User pressed "Clear" to reset the results table.
    ClearResults,
    /// User pressed "Start SCP".
    StartScp,
    /// User pressed "Stop SCP".
    StopScp,
    /// User pressed "Load Received" to load buffered SCP instances into the viewer.
    LoadReceived,
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
/// - `study_date` — study date range filter (DICOM range format, empty = all).
/// - `accession_number` — accession number filter (exact match, empty = all).
/// - `selected_row` — index of the currently selected study result row (for C-MOVE).
/// - `selected_series_row` — index of the currently selected series result row.
/// - `study_context_uid` — StudyInstanceUID of the study being drilled into via series.
///
/// # Returns
///
/// A [`PacsPanelAction`] describing the action triggered this frame, if any.
pub fn show_pacs_panel(
    ui: &mut egui::Ui,
    config: &mut PacsConfig,
    query_state: &mut QueryState,
    echo_display: &mut str,
    patient_name: &mut String,
    modality: &mut String,
    study_date: &mut String,
    accession_number: &mut String,
    scp_listening: bool,
    scp_actual_port: u16,
    pacs_received_count: u32,
    pacs_pending_count: usize,
    pacs_auto_loaded_this_frame: Option<usize>,
    selected_row: &mut Option<usize>,
    selected_series_row: &mut Option<usize>,
    study_context_uid: &mut String,
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

            ui.label("SCP AE:");
            ui.add(egui::TextEdit::singleline(&mut config.scp_ae_title).desired_width(120.0));
            ui.label("SCP Port:");
            let mut sp = config.scp_port as i32;
            if ui
                .add(egui::DragValue::new(&mut sp).speed(1.0).range(1..=65535))
                .changed()
            {
                config.scp_port = sp as u16;
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
            ui.colored_label(color, &*echo_display);
        }
    });

    ui.horizontal(|ui| {
        if scp_listening {
            if ui.button("Stop SCP").clicked() {
                action = PacsPanelAction::StopScp;
            }
            let actual = if scp_actual_port != 0 {
                scp_actual_port
            } else {
                config.scp_port
            };
            ui.colored_label(
                egui::Color32::GREEN,
                format!("\u{25cf} SCP :{actual} (AE: {})", config.scp_ae_title),
            );
            if pacs_received_count > 0 {
                ui.label(format!("[{pacs_received_count} received]"));
            }
            let auto = config.auto_load_policy == AutoLoadPolicy::Automatic;
            let mut auto_mut = auto;
            ui.checkbox(&mut auto_mut, "Auto-load");
            if auto_mut != auto {
                config.auto_load_policy = if auto_mut {
                    AutoLoadPolicy::Automatic
                } else {
                    AutoLoadPolicy::Manual
                };
            }
            if config.auto_load_policy == AutoLoadPolicy::Automatic {
                ui.label("Limit:");
                let mut limit = config.auto_load_limit as i64;
                if ui
                    .add(
                        egui::DragValue::new(&mut limit)
                            .speed(8.0)
                            .range(1..=100_000),
                    )
                    .changed()
                {
                    config.auto_load_limit = limit as u32;
                }
            }
            let show_load_btn = config.auto_load_policy != AutoLoadPolicy::Automatic
                || pacs_pending_count > config.auto_load_limit as usize;
            if show_load_btn
                && pacs_pending_count > 0
                && ui.button("\u{25b6} Load Received").clicked()
            {
                action = PacsPanelAction::LoadReceived;
            }
            if let Some(n) = pacs_auto_loaded_this_frame {
                ui.colored_label(
                    egui::Color32::from_rgb(100, 200, 100),
                    format!("[auto-loaded {n} instances]"),
                );
            }
        } else {
            if ui.button("Start SCP").clicked() {
                action = PacsPanelAction::StartScp;
            }
            ui.weak("SCP not started");
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

            ui.label("Study Date:");
            ui.add(
                egui::TextEdit::singleline(study_date)
                    .desired_width(150.0)
                    .hint_text("YYYYMMDD-YYYYMMDD (empty=all)"),
            );
            ui.label("Accession #:");
            ui.add(
                egui::TextEdit::singleline(accession_number)
                    .desired_width(150.0)
                    .hint_text("(empty=all)"),
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
                study_date: study_date.clone(),
                accession_number: accession_number.clone(),
            };
        }
        if matches!(query_state, QueryState::Results(_) | QueryState::Error(_))
            && ui.button("Clear").clicked()
        {
            action = PacsPanelAction::ClearResults;
        }
    });

    ui.separator();

    // ── Results ───────────────────────────────────────────────────────────────
    ui.heading("Results");
    results::show_results_section(
        ui,
        config,
        query_state,
        selected_row,
        selected_series_row,
        study_context_uid,
        &mut action,
    );

    action
}
