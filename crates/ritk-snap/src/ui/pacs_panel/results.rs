// â”€â”€ Results section â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

use super::{PacsConfig, PacsPanelAction, QueryState};

pub(super) fn show_results_section(
    ui: &mut egui::Ui,
    config: &PacsConfig,
    query_state: &mut QueryState,
    selected_row: &mut Option<usize>,
    selected_series_row: &mut Option<usize>,
    study_context_uid: &mut String,
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
                        .num_columns(7)
                        .striped(true)
                        .spacing([8.0, 3.0])
                        .show(ui, |ui| {
                            ui.strong("Patient");
                            ui.strong("Date");
                            ui.strong("Modality");
                            ui.strong("#S");
                            ui.strong("#I");
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
                                let hover = if row.patient_id.is_empty() {
                                    "PatientID: (unknown)".to_owned()
                                } else {
                                    format!("PatientID: {}", row.patient_id)
                                };
                                if ui
                                    .selectable_label(is_sel, name)
                                    .on_hover_text(hover)
                                    .clicked()
                                {
                                    *selected_row = Some(idx);
                                }
                                ui.label(&row.study_date);
                                ui.label(&row.modality);
                                ui.label(&row.num_series);
                                ui.label(&row.num_instances);

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

                                let uid_tail = if row.study_instance_uid.len() > 12 {
                                    let s = &row.study_instance_uid;
                                    format!("\u{2026}{}", &s[s.len() - 12..])
                                } else {
                                    row.study_instance_uid.clone()
                                };
                                ui.label(uid_tail).on_hover_text(&row.study_instance_uid);
                                ui.end_row();
                            }
                        });
                });

            if let Some(idx) = *selected_row {
                if idx < rows.len() {
                    ui.separator();
                    let study_uid = rows[idx].study_instance_uid.clone();
                    ui.horizontal(|ui| {
                        if ui.button("\u{25b6} Retrieve (C-MOVE)").clicked() {
                            *action = PacsPanelAction::SubmitRetrieve { study_uid };
                        }
                        ui.weak(format!(
                            "\u{2192} destination AE: {}",
                            config.move_destination
                        ));
                    });
                    ui.horizontal(|ui| {
                        if ui.button("Show Series").clicked() {
                            *action = PacsPanelAction::SubmitFindSeries {
                                study_instance_uid: rows[idx].study_instance_uid.clone() };
                        }
                        ui.weak("Drill down to see series-level details");
                    });
                }
            }
        }
        QueryState::SeriesResults {
            study_instance_uid,
            series } => {
            *study_context_uid = study_instance_uid.clone();

            ui.horizontal(|ui| {
                if ui.button("\u{2190} Back to studies").clicked() {
                    *action = PacsPanelAction::BackToStudies;
                }
                let uid_tail = if study_instance_uid.len() > 12 {
                    let s = study_instance_uid.as_str();
                    format!("\u{2026}{}", &s[s.len() - 12..])
                } else {
                    study_instance_uid.clone()
                };
                ui.label(format!("Series for study {uid_tail}"));
            });

            let n = series.len();
            ui.label(format!("{n} series result(s)"));

            if n == 0 {
                ui.weak("No series found for this study.");
                return;
            }

            egui::ScrollArea::vertical()
                .id_source("pacs_series_scroll")
                .max_height(220.0)
                .show(ui, |ui| {
                    egui::Grid::new("pacs_series_grid")
                        .num_columns(6)
                        .striped(true)
                        .spacing([8.0, 3.0])
                        .show(ui, |ui| {
                            ui.strong("#");
                            ui.strong("Modality");
                            ui.strong("Series#");
                            ui.strong("Description");
                            ui.strong("#I");
                            ui.strong("Date");
                            ui.end_row();

                            for (idx, srow) in series.iter().enumerate() {
                                let is_sel = *selected_series_row == Some(idx);
                                let snum = if srow.series_number.is_empty() {
                                    "\u{2014}".to_owned()
                                } else {
                                    srow.series_number.clone()
                                };
                                let desc = if srow.series_description.is_empty() {
                                    "(no description)".to_owned()
                                } else {
                                    let s = &srow.series_description;
                                    let chars = s.chars().count();
                                    if chars > 30 {
                                        format!(
                                            "{}\u{2026}",
                                            s.chars().take(29).collect::<String>()
                                        )
                                    } else {
                                        s.to_owned()
                                    }
                                };
                                let date_str = if srow.series_date.is_empty() {
                                    srow.series_time.clone()
                                } else {
                                    srow.series_date.clone()
                                };

                                if ui.selectable_label(is_sel, idx.to_string()).clicked() {
                                    *selected_series_row = Some(idx);
                                }
                                ui.label(&srow.modality);
                                ui.label(snum);
                                ui.label(desc).on_hover_text(&srow.series_description);
                                ui.label(&srow.num_instances);
                                ui.label(date_str);
                                ui.end_row();
                            }
                        });
                });

            if let Some(idx) = *selected_series_row {
                if idx < series.len() {
                    ui.separator();
                    let srow = &series[idx];
                    let series_uid = srow.series_instance_uid.clone();
                    let study_uid = srow.study_instance_uid.clone();
                    let label = format!(
                        "\u{25b6} Retrieve {} {} (C-MOVE)",
                        srow.modality,
                        if srow.series_number.is_empty() {
                            String::new()
                        } else {
                            format!("#{} ", srow.series_number)
                        }
                    );
                    ui.horizontal(|ui| {
                        if ui.button(label.trim()).clicked() {
                            *action = PacsPanelAction::SubmitRetrieveSeries {
                                series_uid,
                                study_uid };
                        }
                        ui.weak(format!(
                            "\u{2192} destination AE: {}",
                            config.move_destination
                        ));
                    });
                }
            }
        }
    }
}
