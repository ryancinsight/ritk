use super::*;
use crate::pacs::query::FindResultRowSeries;
use ritk_io::MoveResponse;

fn default_app() -> SnapApp {
    let mut app = SnapApp::default();
    app.pacs_config.calling_ae_title = "TESTAPP".into();
    app.pacs_config.host = "127.0.0.1".into();
    app
}

// â”€â”€ apply_pacs_response â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// FindSeriesOk transitions to SeriesResults and clears selection state.
#[test]
fn apply_pacs_find_series_ok_transitions_to_series_results() {
    let mut app = default_app();
    app.pacs_query_state = crate::pacs::query::QueryState::Pending {
        label: "C-FIND".into() };
    app.pacs_selected_row = Some(0);
    app.pacs_selected_series_row = Some(2);

    let series = vec![FindResultRowSeries {
        series_instance_uid: "1.2.3.4.1".into(),
        modality: "CT".into(),
        series_number: "1".into(),
        series_description: "CHEST".into(),
        num_instances: "100".into(),
        series_date: "20240101".into(),
        study_instance_uid: "STUDY-UID-1".into(),
        ..Default::default()
    }];

    app.apply_pacs_response(PacsResponse::FindSeriesOk(series));

    match &app.pacs_query_state {
        QueryState::SeriesResults {
            study_instance_uid,
            series } => {
            assert_eq!(
                study_instance_uid, "STUDY-UID-1",
                "study_instance_uid from first series"
            );
            assert_eq!(series.len(), 1, "series count");
            assert_eq!(series[0].modality, "CT", "series modality");
            assert_eq!(series[0].series_number, "1", "series number");
        }
        other => panic!("expected SeriesResults, got {other:?}") }

    assert_eq!(app.pacs_selected_row, None, "study selection cleared");
    assert_eq!(
        app.pacs_selected_series_row, None,
        "series selection cleared"
    );
}

/// FindSeriesOk with empty series list keeps empty study_instance_uid.
#[test]
fn apply_pacs_find_series_ok_empty_list_uses_default_uid() {
    let mut app = default_app();
    app.apply_pacs_response(PacsResponse::FindSeriesOk(vec![]));

    match &app.pacs_query_state {
        QueryState::SeriesResults {
            study_instance_uid,
            series } => {
            assert!(
                study_instance_uid.is_empty(),
                "empty series â†’ empty uid default"
            );
            assert!(series.is_empty(), "series list empty");
        }
        other => panic!("expected SeriesResults, got {other:?}") }
}

/// RetrieveSeriesOk transitions to Idle with status message.
#[test]
fn apply_pacs_retrieve_series_ok_sets_status_message() {
    let mut app = default_app();
    app.pacs_query_state = QueryState::Pending {
        label: "C-MOVE".into() };

    app.apply_pacs_response(PacsResponse::RetrieveSeriesOk(MoveResponse {
        completed: 42,
        failed: 0,
        warning: 0,
        final_status: 0x0000 }));

    assert!(
        matches!(app.pacs_query_state, QueryState::Idle),
        "state must be Idle after successful retrieve"
    );
    assert!(
        app.status_message.contains("42"),
        "status message must contain completed count"
    );
    assert!(
        app.status_message.contains("0x0000"),
        "status message must contain final status"
    );
}

/// RetrieveSeriesErr transitions to Error with stored error message.
#[test]
fn apply_pacs_retrieve_series_err_transitions_to_error() {
    let mut app = default_app();
    app.apply_pacs_response(PacsResponse::RetrieveSeriesErr("PACS timeout".into()));

    match &app.pacs_query_state {
        QueryState::Error(msg) => {
            assert!(msg.contains("timeout"), "error message must be stored");
        }
        other => panic!("expected Error, got {other:?}") }
}

// â”€â”€ handle_pacs_action â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// SubmitFindSeries transitions to Pending when no worker is active.
#[test]
fn handle_submit_find_series_sets_pending_state() {
    let mut app = default_app();
    app.pacs_worker = None;

    // On WASM this sets Error â€” we check the cfg-gated path at compile time.
    // On native, it spawns a worker. In both, state leaves Idle.
    app.handle_pacs_action(PacsPanelAction::SubmitFindSeries {
        study_instance_uid: "STUDY-UID".into() });

    // The state must no longer be Idle.
    assert!(
        !matches!(app.pacs_query_state, QueryState::Idle),
        "state must leave Idle after SubmitFindSeries (Pending or Error)"
    );
}

/// SubmitRetrieveSeries transitions to Pending when no worker is active.
#[test]
fn handle_submit_retrieve_series_sets_pending_state() {
    let mut app = default_app();
    app.pacs_worker = None;

    app.handle_pacs_action(PacsPanelAction::SubmitRetrieveSeries {
        study_uid: "STUDY-UID".into(),
        series_uid: "SERIES-UID".into() });

    assert!(
        !matches!(app.pacs_query_state, QueryState::Idle),
        "state must leave Idle after SubmitRetrieveSeries (Pending or Error)"
    );
}

/// SubmitFindSeries with active worker sets status_message instead.
#[test]
fn handle_submit_find_series_with_active_worker_rejects() {
    let mut app = default_app();

    // Simulate active worker by setting a non-None handle (mocked for test).
    // We construct a channel pair with a dropped sender so try_recv returns None.
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    std::mem::drop(tx);
    app.pacs_worker = Some(crate::pacs::worker::PacsWorkerHandle::for_test(rx));

    app.handle_pacs_action(PacsPanelAction::SubmitFindSeries {
        study_instance_uid: "STUDY-UID".into() });

    assert!(
        app.status_message.contains("already in progress"),
        "must reject duplicate request"
    );
}

/// BackToStudies transitions to Idle and clears selection state.
#[test]
fn handle_back_to_studies_resets_to_idle() {
    let mut app = default_app();
    app.pacs_query_state = QueryState::SeriesResults {
        study_instance_uid: "UID".into(),
        series: vec![FindResultRowSeries::default()] };
    app.pacs_selected_row = Some(1);
    app.pacs_selected_series_row = Some(0);

    app.handle_pacs_action(PacsPanelAction::BackToStudies);

    assert!(
        matches!(app.pacs_query_state, QueryState::Idle),
        "state must be Idle after BackToStudies"
    );
    assert_eq!(
        app.pacs_selected_row, None,
        "study selection must be cleared"
    );
    assert_eq!(
        app.pacs_selected_series_row, None,
        "series selection must be cleared"
    );
}

/// None action produces no state changes.
#[test]
fn handle_pacs_action_none_is_noop() {
    let mut app = default_app();
    app.pacs_query_state = QueryState::SeriesResults {
        study_instance_uid: "UID".into(),
        series: vec![FindResultRowSeries::default()] };
    app.pacs_selected_row = Some(1);
    app.pacs_selected_series_row = Some(0);

    app.handle_pacs_action(PacsPanelAction::None);

    assert!(
        matches!(app.pacs_query_state, QueryState::SeriesResults { .. }),
        "None action must not change state"
    );
    assert_eq!(
        app.pacs_selected_row,
        Some(1),
        "study selection must be preserved"
    );
    assert_eq!(
        app.pacs_selected_series_row,
        Some(0),
        "series selection must be preserved"
    );
}
