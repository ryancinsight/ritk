//! PACS networking operations for `SnapApp`.
//!
//! Provides methods to submit PACS requests, poll the background worker,
//! and apply worker responses to the viewer state.

use super::state::SnapApp;
use crate::pacs::query::{PacsRequest, PacsResponse, QueryState};
use crate::ui::pacs_panel::PacsPanelAction;
use tracing::{error, info};

impl SnapApp {
    /// Poll the background PACS worker for a completed response.
    ///
    /// Must be called every egui frame (registered in the `update` loop) so
    /// that completed responses are applied promptly even while the PACS panel
    /// is closed.
    pub(crate) fn poll_pacs_worker(&mut self) {
        if let Some(worker) = &self.pacs_worker {
            if let Some(resp) = worker.try_recv() {
                self.pacs_worker = None;
                self.apply_pacs_response(resp);
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        self.poll_pacs_scp();
    }

    /// Apply a completed [`PacsResponse`] to the viewer state.
    fn apply_pacs_response(&mut self, resp: PacsResponse) {
        match resp {
            PacsResponse::EchoOk { status } => {
                self.pacs_echo_display = format!("\u{2713} C-ECHO OK (0x{:04X})", status);
                self.pacs_query_state = QueryState::Idle;
                let msg = format!("PACS C-ECHO succeeded (status 0x{:04X})", status);
                self.status_message = msg.clone();
                info!("{}", msg);
            }
            PacsResponse::EchoErr(e) => {
                self.pacs_echo_display = format!("\u{2717} {}", e);
                self.pacs_query_state = QueryState::Idle;
                let msg = format!("PACS C-ECHO failed: {e}");
                self.status_message = msg.clone();
                error!("{}", msg);
            }
            PacsResponse::FindOk(rows) => {
                let n = rows.len();
                let msg = format!("PACS C-FIND: {n} result(s)");
                self.status_message = msg.clone();
                info!("{}", msg);
                self.pacs_query_state = QueryState::Results(rows);
                self.pacs_selected_row = None;
            }
            PacsResponse::FindErr(e) => {
                let msg = format!("PACS C-FIND failed: {e}");
                self.status_message = msg.clone();
                error!("{}", msg);
                self.pacs_query_state = QueryState::Error(e);
            }
            PacsResponse::RetrieveOk(rsp) => {
                let msg = format!(
                    "PACS C-MOVE: completed={} failed={} warning={} status=0x{:04X}",
                    rsp.completed, rsp.failed, rsp.warning, rsp.final_status
                );
                self.status_message = msg.clone();
                info!("{}", msg);
                self.pacs_query_state = QueryState::Idle;
            }
            PacsResponse::RetrieveErr(e) => {
                let msg = format!("PACS C-MOVE failed: {e}");
                self.status_message = msg.clone();
                error!("{}", msg);
                self.pacs_query_state = QueryState::Error(e);
            }
        }
    }

    /// Handle a [`PacsPanelAction`] returned from the PACS panel UI.
    ///
    /// Dispatches to the appropriate submit method or applies the action
    /// directly (e.g. clearing results).
    pub(crate) fn handle_pacs_action(&mut self, action: PacsPanelAction) {
        match action {
            PacsPanelAction::None => {}
            PacsPanelAction::SubmitEcho => self.submit_pacs_echo(),
            PacsPanelAction::SubmitFind { patient_name, modality, study_date, accession_number } => {
                self.submit_pacs_find(patient_name, modality, study_date, accession_number);
            }
            PacsPanelAction::SubmitRetrieve { study_uid } => {
                self.submit_pacs_retrieve(study_uid);
            }
            PacsPanelAction::ClearResults => {
                self.pacs_query_state = QueryState::Idle;
                self.pacs_selected_row = None;
            }
            PacsPanelAction::StartScp => {
                #[cfg(not(target_arch = "wasm32"))]
                self.start_pacs_scp();
            }
            PacsPanelAction::StopScp => {
                #[cfg(not(target_arch = "wasm32"))]
                self.stop_pacs_scp();
            }
        }
    }

    // ── Submit helpers ────────────────────────────────────────────────────────

    pub(crate) fn submit_pacs_echo(&mut self) {
        if self.pacs_worker.is_some() {
            self.status_message = "PACS: a request is already in progress.".to_owned();
            return;
        }
        self.pacs_query_state = QueryState::Pending { label: "C-ECHO…".to_owned() };

        #[cfg(not(target_arch = "wasm32"))]
        {
            use crate::pacs::spawn_pacs_request;
            self.pacs_worker = Some(spawn_pacs_request(
                self.pacs_config.clone(),
                PacsRequest::Echo,
            ));
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.pacs_query_state = QueryState::Error(
                "PACS networking is not available in browser builds.".to_owned(),
            );
        }
    }

    pub(crate) fn submit_pacs_find(
        &mut self,
        patient_name: String,
        modality: String,
        study_date: String,
        accession_number: String,
    ) {
        if self.pacs_worker.is_some() {
            self.status_message = "PACS: a request is already in progress.".to_owned();
            return;
        }
        self.pacs_query_state = QueryState::Pending { label: "C-FIND…".to_owned() };
        self.pacs_selected_row = None;

        #[cfg(not(target_arch = "wasm32"))]
        {
            use crate::pacs::spawn_pacs_request;
            self.pacs_worker = Some(spawn_pacs_request(
                self.pacs_config.clone(),
                PacsRequest::FindStudies { patient_name, modality, study_date, accession_number },
            ));
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (patient_name, modality, study_date, accession_number);
            self.pacs_query_state = QueryState::Error(
                "PACS networking is not available in browser builds.".to_owned(),
            );
        }
    }

    pub(crate) fn submit_pacs_retrieve(&mut self, study_uid: String) {
        if self.pacs_worker.is_some() {
            self.status_message = "PACS: a request is already in progress.".to_owned();
            return;
        }
        // Auto-start embedded SCP so it is ready to receive C-STORE sub-operations.
        #[cfg(not(target_arch = "wasm32"))]
        self.start_pacs_scp();
        let move_destination = self.pacs_config.move_destination.clone();
        self.pacs_query_state = QueryState::Pending { label: format!("C-MOVE → {move_destination}…") };

        #[cfg(not(target_arch = "wasm32"))]
        {
            use crate::pacs::spawn_pacs_request;
            self.pacs_worker = Some(spawn_pacs_request(
                self.pacs_config.clone(),
                PacsRequest::RetrieveStudy { study_instance_uid: study_uid, move_destination },
            ));
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (study_uid, move_destination);
            self.pacs_query_state = QueryState::Error(
                "PACS networking is not available in browser builds.".to_owned(),
            );
        }
    }

    /// Poll the embedded SCP for received instances on every egui frame.
    ///
    /// Drains all buffered instances from the bounded channel into
    /// `pacs_received_count`.  Future work: queue instances for loading.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn poll_pacs_scp(&mut self) {
        let Some(handle) = &self.pacs_scp_handle else { return };
        while let Some(inst) = handle.try_recv() {
            self.pacs_received_count = self.pacs_received_count.saturating_add(1);
            tracing::info!(
                sop_instance_uid = %inst.sop_instance_uid,
                sop_class_uid = %inst.sop_class_uid,
                bytes = inst.dataset_bytes.len(),
                "SCP received instance",
            );
        }
    }

    /// Start the embedded C-STORE SCP if not already running.
    ///
    /// Uses `pacs_config.scp_ae_title` and `pacs_config.scp_port`.
    /// No-op if SCP is already running.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn start_pacs_scp(&mut self) {
        use ritk_io::{ScpConfig, StoreScp};
        if self.pacs_scp_handle.is_some() {
            return;
        }
        let config = ScpConfig {
            ae_title: self.pacs_config.scp_ae_title.clone(),
            port: self.pacs_config.scp_port,
            ..ScpConfig::default()
        };
        match StoreScp::start(config) {
            Ok(handle) => {
                let port = handle.port();
                let ae = handle.ae_title().to_owned();
                self.pacs_scp_handle = Some(handle);
                self.pacs_received_count = 0;
                let msg = format!("SCP started: AE={ae} port={port}");
                self.status_message = msg.clone();
                info!("{}", msg);
            }
            Err(e) => {
                let msg = format!("SCP start failed: {e}");
                self.status_message = msg.clone();
                error!("{}", msg);
            }
        }
    }

    /// Stop the embedded C-STORE SCP if running.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn stop_pacs_scp(&mut self) {
        if let Some(handle) = self.pacs_scp_handle.take() {
            let msg = format!(
                "SCP stopped (received {} instance(s))",
                self.pacs_received_count
            );
            handle.stop();
            self.status_message = msg.clone();
            info!("{}", msg);
        }
    }
}
