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
        // Clear auto-loaded notification from the previous frame.
        self.pacs_auto_loaded_this_frame = None;

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
            PacsResponse::FindSeriesOk(series) => {
                let n = series.len();
                let study_uid = series
                    .first()
                    .map(|s| s.study_instance_uid.clone())
                    .unwrap_or_default();
                let msg = format!("PACS C-FIND series: {n} result(s)");
                self.status_message = msg.clone();
                info!("{}", msg);
                self.pacs_query_state = QueryState::SeriesResults {
                    study_instance_uid: study_uid,
                    series };
                self.pacs_selected_row = None;
                self.pacs_selected_series_row = None;
            }
            PacsResponse::RetrieveErr(e) => {
                let msg = format!("PACS C-MOVE failed: {e}");
                self.status_message = msg.clone();
                error!("{}", msg);
                self.pacs_query_state = QueryState::Error(e);
            }
            PacsResponse::RetrieveSeriesOk(rsp) => {
                let msg = format!(
                    "PACS C-MOVE series: completed={} failed={} warning={} status=0x{:04X}",
                    rsp.completed, rsp.failed, rsp.warning, rsp.final_status
                );
                self.status_message = msg.clone();
                info!("{}", msg);
                self.pacs_query_state = QueryState::Idle;
            }
            PacsResponse::RetrieveSeriesErr(e) => {
                let msg = format!("PACS C-MOVE series failed: {e}");
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
            PacsPanelAction::SubmitFind {
                patient_name,
                modality,
                study_date,
                accession_number } => {
                self.submit_pacs_find(patient_name, modality, study_date, accession_number);
            }
            PacsPanelAction::SubmitRetrieve { study_uid } => {
                self.submit_pacs_retrieve(study_uid);
            }
            PacsPanelAction::SubmitFindSeries { study_instance_uid } => {
                self.submit_pacs_find_series(study_instance_uid);
            }
            PacsPanelAction::SubmitRetrieveSeries {
                series_uid,
                study_uid } => {
                self.submit_pacs_retrieve_series(study_uid, series_uid);
            }
            PacsPanelAction::BackToStudies => {
                self.pacs_query_state = QueryState::Idle;
                self.pacs_selected_row = None;
                self.pacs_selected_series_row = None;
            }
            PacsPanelAction::ClearResults => {
                self.pacs_query_state = QueryState::Idle;
                self.pacs_selected_row = None;
                self.pacs_selected_series_row = None;
            }
            PacsPanelAction::StartScp => {
                #[cfg(not(target_arch = "wasm32"))]
                self.start_pacs_scp();
            }
            PacsPanelAction::StopScp => {
                #[cfg(not(target_arch = "wasm32"))]
                self.stop_pacs_scp();
            }
            PacsPanelAction::LoadReceived => {
                #[cfg(not(target_arch = "wasm32"))]
                self.load_received_scp_instances();
            }
        }
    }

    // â”€â”€ Submit helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    pub(crate) fn submit_pacs_echo(&mut self) {
        if self.pacs_worker.is_some() {
            self.status_message = "PACS: a request is already in progress.".to_owned();
            return;
        }
        self.pacs_query_state = QueryState::Pending {
            label: "C-ECHOâ€¦".to_owned() };

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
            self.pacs_query_state =
                QueryState::Error("PACS networking is not available in browser builds.".to_owned());
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
        self.pacs_query_state = QueryState::Pending {
            label: "C-FINDâ€¦".to_owned() };
        self.pacs_selected_row = None;

        #[cfg(not(target_arch = "wasm32"))]
        {
            use crate::pacs::spawn_pacs_request;
            self.pacs_worker = Some(spawn_pacs_request(
                self.pacs_config.clone(),
                PacsRequest::FindStudies {
                    patient_name,
                    modality,
                    study_date,
                    accession_number },
            ));
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (patient_name, modality, study_date, accession_number);
            self.pacs_query_state =
                QueryState::Error("PACS networking is not available in browser builds.".to_owned());
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
        self.pacs_query_state = QueryState::Pending {
            label: format!("C-MOVE â†’ {move_destination}â€¦") };

        #[cfg(not(target_arch = "wasm32"))]
        {
            use crate::pacs::spawn_pacs_request;
            self.pacs_worker = Some(spawn_pacs_request(
                self.pacs_config.clone(),
                PacsRequest::RetrieveStudy {
                    study_instance_uid: study_uid,
                    move_destination },
            ));
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (study_uid, move_destination);
            self.pacs_query_state =
                QueryState::Error("PACS networking is not available in browser builds.".to_owned());
        }
    }

    pub(crate) fn submit_pacs_find_series(&mut self, study_instance_uid: String) {
        if self.pacs_worker.is_some() {
            self.status_message = "PACS: a request is already in progress.".to_owned();
            return;
        }
        self.pacs_query_state = QueryState::Pending {
            label: "C-FIND series\u{2026}".to_owned() };
        self.pacs_selected_series_row = None;

        #[cfg(not(target_arch = "wasm32"))]
        {
            use crate::pacs::spawn_pacs_request;
            self.pacs_worker = Some(spawn_pacs_request(
                self.pacs_config.clone(),
                PacsRequest::FindSeries { study_instance_uid },
            ));
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = study_instance_uid;
            self.pacs_query_state =
                QueryState::Error("PACS networking is not available in browser builds.".to_owned());
        }
    }

    pub(crate) fn submit_pacs_retrieve_series(&mut self, study_uid: String, series_uid: String) {
        if self.pacs_worker.is_some() {
            self.status_message = "PACS: a request is already in progress.".to_owned();
            return;
        }
        #[cfg(not(target_arch = "wasm32"))]
        self.start_pacs_scp();
        let move_destination = self.pacs_config.move_destination.clone();
        self.pacs_query_state = QueryState::Pending {
            label: format!("C-MOVE series â†’ {move_destination}\u{2026}") };

        #[cfg(not(target_arch = "wasm32"))]
        {
            use crate::pacs::spawn_pacs_request;
            self.pacs_worker = Some(spawn_pacs_request(
                self.pacs_config.clone(),
                PacsRequest::RetrieveSeries {
                    study_instance_uid: study_uid,
                    series_instance_uid: series_uid,
                    move_destination },
            ));
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (study_uid, series_uid, move_destination);
            self.pacs_query_state =
                QueryState::Error("PACS networking is not available in browser builds.".to_owned());
        }
    }

    /// Poll the embedded SCP for received instances on every egui frame.
    ///
    /// Drains all buffered instances from the bounded channel into
    /// `pacs_pending_instances`. When `pacs_config.auto_load_policy` is
    /// [`AutoLoadPolicy::Automatic`] and instances were received this frame,
    /// automatically triggers loading into the viewer.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn poll_pacs_scp(&mut self) {
        let Some(handle) = &self.pacs_scp_handle else {
            return;
        };
        let had_pending = !self.pacs_pending_instances.is_empty();
        while let Some(inst) = handle.try_recv() {
            self.pacs_received_count = self.pacs_received_count.saturating_add(1);
            tracing::info!(
                sop_instance_uid = %inst.sop_instance_uid,
                sop_class_uid = %inst.sop_class_uid,
                bytes = inst.dataset_bytes.len(),
                "SCP received instance",
            );
            self.pacs_pending_instances.push(inst);
        }
        let pending_count = self.pacs_pending_instances.len();
        if self.pacs_config.auto_load_policy == crate::pacs::config::AutoLoadPolicy::Automatic
            && !had_pending
            && !self.pacs_pending_instances.is_empty()
        {
            if pending_count <= self.pacs_config.auto_load_limit as usize {
                tracing::info!("auto-load: triggering load of SCP-received instances");
                self.pacs_auto_loaded_this_frame = Some(pending_count);
                self.load_received_scp_instances();
            } else {
                let limit = self.pacs_config.auto_load_limit;
                self.status_message = format!(
                    "Auto-load suppressed: {pending_count} instances exceed limit ({limit}). Click \u{25b6} Load Received to proceed."
                );
                tracing::info!("auto-load suppressed: {pending_count} pending > limit {limit}");
            }
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
            ae_title: self
                .pacs_config
                .scp_ae_title
                .as_str()
                .try_into()
                .unwrap_or_else(|_| ritk_io::literal_arraystring("RITKSNAP")),
            port: self.pacs_config.scp_port,
            ..ScpConfig::default()
        };
        match StoreScp::start(config) {
            Ok(handle) => {
                let port = handle.port();
                let ae = handle.ae_title().to_owned();
                self.pacs_scp_handle = Some(handle);
                self.pacs_received_count = 0;
                self.pacs_pending_instances.clear();
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

    /// Load all buffered SCP-received instances into the viewer.
    ///
    /// Parses each [`StoredInstance`] in-memory via the zero-disk DICOM
    /// loading path (Part 10 bytes â†’ `scan_dicom_instances` â†’ pixel decode).
    /// On success, the received instances buffer and counter are cleared.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn load_received_scp_instances(&mut self) {
        if self.pacs_pending_instances.is_empty() {
            self.status_message = "No received DICOM instances to load.".to_owned();
            return;
        }
        let count = self.pacs_pending_instances.len();
        self.status_message = format!("Loading {count} received DICOM instance(s)â€¦");
        match crate::dicom::loader::load_dicom_series_from_stored_instances(
            &self.pacs_pending_instances,
        ) {
            Ok(vol) => {
                let shape = vol.shape;
                let msg = format!(
                    "Loaded {count} SCP-received instance(s) â€” shape {:?}",
                    shape
                );
                self.load_volume(vol, msg);
                self.pacs_pending_instances.clear();
                self.pacs_received_count = 0;
            }
            Err(e) => {
                let msg = format!("Failed to load SCP-received instances: {e:#}");
                self.status_message = msg.clone();
                tracing::error!("{}", msg);
            }
        }
    }
}

// â”€â”€ Tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#[cfg(test)]
#[path = "tests_pacs_ops.rs"]
mod tests;
