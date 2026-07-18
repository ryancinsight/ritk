//! Background PACS worker: spawns a thread per request, returns a bounded
//! response channel.
//!
//! # Threading model
//!
//! Each call to [`spawn_pacs_request`] spawns a new OS thread that runs the
//! blocking DIMSE SCU call and sends exactly one [`PacsResponse`] back through
//! a `sync_channel(1)`.  The channel capacity of 1 bounds the completed-response
//! handoff; viewer state owns at most one worker handle while a request is
//! in-flight.
//!
//! # Async-contagion prohibition
//!
//! The DIMSE SCU functions (`dicom_echo`, `dicom_find`, `dicom_retrieve`)
//! perform blocking TCP I/O via `std::net::TcpStream`.  Routing them through a
//! runtime blocking-task API or `async fn` would propagate async coloring into
//! the viewer domain.  `std::thread::spawn` keeps all callers synchronous and
//! the domain layer free of runtime coupling.
//!
//! # WASM note
//!
//! `std::net::TcpStream` is unavailable on `wasm32` targets.  The struct and
//! channel types compile on WASM; only [`spawn_pacs_request`] and the
//! execution helpers are cfg-gated to non-WASM.  The viewer's submit methods
//! set an error state immediately on WASM without attempting to spawn a thread.

use std::sync::mpsc;

use super::config::PacsConfig;
use super::query::{FindResultRow, FindResultRowSeries, PacsRequest, PacsResponse};

// â”€â”€ PacsWorkerHandle â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Handle to an in-flight PACS background operation.
///
/// The caller polls [`PacsWorkerHandle::try_recv`] on every egui frame to
/// detect completion.  Exactly one response is produced per handle.
pub struct PacsWorkerHandle {
    rx: mpsc::Receiver<PacsResponse> }

impl PacsWorkerHandle {
    /// Poll for a completed response without blocking.
    ///
    /// Returns `Some(response)` when the worker has finished; `None` while the
    /// operation is still running.  After `Some` is returned the handle is
    /// exhausted â€” no further responses will arrive.
    pub fn try_recv(&self) -> Option<PacsResponse> {
        self.rx.try_recv().ok()
    }

    /// Construct a handle from an existing receiver (test support).
    #[cfg(test)]
    pub fn for_test(rx: mpsc::Receiver<PacsResponse>) -> Self {
        Self { rx }
    }
}

// â”€â”€ spawn_pacs_request â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Spawn a background thread to execute `request` against `config`.
///
/// Returns a [`PacsWorkerHandle`] whose [`PacsWorkerHandle::try_recv`]
/// produces exactly one [`PacsResponse`] when the request completes or fails.
///
/// Not available on `wasm32` targets (no TCP stack).
#[cfg(not(target_arch = "wasm32"))]
pub fn spawn_pacs_request(config: PacsConfig, request: PacsRequest) -> PacsWorkerHandle {
    let (tx, rx) = mpsc::sync_channel::<PacsResponse>(1);
    std::thread::spawn(move || {
        let resp = execute_request(&config, request);
        send_worker_response(tx, resp);
    });
    PacsWorkerHandle { rx }
}

#[cfg(not(target_arch = "wasm32"))]
fn send_worker_response(tx: mpsc::SyncSender<PacsResponse>, resp: PacsResponse) -> bool {
    match tx.send(resp) {
        Ok(()) => true,
        Err(_) => {
            tracing::debug!("PACS worker response receiver dropped before completion");
            false
        }
    }
}

// â”€â”€ Execution helpers (non-WASM only) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[cfg(not(target_arch = "wasm32"))]
fn execute_request(config: &PacsConfig, request: PacsRequest) -> PacsResponse {
    match request {
        PacsRequest::Echo => execute_echo(config),
        PacsRequest::FindStudies {
            patient_name,
            modality,
            study_date,
            accession_number } => execute_find(
            config,
            &patient_name,
            &modality,
            &study_date,
            &accession_number,
        ),
        PacsRequest::RetrieveStudy {
            study_instance_uid,
            move_destination } => execute_retrieve(config, &study_instance_uid, &move_destination),
        PacsRequest::FindSeries { study_instance_uid } => {
            execute_find_series(config, &study_instance_uid)
        }
        PacsRequest::RetrieveSeries {
            study_instance_uid,
            series_instance_uid,
            move_destination } => execute_retrieve_series(
            config,
            &study_instance_uid,
            &series_instance_uid,
            &move_destination,
        ) }
}

#[cfg(not(target_arch = "wasm32"))]
fn execute_echo(config: &PacsConfig) -> PacsResponse {
    use ritk_io::dicom_echo;
    let assoc_cfg = config.to_association_config();
    match dicom_echo(&assoc_cfg) {
        Ok(rsp) => PacsResponse::EchoOk { status: rsp.status },
        Err(e) => PacsResponse::EchoErr(e.to_string()) }
}

#[cfg(not(target_arch = "wasm32"))]
fn execute_find(
    config: &PacsConfig,
    patient_name: &str,
    modality: &str,
    study_date: &str,
    accession_number: &str,
) -> PacsResponse {
    use ritk_io::{dicom_find, FindResult};
    let assoc_cfg = config.to_association_config();
    let query =
        FindResultRow::build_study_query(patient_name, modality, study_date, accession_number);
    match dicom_find(&assoc_cfg, &query) {
        Ok(raw_results) => {
            let rows: Vec<FindResultRow> = raw_results
                .iter()
                .flat_map(|r: &FindResult| r.matches.iter())
                .map(|bytes| FindResultRow::from_raw_bytes(bytes))
                .collect();
            PacsResponse::FindOk(rows)
        }
        Err(e) => PacsResponse::FindErr(e.to_string()) }
}

#[cfg(not(target_arch = "wasm32"))]
fn execute_find_series(config: &PacsConfig, study_instance_uid: &str) -> PacsResponse {
    use ritk_io::{dicom_find, FindResult};
    let assoc_cfg = config.to_association_config();
    let query = FindResultRowSeries::build_series_query(study_instance_uid);
    match dicom_find(&assoc_cfg, &query) {
        Ok(raw_results) => {
            let rows: Vec<FindResultRowSeries> = raw_results
                .iter()
                .flat_map(|r: &FindResult| r.matches.iter())
                .map(|bytes| FindResultRowSeries::from_raw_bytes(bytes))
                .collect();
            PacsResponse::FindSeriesOk(rows)
        }
        Err(e) => PacsResponse::FindErr(e.to_string()) }
}

#[cfg(not(target_arch = "wasm32"))]
fn execute_retrieve(config: &PacsConfig, study_uid: &str, move_destination: &str) -> PacsResponse {
    use ritk_io::{dicom_retrieve, AeTitle, MoveDestination};
    let assoc_cfg = config.to_association_config();
    let dest_ae = match AeTitle::new(move_destination) {
        Ok(ae) => ae,
        Err(e) => return PacsResponse::RetrieveErr(e.to_string()) };
    let destination = MoveDestination::new(dest_ae);
    match dicom_retrieve(&assoc_cfg, &destination, study_uid) {
        Ok(rsp) => PacsResponse::RetrieveOk(rsp),
        Err(e) => PacsResponse::RetrieveErr(e.to_string()) }
}

#[cfg(not(target_arch = "wasm32"))]
fn execute_retrieve_series(
    config: &PacsConfig,
    study_uid: &str,
    series_uid: &str,
    move_destination: &str,
) -> PacsResponse {
    use ritk_io::{dicom_retrieve_series, AeTitle, MoveDestination};
    let assoc_cfg = config.to_association_config();
    let dest_ae = match AeTitle::new(move_destination) {
        Ok(ae) => ae,
        Err(e) => return PacsResponse::RetrieveSeriesErr(e.to_string()) };
    let destination = MoveDestination::new(dest_ae);
    match dicom_retrieve_series(&assoc_cfg, &destination, study_uid, series_uid) {
        Ok(rsp) => PacsResponse::RetrieveSeriesOk(rsp),
        Err(e) => PacsResponse::RetrieveSeriesErr(e.to_string()) }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn send_worker_response_reports_delivered_response() {
        let (tx, rx) = mpsc::sync_channel(1);

        assert!(send_worker_response(tx, PacsResponse::EchoOk { status: 0 }));
        assert!(matches!(
            rx.try_recv(),
            Ok(PacsResponse::EchoOk { status: 0 })
        ));
    }

    #[test]
    fn send_worker_response_reports_dropped_receiver() {
        let (tx, rx) = mpsc::sync_channel(1);
        drop(rx);

        assert!(!send_worker_response(
            tx,
            PacsResponse::EchoErr("dropped".to_owned())
        ));
    }
}
