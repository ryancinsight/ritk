//! PACS (Picture Archiving and Communication System) networking module.
//!
//! Provides the viewer-side PACS integration layer:
//!
//! - [`PacsConfig`] — UI-facing server configuration (AE titles, host, port).
//! - [`QueryState`] — state machine driving the PACS panel (Idle/Pending/Results/Error).
//! - [`FindResultRow`] — decoded C-FIND response row.
//! - [`PacsRequest`] / [`PacsResponse`] — typed request/response for the background worker.
//! - [`PacsWorkerHandle`] — non-blocking channel handle for polling worker results.
//! - [`spawn_pacs_request`] — spawn a thread to execute a PACS request (non-WASM only).

pub mod config;
pub mod query;
pub mod worker;

pub use config::{AutoLoadPolicy, PacsConfig};
pub use query::{FindResultRow, FindResultRowSeries, PacsRequest, PacsResponse, QueryState};
#[cfg(not(target_arch = "wasm32"))]
pub use worker::spawn_pacs_request;
pub use worker::PacsWorkerHandle;

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
#[cfg(test)]
#[path = "tests_query.rs"]
mod tests_query;
