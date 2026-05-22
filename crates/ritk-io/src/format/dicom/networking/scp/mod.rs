//! Embedded C-STORE SCP — DICOM Storage Class Provider (PS3.4 §B).
//!
//! # Overview
//!
//! [`StoreScp::start`] binds a TCP listener, spawns an accept thread, and
//! returns a [`StoreScpHandle`]. For every incoming DICOM association the SCP:
//!
//! 1. Parses A-ASSOCIATE-RQ and accepts all offered presentation contexts
//!    using the first offered transfer syntax per context.
//! 2. Sends A-ASSOCIATE-AC.
//! 3. Receives C-STORE-RQ messages; emits each as a [`StoredInstance`] on a
//!    bounded [`mpsc::sync_channel`]; responds with C-STORE-RSP Success.
//! 4. Sends A-RELEASE-RP on A-RELEASE-RQ and closes the connection.
//!
//! # Invariants
//!
//! - **Bounded memory**: the internal channel capacity is `ScpConfig::queue_capacity`.
//!   When the consumer falls behind, additional instances are acknowledged to the
//!   PACS but discarded with a `tracing::warn` event.
//! - **No async contagion**: all I/O uses blocking `std::net::TcpStream` on
//!   dedicated OS threads. The accept loop polls the shutdown flag every
//!   [`ACCEPT_POLL_INTERVAL`] to support clean termination.
//! - **Send safety**: `StoredInstance` and `ScpConfig` are `Send + Sync`.
//!   `StoreScpHandle` is `Send + !Sync` (contains `mpsc::Receiver`).

mod accept;
mod config;
mod handler;

pub use accept::StoreScp;
pub use config::{ScpConfig, StoreScpHandle, StoredInstance};

#[cfg(test)]
#[path = "../tests_scp.rs"]
mod tests_scp;

// Re-export `pad_uid` so the integration tests in `tests_scp` can access it
// via `super::pad_uid` (their `super` is the `scp` module).
#[cfg(test)]
pub(crate) use config::pad_uid;
