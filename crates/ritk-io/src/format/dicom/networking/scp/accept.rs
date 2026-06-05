//! SCP accept loop and entry-point for starting the embedded C-STORE SCP.

use super::super::association::NetworkingError;
use super::config::{ScpConfig, StoreScpHandle, ACCEPT_POLL_INTERVAL};
use super::handler::handle_connection;
use std::net::TcpListener;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};

// ── StoreScp ──────────────────────────────────────────────────────────────────

/// Embedded C-STORE SCP factory.
///
/// # Example
///
/// ```ignore
/// let handle = StoreScp::start(ScpConfig::default()).expect("SCP bind");
/// // Direct C-MOVE to handle.ae_title() @ handle.port().
/// while let Some(inst) = handle.try_recv() {
///     println!("Received {}", inst.sop_instance_uid);
/// }
/// ```
pub struct StoreScp;

impl StoreScp {
    /// Bind a TCP listener and start the SCP accept thread.
    ///
    /// # Errors
    ///
    /// Returns [`NetworkingError::Protocol`] if the TCP bind or
    /// `set_nonblocking` call fails.
    pub fn start(config: ScpConfig) -> Result<StoreScpHandle, NetworkingError> {
        let listener = TcpListener::bind(("0.0.0.0", config.port)).map_err(|e| {
            NetworkingError::Protocol(format!("SCP bind on port {}: {e}", config.port))
        })?;
        listener
            .set_nonblocking(true)
            .map_err(|e| NetworkingError::Protocol(format!("SCP set_nonblocking: {e}")))?;
        let actual_port = listener
            .local_addr()
            .map(|a| a.port())
            .unwrap_or(config.port);
        let (tx, rx) = mpsc::sync_channel::<super::config::StoredInstance>(config.queue_capacity);
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_thread = Arc::clone(&shutdown);
        let ae_title = config.ae_title;
        std::thread::spawn(move || {
            scp_accept_loop(listener, config, tx, shutdown_thread);
        });
        Ok(StoreScpHandle {
            rx,
            shutdown,
            actual_port,
            ae_title,
        })
    }
}

// ── Accept loop ───────────────────────────────────────────────────────────────

fn scp_accept_loop(
    listener: TcpListener,
    config: ScpConfig,
    tx: mpsc::SyncSender<super::config::StoredInstance>,
    shutdown: Arc<AtomicBool>,
) {
    use std::io::ErrorKind;

    loop {
        if shutdown.load(Ordering::Relaxed) {
            break;
        }
        match listener.accept() {
            Ok((stream, peer)) => {
                tracing::debug!("SCP: accepted connection from {peer}");
                let cfg = config.clone();
                let tx2 = tx.clone();
                std::thread::spawn(move || {
                    if let Err(e) = handle_connection(stream, &cfg, &tx2) {
                        tracing::warn!("SCP connection error: {e}");
                    }
                });
            }
            Err(ref e) if e.kind() == ErrorKind::WouldBlock => {
                std::thread::sleep(ACCEPT_POLL_INTERVAL);
            }
            Err(e) => {
                if !shutdown.load(Ordering::Relaxed) {
                    tracing::warn!("SCP accept error: {e}");
                }
                std::thread::sleep(ACCEPT_POLL_INTERVAL);
            }
        }
    }
    tracing::debug!("SCP accept loop exited");
}
