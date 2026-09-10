//! Browser-host file handoff into the RITK viewer shell.
//!
//! Métis owns browser file handles and bounded byte transfer. This adapter
//! only converts the completed, pathless payloads into egui's input carrier so
//! the RITK drop-routing policy can classify and decode them.

/// Append completed Métis browser payloads to the egui input queue.
pub(crate) fn extend_dropped_files(dropped: &mut Vec<egui::DroppedFile>) {
    let Some(batch) = metis_web::take_file_drop() else {
        return;
    };

    dropped.extend(batch.into_files().into_vec().into_iter().map(|payload| {
        let (name, mime, bytes) = payload.into_parts();
        egui::DroppedFile {
            name,
            mime,
            bytes: Some(std::sync::Arc::from(bytes)),
            ..Default::default()
        }
    }));
}
